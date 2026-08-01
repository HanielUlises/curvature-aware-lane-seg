"""Export the segmenter to ONNX and measure what the runtimes actually buy.

The geometry chain runs in C++ at 191 microseconds a frame while the network takes 8,263,
so the network is the whole budget and the only thing worth optimizing. This exports it
once and benchmarks the runtimes that can consume the result, so the choice is made on
measurements rather than on reputation.

Parity is checked before any timing. A faster runtime that produces a different mask is
not faster, it is broken, and the tolerance is stated in terms the rest of the project
uses: agreement on the thresholded mask, since that is what the chain consumes, not on
the logits.

    python -m scripts.export_onnx infer.ckpt=<checkpoint>
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.subset import read_manifest
from src.models.lane_segmenter import LaneSegmenter
from scripts.infer_video import _resolve_ckpt

OPSET = 17


def _bench(run, warmup: int = 10, iters: int = 60) -> tuple[float, float]:
    """Mean and p95 microseconds per call."""
    for _ in range(warmup):
        run()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return sum(times) / len(times), times[int(0.95 * (len(times) - 1))]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import onnx
    import onnxruntime as ort

    target = tuple(cfg.data.target_size)
    out_dir = Path(cfg.paths.output_root) / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "lane_segmenter.onnx"

    _, meta = read_manifest(Path(cfg.paths.output_root) / "manifests" / "val.json")
    ckpt = _resolve_ckpt(cfg)
    model = LaneSegmenter.load_from_checkpoint(
        str(ckpt), bin_edges=meta["bin_edges"], map_location="cpu"
    ).eval()
    print(f"exporting {ckpt}")

    dummy = torch.randn(1, 3, target[1], target[0])
    # Only the batch axis is dynamic. Height and width are fixed by the preprocessing,
    # and pinning them lets a runtime specialize the kernels for this exact shape.
    torch.onnx.export(
        model.net, dummy, str(onnx_path), opset_version=OPSET,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"wrote {onnx_path} ({size_mb:.1f} MB, opset {OPSET})")

    # Parity, on real-shaped input, against the mask the chain would consume.
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((1, 3, target[1], target[0])).astype(np.float32)
    with torch.no_grad():
        torch_logits = model.net(torch.from_numpy(sample)).numpy()

    providers_available = ort.get_available_providers()
    results = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_model = model.net.to(device).eval()
    gpu_input = torch.from_numpy(sample).to(device)

    def run_torch():
        with torch.no_grad():
            gpu_model(gpu_input)
        if device == "cuda":
            torch.cuda.synchronize()

    results.append(("PyTorch " + device, *_bench(run_torch)))

    # Half precision, which for a convolutional net on a tensor-core GPU is usually the
    # cheapest real win and needs no export at all. Parity is checked on the mask, since
    # fp16 moves logits by far more than the runtimes do and the question is only whether
    # the thresholded mask the chain consumes still agrees.
    if device == "cuda":
        half_model = model.net.to(device).half().eval()
        half_input = gpu_input.half()

        def run_half():
            with torch.no_grad():
                half_model(half_input)
            torch.cuda.synchronize()

        with torch.no_grad():
            half_logits = half_model(half_input).float().cpu().numpy()
        agree = float(np.mean((half_logits >= 0.0) == (torch_logits >= 0.0)))
        print(f"{'PyTorch fp16':<18} mask agreement {100 * agree:.4f}%  "
              f"max logit delta {np.abs(half_logits - torch_logits).max():.4g}")
        results.append(("PyTorch fp16", *_bench(run_half)))
        # Put the model back so the ONNX comparison below is against fp32.
        model.net.float()
        gpu_model = model.net.to(device).eval()

    for provider in ("CPUExecutionProvider", "CUDAExecutionProvider",
                     "TensorrtExecutionProvider"):
        if provider not in providers_available:
            print(f"skipping {provider}: not available")
            continue
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider_opts = [{}]
        if provider == "TensorrtExecutionProvider":
            # Cache the built engine: the first build takes minutes and would otherwise
            # be paid on every process start, which is not what a deployment does.
            cache = out_dir / "trt_cache"
            cache.mkdir(exist_ok=True)
            provider_opts = [{
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
                "trt_fp16_enable": True,
            }]
        try:
            sess = ort.InferenceSession(
                str(onnx_path), opts, providers=[provider],
                provider_options=provider_opts,
            )
        except Exception as exc:  # a provider can be listed but fail to initialize
            print(f"skipping {provider}: {type(exc).__name__}: {str(exc)[:120]}")
            continue

        # onnxruntime falls back to CPU when a provider fails to initialize, and only
        # warns. Without this check the CPU timings get reported under the accelerator's
        # name, which is worse than having no number at all.
        actual = sess.get_providers()
        if actual[0] != provider:
            print(f"skipping {provider}: it fell back to {actual[0]}, so any timing "
                  "would be that runtime under this one's name")
            continue

        name = sess.get_inputs()[0].name
        out = sess.run(None, {name: sample})[0]
        # The chain consumes a thresholded mask, so that is what parity is measured on.
        agree = float(
            np.mean((out >= 0.0) == (torch_logits >= 0.0))
        )
        max_abs = float(np.abs(out - torch_logits).max())
        label = provider.replace("ExecutionProvider", "")
        if provider == "TensorrtExecutionProvider":
            label += " fp16"
        print(f"{label:<18} mask agreement {100 * agree:.4f}%  "
              f"max logit delta {max_abs:.4g}")
        results.append((f"ONNX {label}", *_bench(lambda: sess.run(None, {name: sample}))))

    print(f"\n{'runtime':<22}{'mean us':>10}{'p95 us':>10}{'fps':>8}{'vs torch':>10}")
    baseline = results[0][1]
    for name, mean, p95 in results:
        print(f"{name:<22}{mean:>10.0f}{p95:>10.0f}{1e6 / mean:>8.0f}"
              f"{baseline / mean:>9.2f}x")

    print("\nThe geometry chain is 191 us/frame, so compare that against the numbers "
          "above to see what share of the budget is left for it.")


if __name__ == "__main__":
    main()
