"""Export the segmenter to ONNX, which is the last thing this project asks Python to do.

The export has to happen here: it reads a PyTorch checkpoint, and only PyTorch can. But
it is a build step, run once per trained model, and what it produces is consumed
entirely by the C++ deployment path — the session, the execution providers, the
preprocessing and the geometry all run there now.

So this script no longer benchmarks anything. It used to, and the numbers were
misleading in a way that was hard to see: they timed onnxruntime through its Python
bindings and compared the result against a PyTorch baseline the deployed path does not
contain, while leaving out the preprocessing the deployed path does contain. The
measurement belongs where the code runs:

    deploy/build/bench_backends --model <onnx> --image <frame.jpg>   # provider choice
    deploy/build/run_infer --model <onnx> --source <clip>            # the whole pipeline

What this does check is that the exported graph is the model: ONNX Runtime's CPU
provider, which involves no fp16 engine and no fused kernels, has to produce the same
thresholded mask as PyTorch. That is a property of the export, so it is verified at the
point of export. Whether the *deployed* path agrees is a different question, answered
end to end by scripts/check_native_parity.py.

    python -m scripts.export_onnx infer.ckpt=<checkpoint>
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.subset import read_manifest
from src.models.lane_segmenter import LaneSegmenter
from scripts.infer_video import _resolve_ckpt

OPSET = 17


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
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB, opset {OPSET})")

    # Does the graph still compute the model? Checked on the mask rather than the
    # logits, because the mask is what the chain consumes and a logit tolerance would
    # either be loose enough to prove nothing or tight enough to fail on a difference
    # the mask cannot see.
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((1, 3, target[1], target[0])).astype(np.float32)
    with torch.no_grad():
        want = model.net(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {session.get_inputs()[0].name: sample})[0]
    agreement = float(np.mean((got >= 0.0) == (want >= 0.0)))
    print(f"CPU provider vs PyTorch: mask agreement {100 * agreement:.4f}%, "
          f"max logit delta {np.abs(got - want).max():.4g}")
    if agreement < 1.0:
        # Nothing between the two should move a pixel across the threshold. If one
        # moved, the export changed the model and no amount of runtime tuning downstream
        # will put it back.
        raise SystemExit(
            f"export is not faithful: {(1 - agreement) * want.size:.0f} pixels disagree"
        )

    print("\nMeasure it where it runs:\n"
          f"  deploy/build/bench_backends --model {onnx_path} --image <frame.jpg>\n"
          f"  deploy/build/run_infer --model {onnx_path} --source <clip-or-video> \\\n"
          "      --calibration <calibration.json> --backend tensorrt --cache <dir>")


if __name__ == "__main__":
    main()
