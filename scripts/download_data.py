"""Download and unpack the CurveLanes dataset.

This script is intentionally NOT run automatically. CurveLanes is ~150k frames
and several GB; the maintainer runs it by hand after supplying the current
download links.

CurveLanes is distributed by the official repo as a single ``.tar.gz`` split
into ordered parts:

    https://github.com/SoulmateB/CurveLanes

There are three source modes:

* ``--kaggle OWNER/SLUG`` — the easiest path. Pulls the single dataset zip from
  the Kaggle mirror (e.g. ``bnyadmohammed/curvelanes``) over authenticated HTTP,
  stdlib only. Needs a free Kaggle account and API token (``~/.kaggle/kaggle.json``
  or ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` env vars). This is an *unofficial*
  mirror — verify the label schema before trusting it downstream.
* ``--part`` as a **plain HTTP(S) URL** — stdlib only, resumable. For an
  institutional mirror or a bucket you control.
* ``--part`` as a **Google Drive** share link / bare file id — needs ``gdown``.

``--part`` values are downloaded in the order given, then concatenated and
extracted. ``--kaggle`` and ``--part`` are mutually exclusive.

Usage
-----
    # Kaggle mirror (easiest)
    python -m scripts.download_data --dest data/raw --kaggle bnyadmohammed/curvelanes

    # Plain HTTP mirror
    python -m scripts.download_data \
        --dest data/raw \
        --part https://example.org/curvelanes.tar.gz.part00 \
        --part https://example.org/curvelanes.tar.gz.part01

    # Bare Google Drive ids (requires: pip install gdown)
    python -m scripts.download_data --dest data/raw \
        --part <file-id-0> --part <file-id-1>

Optionally verify each downloaded part against a known sha256:

    python -m scripts.download_data --dest data/raw \
        --part https://example.org/part00=<sha256> \
        --part https://example.org/part01=<sha256>

After extraction the tree is::

    <dest>/CurveLanes/
        train/  images/  labels/  train.txt
        valid/  images/  labels/  valid.txt
        test/   images/          test.txt
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

_DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}
_USER_AGENT = "curvature-aware-lane-seg/download_data"
_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB


def _parse_part(spec: str) -> tuple[str, str | None]:
    """Split a ``source[=sha256]`` spec into (source, sha256|None).

    ``source`` is a plain URL, a Google Drive link, or a bare Drive file id.
    A ``sha256`` is only appended when the source itself contains no ``=``
    (Drive ids and typical URLs do not), so ``rpartition`` disambiguates.
    """
    source, sep, sha = spec.rpartition("=")
    if not sep:
        return spec.strip(), None
    return source.strip(), (sha.strip().lower() or None)


def _is_drive(source: str) -> bool:
    """True if the source should be fetched via Google Drive rather than HTTP."""
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return parsed.netloc in _DRIVE_HOSTS
    # No scheme => treat as a bare Drive file id.
    return True


def _sha256(path: Path, chunk: int = _DOWNLOAD_CHUNK) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_http(url: str, out: Path, extra_headers: dict[str, str] | None = None) -> None:
    """Stream ``url`` to ``out`` over HTTP(S), resuming a partial file if any."""
    tmp = out.with_suffix(out.suffix + ".partial")
    existing = tmp.stat().st_size if tmp.exists() else 0

    headers = {"User-Agent": _USER_AGENT, **(extra_headers or {})}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"[http] resuming {out.name} at {existing} bytes")
    else:
        print(f"[http] {url} -> {out}")

    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - http(s) enforced by caller
    try:
        response = urllib.request.urlopen(request)  # noqa: S310
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # Range not satisfiable => already complete.
            tmp.replace(out)
            print(f"[http] {out.name} already complete")
            return
        raise

    # If the server ignored our Range and sent 200, start over from scratch.
    mode = "ab" if (existing and response.status == 206) else "wb"
    if mode == "wb":
        existing = 0

    with response, tmp.open(mode) as dst:
        while True:
            block = response.read(_DOWNLOAD_CHUNK)
            if not block:
                break
            dst.write(block)
    tmp.replace(out)


def _download_drive(source: str, out: Path) -> None:
    """Fetch a Google Drive file (share link or bare id) via gdown."""
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "gdown is required for Google Drive sources. Either install it with\n"
            "    pip install gdown\n"
            "or pass a plain HTTP(S) mirror URL instead."
        ) from exc

    print(f"[drive] {source} -> {out}")
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        gdown.download(url=source, output=str(out), quiet=False, fuzzy=True)
    else:
        gdown.download(id=source, output=str(out), quiet=False)


def _download_part(source: str, out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name} already present")
        return
    if _is_drive(source):
        _download_drive(source, out)
    else:
        _download_http(source, out)


def _reassemble(parts: list[Path], archive: Path) -> None:
    """Concatenate the ordered parts into a single tarball."""
    if archive.exists():
        print(f"[skip] {archive.name} already reassembled")
        return
    print(f"[join] {len(parts)} parts -> {archive.name}")
    with archive.open("wb") as dst:
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, dst, length=_DOWNLOAD_CHUNK)


def _kaggle_credentials() -> tuple[str, str]:
    """Read Kaggle API credentials from env vars or ``~/.kaggle/kaggle.json``."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    config = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if config.is_file():
        data = json.loads(config.read_text())
        username, key = data.get("username"), data.get("key")
        if username and key:
            return username, key

    raise SystemExit(
        "No Kaggle credentials found. Create an API token at "
        "https://www.kaggle.com/settings (Account -> Create New Token), then save it "
        "to ~/.kaggle/kaggle.json, or export KAGGLE_USERNAME and KAGGLE_KEY."
    )


def _download_kaggle(slug: str, out: Path) -> None:
    """Download a Kaggle dataset (``owner/slug``) zip over authenticated HTTP."""
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name} already present")
        return
    if slug.count("/") != 1 or not all(slug.split("/")):
        raise SystemExit(f"--kaggle expects 'owner/slug', got {slug!r}")

    username, key = _kaggle_credentials()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    token = base64.b64encode(f"{username}:{key}".encode()).decode()
    print(f"[kagl] {slug} -> {out}")
    # Reuse the resumable HTTP downloader, injecting Basic auth.
    _download_http(url, out, extra_headers={"Authorization": f"Basic {token}"})


def _extract_archive(archive: Path, dest: Path) -> None:
    """Extract a tar or zip ``archive`` into ``dest``, rejecting path traversal."""
    dest_root = dest.resolve()

    def _is_safe(name: str) -> bool:
        return str((dest_root / name).resolve()).startswith(str(dest_root))

    if zipfile.is_zipfile(archive):
        print(f"[zip ] extracting {archive.name} -> {dest}")
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if not _is_safe(name):
                    raise RuntimeError(f"Refusing unsafe zip member: {name!r}")
            zf.extractall(dest)  # noqa: S202 - members validated above
        return

    print(f"[tar ] extracting {archive.name} -> {dest}")
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            if not _is_safe(member.name):
                raise RuntimeError(f"Refusing unsafe tar member: {member.name!r}")
        tar.extractall(dest)  # noqa: S202 - members validated above


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Directory to download into and extract under.",
    )
    parser.add_argument(
        "--kaggle",
        metavar="OWNER/SLUG",
        help="Kaggle dataset mirror to pull, e.g. bnyadmohammed/curvelanes.",
    )
    parser.add_argument(
        "--part",
        action="append",
        dest="parts",
        default=[],
        metavar="URL_OR_ID[=SHA256]",
        help="HTTP(S) URL, Drive link, or Drive id for one tarball part, in order. Repeatable.",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Do not delete the downloaded parts and joined archive after extraction.",
    )
    args = parser.parse_args(argv)

    if bool(args.kaggle) == bool(args.parts):
        parser.error(
            "Supply exactly one source: --kaggle OWNER/SLUG, or one or more --part links. "
            "For part links, see https://github.com/SoulmateB/CurveLanes."
        )

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    if args.kaggle:
        archive = dest / "curvelanes.zip"
        _download_kaggle(args.kaggle, archive)
        if not archive.exists() or archive.stat().st_size == 0:
            print(f"[fail] Kaggle download produced no data at {archive}", file=sys.stderr)
            return 1
        _extract_archive(archive, dest)
        if not args.keep_parts:
            archive.unlink(missing_ok=True)
            print("[rm  ] removed downloaded zip (--keep-parts to retain)")
        print(f"[done] CurveLanes extracted under {dest}")
        return 0

    downloaded: list[Path] = []
    for index, spec in enumerate(args.parts):
        source, expected_sha = _parse_part(spec)
        out = dest / f"curvelanes.tar.gz.part{index:02d}"
        _download_part(source, out)
        if not out.exists() or out.stat().st_size == 0:
            print(f"[fail] part {index} did not download to {out}", file=sys.stderr)
            return 1
        if expected_sha is not None:
            actual = _sha256(out)
            if actual != expected_sha:
                print(
                    f"[fail] sha256 mismatch for {out.name}:\n"
                    f"       expected {expected_sha}\n       actual   {actual}",
                    file=sys.stderr,
                )
                return 1
            print(f"[ok  ] sha256 verified for {out.name}")
        downloaded.append(out)

    archive = dest / "curvelanes.tar.gz"
    _reassemble(downloaded, archive)
    _extract_archive(archive, dest)

    if not args.keep_parts:
        for part in downloaded:
            part.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)
        print("[rm  ] removed parts and joined archive (--keep-parts to retain)")

    print(f"[done] CurveLanes extracted under {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
