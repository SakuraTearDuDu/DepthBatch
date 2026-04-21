from __future__ import annotations

import argparse
import hashlib
import ssl
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/"
    "resolve/main/depth_anything_v2_vits.pth?download=true"
)
EXPECTED_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official DA-V2 Small checkpoint.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/weights/depth_anything_v2_vits.pth"),
        help="Destination checkpoint path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination even if it already exists.",
    )
    return parser.parse_args()


def stream_download(url: str, output_path: Path) -> str:
    context = ssl.create_default_context()
    request = urllib.request.Request(url, headers={"User-Agent": "depthbatch/0.1.0a0"})
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        final_url = response.geturl()
    final_request = urllib.request.Request(final_url, headers={"User-Agent": "depthbatch/0.1.0a0"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(final_request, timeout=300, context=context) as response:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    if output_path.exists() and not args.force:
        actual = hashlib.sha256(output_path.read_bytes()).hexdigest()
        if actual != EXPECTED_SHA256:
            print(
                f"Existing file hash mismatch for {output_path}. "
                "Use --force to replace it with the official DA-V2 Small checkpoint.",
                file=sys.stderr,
            )
            return 1
        print(output_path)
        print(f"sha256={actual}")
        return 0

    actual = stream_download(MODEL_URL, output_path)
    if actual != EXPECTED_SHA256:
        print(
            f"Downloaded checkpoint hash mismatch. expected={EXPECTED_SHA256} actual={actual}",
            file=sys.stderr,
        )
        return 1

    print(output_path)
    print(f"sha256={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
