from pathlib import Path

from depthbatch.api import infer_images


def main() -> None:
    result = infer_images(
        backend_name="fake",
        input_path=Path("tests/fixtures/images"),
        output_root=Path("runs/example-fake"),
        save_raw=True,
    )
    print(result.manifest_path)


if __name__ == "__main__":
    main()
