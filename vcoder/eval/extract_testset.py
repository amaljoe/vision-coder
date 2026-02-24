"""Extract Design2Code testset from HuggingFace parquet into individual files.

The SALT-NLP/Design2Code-hf dataset parquet has 'image' (PNG bytes dict) and
'text' (HTML) columns. This script extracts them into individual files:
  testset_final/0.png, testset_final/0.html
  testset_final/1.png, testset_final/1.html
  ...

Usage:
    python vcoder/eval/extract_testset.py \\
        --parquet_path ../Design2Code/testset_final/data/train-00000-of-00001.parquet \\
        --rick_jpg ../Design2Code/testset_final/rick.jpg \\
        --output_dir ../Design2Code/testset_final_extracted
"""

import argparse
import io
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_path",
        default="/home/compiling-ganesh/24m0797/workspace/Design2Code/testset_final/data/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--rick_jpg",
        default="/home/compiling-ganesh/24m0797/workspace/Design2Code/testset_final/rick.jpg",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/compiling-ganesh/24m0797/workspace/Design2Code/testset_final_extracted",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Copy rick.jpg (used as placeholder in HTML files)
    rick_src = Path(args.rick_jpg)
    if rick_src.exists():
        shutil.copy(rick_src, out / "rick.jpg")
        print(f"Copied rick.jpg to {out}")
    else:
        print(f"WARNING: rick.jpg not found at {rick_src}")

    print(f"Loading parquet: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)
    print(f"  {len(df)} examples, columns: {df.columns.tolist()}")

    existing = len(list(out.glob("*.html")))
    if existing > 0:
        print(f"  {existing} HTML files already extracted, skipping existing.")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        html_path = out / f"{idx}.html"
        png_path = out / f"{idx}.png"

        # Extract HTML
        if not html_path.exists():
            html_path.write_text(row["text"], encoding="utf-8")

        # Extract image (stored as dict with 'bytes' key in parquet)
        if not png_path.exists():
            img_data = row["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            else:
                img = img_data  # already a PIL Image
            img.save(png_path, format="PNG")

    html_count = len(list(out.glob("*.html")))
    png_count = len(list(out.glob("*.png")))
    print(f"\nExtracted to {out}")
    print(f"  HTML files : {html_count}")
    print(f"  PNG  files : {png_count}")
    print(f"  rick.jpg   : {(out / 'rick.jpg').exists()}")


if __name__ == "__main__":
    main()
