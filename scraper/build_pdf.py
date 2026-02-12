"""INTERNECHIVE PDF BUILDER — Convert captured pages to PDF.

Usage:
    python build_pdf.py <tar_file> [output_name]

Examples:
    python build_pdf.py ~/Downloads/newtoughnesstrai0000loeh_pages.tar
    python build_pdf.py ~/Downloads/mybook_pages.tar "My Book"

Outputs:
    <output_name>.pdf    — Full PDF of all pages
    pages/               — Individual JPGs (useful for LLM transcription)
"""
import glob
import os
import sys
import tarfile

from PIL import Image


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_pdf.py <tar_file> [output_name]")
        print("  tar_file:    path to the .tar from capture_pages.js")
        print("  output_name: optional name for the PDF (default: from tar filename)")
        sys.exit(1)

    tar_path = sys.argv[1]

    # Handle Windows paths from WSL
    if not os.path.exists(tar_path):
        # Try common download locations
        for prefix in ["/mnt/c/Users/sam/Downloads/", os.path.expanduser("~/Downloads/")]:
            candidate = os.path.join(prefix, os.path.basename(tar_path))
            if os.path.exists(candidate):
                tar_path = candidate
                break

    if not os.path.exists(tar_path):
        print(f"Error: cannot find {sys.argv[1]}")
        sys.exit(1)

    # Determine output name
    if len(sys.argv) >= 3:
        output_name = sys.argv[2]
    else:
        base = os.path.basename(tar_path).replace("_pages.tar", "").replace(".tar", "")
        output_name = base

    output_dir = "pages"
    pdf_file = f"{output_name}.pdf"

    # Extract
    print(f"Extracting {tar_path}...")
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(tar_path) as tf:
        tf.extractall(output_dir)

    image_files = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
    print(f"Extracted {len(image_files)} images to {output_dir}/")

    if not image_files:
        print("No images found!")
        sys.exit(1)

    # Build PDF
    print(f"Building {pdf_file}...")
    pages = []
    for f in image_files:
        img = Image.open(f).convert("RGB")
        pages.append(img)

    pages[0].save(pdf_file, save_all=True, append_images=pages[1:], resolution=150)

    for img in pages:
        img.close()

    size_mb = os.path.getsize(pdf_file) / 1024 / 1024
    print(f"Done! {pdf_file} ({size_mb:.1f} MB, {len(image_files)} pages)")
    print(f"Individual images in {output_dir}/")


if __name__ == "__main__":
    main()
