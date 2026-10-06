"""Re-create data/raw/manual_ocr.txt from the manual PDF (needs poppler + tesseract).

The PDF is image-based, so OCR is needed. This is a ONE-TIME offline step: the curated,
structured text lives in data/ncair_manual.md and the app never runs OCR at start-up.
After re-running, compare the OCR output with ncair_manual.md and update it by hand.
"""
import glob, os, subprocess, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pdf = os.path.join(ROOT, "data", "NCAIR LMS ONBOARDING MANUAL.pdf")
out = os.path.join(ROOT, "data", "raw", "manual_ocr.txt")
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["pdftoppm", "-r", "110", "-png", pdf, os.path.join(tmp, "p")], check=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, img in enumerate(sorted(glob.glob(os.path.join(tmp, "p-*.png"))), 1):
            text = subprocess.run(["tesseract", img, "-"], capture_output=True, text=True).stdout
            f.write(f"===== PAGE {i} =====\n{text}\n")
print("wrote", out)
