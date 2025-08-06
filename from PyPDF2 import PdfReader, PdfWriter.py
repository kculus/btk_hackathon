#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract

# ←— İşlenecek PDF dosyalarını buraya girin:
PDF_FILES = [
    "7 - Matematik - MEB.pdf"
    # "dosya2.pdf",
]

# OCR için dil kodu:
OCR_LANG = "tur"
# Görüntü DPI ayarı:
OCR_DPI  = 300

# Eğer tesseract.exe farklı bir klasördeyse buraya tam yolunu yazın:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Tesseract'ın tessdata klasörünü gösterin (bir üst klasörü de olabilir):
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

def pdf_to_text(input_pdf: str, output_txt: str, lang: str, dpi: int):
    print(f"[📄] İşleniyor: {input_pdf}")
    text = extract_text(input_pdf)
    if len(text.strip()) > 100:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  → Metin katmanından alındı: {output_txt}")
    else:
        print("  (!) Metin katmanı yetersiz, OCR’a geçiliyor…")
        pages = convert_from_path(input_pdf, dpi=dpi)
        with open(output_txt, "w", encoding="utf-8") as f:
            for i, page in enumerate(pages, start=1):
                f.write(f"--- Sayfa {i} ---\n")
                f.write(pytesseract.image_to_string(page, lang=lang))
                f.write("\n\n")
        print(f"  → OCR ile yazıldı: {output_txt}")

def main():
    if not PDF_FILES:
        print("❌ Lütfen en üstte PDF_FILES listesine işlenecek dosyaları girin.", file=sys.stderr)
        sys.exit(1)

    for pdf in PDF_FILES:
        if not os.path.isfile(pdf):
            print(f"[Hata] Dosya bulunamadı: {pdf}", file=sys.stderr)
            continue
        base, _ = os.path.splitext(pdf)
        out_txt = base + ".txt"
        try:
            pdf_to_text(pdf, out_txt, OCR_LANG, OCR_DPI)
        except Exception as e:
            print(f"[Hata] '{pdf}' işlenirken: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
