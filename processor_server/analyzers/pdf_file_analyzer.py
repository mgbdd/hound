from pathlib import Path
from tempfile import TemporaryDirectory
from processor_server.analyzers.image_file_analyzer import image_file_analyzer
import io

import fitz
import pytesseract
from PIL import Image


def _extract_page_text_with_ocr_fallback(page) -> str:
    """Use embedded PDF text layer; fallback to OCR for scanned pages."""
    page_text = (page.get_text() or "").strip()
    if page_text:
        return page_text

    # No text layer detected -> render page and run OCR.
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ocr_text = pytesseract.image_to_string(image, lang="rus+eng")
    return (ocr_text or "").strip()


def pdf_file_analyzer(file_path, img_limit=None):
    text_out = []
    img_count = 0
    src = Path(file_path)

    with TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        doc = fitz.open(str(src))

        for page_num, page in enumerate(doc, start=1):
            page_text = _extract_page_text_with_ocr_fallback(page)
            if page_text:
                text_out.append(page_text)

            for img_index, img in enumerate(page.get_images(full=True), start=1):
                if img_limit is not None and img_count >= img_limit:
                    break

                xref = img[0]
                img_dict = doc.extract_image(xref)
                ext = img_dict["ext"].lower()
                if ext not in {"jpg", "jpeg", "png", "gif"}:
                    continue

                img_bytes = img_dict["image"]
                img_name = f"page{page_num}_img{img_count + 1}.{ext}"
                img_path = tmpdir / img_name
                img_path.write_bytes(img_bytes)

                alt = image_file_analyzer(img_path) or "нет описания"
                text_out.append(f"\n![image {img_count + 1}]({img_name}): {alt}")

                img_count += 1

            if img_limit is not None and img_count >= img_limit:
                break

        doc.close()

    return "\n\n".join(t for t in text_out if t)
