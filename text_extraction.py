import PyPDF2
import joblib
import pandas as pd

# Đọc văn bản từ file PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n".join(text_parts)

        # Nếu PyPDF2 không trích được văn bản (ví dụ PDF là ảnh), thử OCR nếu thư viện có sẵn
        if not text.strip():
            try:
                from pdf2image import convert_from_path
                import pytesseract
                images = convert_from_path(file_path)
                ocr_parts = [pytesseract.image_to_string(img) for img in images]
                text = "\n".join(ocr_parts)
            except Exception:
                # Nếu không có pdf2image/pytesseract hoặc OCR thất bại, giữ text rỗng
                text = ""

        if text:
            return text.lower()
        else:
            return "Lỗi: Không thể trích xuất văn bản từ PDF (file có thể là scan hoặc định dạng không hỗ trợ)."
    except Exception as e:
        return f"Lỗi: Không thể đọc file PDF ({e})"
