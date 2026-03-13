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

        # Nếu PyPDF2 không trích được văn bản (chứng tỏ là PDF ảnh)
        if not text.strip():
            raise ValueError("File PDF ảnh không được hỗ trợ, reject ngay lập tức.")

        return text.lower()
    except Exception as e:
        raise ValueError(f"Không thể đọc file PDF: {e}")
