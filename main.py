import joblib
import pandas as pd
import os

from save_data_to_sql import save_cv_to_sql
from text_extraction import extract_text_from_pdf
from apply_GeminiAPI import get_structured_data
from mapping import transform_to_numeric    

def main(pdf_file_path):

    if not os.path.isfile(pdf_file_path):
        print(f"Lỗi: File không tồn tại: {pdf_file_path}")
        return None

    try:
        print(f"Đang trích xuất văn bản từ: {pdf_file_path}...")
        raw_text = extract_text_from_pdf(pdf_file_path)
    except Exception as e:
        print(f"Lỗi khi trích xuất văn bản: {e}")
        return None

    if isinstance(raw_text, str) and raw_text.startswith("Lỗi:"):
        print(raw_text)
        return None

    print("Đang gửi văn bản tới Gemini API...")
    extracted_json = get_structured_data(raw_text)

    if not extracted_json:
        print("Không lấy được dữ liệu từ Gemini.")
        return None

    try:
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        model = joblib.load('cv_scoring_model.pkl')
    except Exception as e:
        print(f"Lỗi load model: {e}")
        return None

    features_for_model = [
        c for c in feature_columns 
        if c not in ['Score (0-100)', 'Recruiter Decision']
    ]

    try:
        input_vector = transform_to_numeric(
            extracted_json,
            features_for_model,
            scaler
        )

        input_vector = input_vector[features_for_model]
        score = model.predict(input_vector)
        score_value = float(score[0])

    except Exception as e:
        print(f"Lỗi mapping/predict: {e}")
        import pandas as pd
        import numpy as np

        input_vector = pd.DataFrame(
            [[0]*len(features_for_model)],
            columns=features_for_model
        )

        try:
            input_scaled = scaler.transform(input_vector.values)
            input_vector = pd.DataFrame(
                input_scaled,
                columns=features_for_model
            )
        except:
            pass

        score = model.predict(input_vector)
        score_value = float(score[0])

    # ✅ LUÔN LUÔN SAVE DATABASE Ở ĐÂY
    try:
        save_cv_to_sql(extracted_json, score_value, pdf_file_path)
        print("Đã lưu vào database.")
    except Exception as e:
        print(f"Lỗi khi lưu DB: {e}")

    return score_value


# Chạy chương trình chính
if __name__ == "__main__":
    # Tên file CV cần chấm điểm
    target_cv = "CV(PDF)/AI CV 3.pdf"
    
    if os.path.exists(target_cv):
        main(target_cv)
    else:
        print(f"Vui lòng đặt file '{target_cv}' vào thư mục để bắt đầu chấm điểm.")