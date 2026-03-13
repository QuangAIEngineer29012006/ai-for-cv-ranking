import os
import json

from text_extraction import extract_text_from_pdf
from gemini_api import get_structured_data
from ranking import hybrid_score
from save_data_to_sql import save_cv_to_sql


# ======================================
# Convert Gemini JSON → format ranking.py
# ======================================

def convert_to_ranking_format(data):

    skills = data.get("skills", "")

    # nếu Gemini trả list
    if isinstance(skills, list):
        skills = ", ".join(skills)

    cv_dict = {
        "Skills": skills,
        "Education": data.get("education", ""),
        "Certifications": data.get("certifications", ""),
        "Job Role": data.get("job_role", ""),
        "Experience (Years)": data.get("experience_years", 0),
        "Projects Count": data.get("projects", 0)
    }

    return cv_dict


# ======================================
# Main pipeline
# ======================================

def main(pdf_file_path):

    # ---------------------------
    # Check file
    # ---------------------------

    if not os.path.isfile(pdf_file_path):
        print(f"Lỗi: File không tồn tại: {pdf_file_path}")
        return None


    # ---------------------------
    # Extract text
    # ---------------------------

    try:

        print(f"\nĐang trích xuất văn bản từ: {pdf_file_path}")
        raw_text = extract_text_from_pdf(pdf_file_path)

    except Exception as e:

        print(f"Lỗi khi trích xuất PDF: {e}")
        return None


    if not raw_text:
        print("Không đọc được nội dung CV.")
        return None


    # ---------------------------
    # Gemini parsing
    # ---------------------------

    print("\nĐang gửi văn bản tới Gemini API...")

    try:

        extracted_json = get_structured_data(raw_text)

    except Exception as e:

        print(f"Lỗi Gemini API: {e}")
        return None


    if not extracted_json:
        print("Gemini không trả về dữ liệu.")
        return None


    # =============================
    # Print JSON
    # =============================

    print("\n==============================")
    print("JSON TRÍCH XUẤT TỪ CV")
    print("==============================")

    print(json.dumps(extracted_json, indent=4, ensure_ascii=False))


    # ---------------------------
    # Convert format
    # ---------------------------

    try:

        cv_dict = convert_to_ranking_format(extracted_json)

    except Exception as e:

        print(f"Lỗi convert JSON: {e}")
        return None


    # ---------------------------
    # Ranking
    # ---------------------------

    try:

        score_info = hybrid_score(cv_dict)

        print("\n========== AI SCORING RESULT ==========")

        print(f"\nFinal Score        : {score_info['final_score']:.1f}")
        print(f"Random Forest Score: {score_info['rf_score']:.1f}")
        print(f"Semantic Similarity: {score_info['semantic_score']:.1f}")

        print("\nReasons:")

        if score_info["reasons"]:
            for r in score_info["reasons"]:
                print("✔", r)
        else:
            print("Không có lý do nổi bật.")

        score_value = round(score_info["final_score"], 1)

    except Exception as e:

        print(f"Lỗi khi tính ranking: {e}")
        score_value = 0


    # ---------------------------
    # Save database
    # ---------------------------

    try:

        save_cv_to_sql(
            extracted_json,
            score_value,
            pdf_file_path
        )

        print("\nĐã lưu CV vào database.")

    except Exception as e:

        print(f"Lỗi khi lưu database: {e}")


    return score_value


# ======================================
# Run program
# ======================================