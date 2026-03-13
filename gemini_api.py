import os
import json
import google.generativeai as genai


# ==============================
# Load API KEY
# ==============================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=api_key)


# ==============================
# Model candidates
# ==============================

MODEL_CANDIDATES = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
]


# ==============================
# Clean JSON response
# ==============================

def clean_json(text):

    text = text.strip()

    if "```json" in text:
        text = text.split("```json")[1]

    if "```" in text:
        text = text.split("```")[0]

    return text.strip()


# ==============================
# Main function
# ==============================

def get_structured_data(pdf_text):

    prompt = f"""
Bạn là chuyên gia HR AI.

Hãy phân tích CV và trả về JSON duy nhất theo format:

{{
"name": "",
"skills": "Python, TensorFlow, SQL",
"education": "B.Tech | B.Sc | M.Tech | MBA | PhD",
"certifications": "Google ML | AWS Certified | Deep Learning Specialization | None",
"job_role": "AI Researcher | Data Scientist | Cybersecurity Analyst | Software Engineer",
"experience_years": number,
"projects": number
}}
*lưu ý cho experience_years: nếu trong CV viết từ năm bất kì đến năm bất kì hãy tính toán để đưa ra số năm đúng, trường hợp có ghi rõ số năm thì không cần tính toán mà lấy luôn số liệu
*lưu ý cho projects: hãy đếm số lượng project có trong CV, nếu CV ghi rõ số lượng thì ko cần đếm mà lấy luôn số liệu
Chỉ trả JSON. Không giải thích.

CV:
{pdf_text}
"""

    last_exception = None

    for candidate in MODEL_CANDIDATES:

        print(f"Trying model: {candidate}")

        try:

            model = genai.GenerativeModel(candidate)

            response = model.generate_content(prompt)

            cleaned = clean_json(response.text)

            data = json.loads(cleaned)

            print(f"Model success: {candidate}")

            return data

        except Exception as e:

            last_exception = e
            msg = str(e).lower()

            if "not found" in msg or "unsupported" in msg:
                continue

            print("Gemini parsing error:", e)

    raise RuntimeError(
        f"Không gọi được Gemini API. Lỗi cuối: {last_exception}"
    )