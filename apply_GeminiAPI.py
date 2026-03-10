import os
import json
import google.generativeai as genai

# Prefer reading API key from environment for safety; fall back to existing key if unset
api_key = os.getenv('GENAI_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
else:
    genai.configure(api_key="AIzaSyAacogP_OOp8Kwm6qHBeBthPWU8SBBIpFA")

# Candidate model names to try (some environments name models differently)
MODEL_CANDIDATES = [
    'models/gemini-2.5-flash',
    'models/gemini-flash-latest',
    'models/gemini-2.0-flash',
    'models/gemini-2.0-flash-001',
    'models/gemini-2.5-flash-lite',
]

def get_structured_data(pdf_text):
    prompt = f"""
    Bạn là một trợ lý AI chuyên về nhân sự. Hãy phân tích văn bản CV sau và trích xuất thông tin 
    để khớp với các danh mục dữ liệu của tôi.
    
    YÊU CẦU TRẢ VỀ ĐỊNH DẠNG JSON THUẦN TÚY:
    0. Name: Tên đầy đủ của ứng viên (Ví dụ: "Nguyễn Văn A"). Nếu không tìm thấy, hãy để null.
    1. Skills: Danh sách kỹ năng (ví dụ: ["Python", "TensorFlow", "React"]).
    2. Experience: Số năm kinh nghiệm (số nguyên).
    3. Education: Chỉ chọn một trong: "B.Sc", "B.Tech", "M.Tech", "MBA", "PhD".
    4. Certifications: Chỉ chọn một trong: "Google ML", "AWS Certified", "Deep Learning Specialization", "None".
    5. Job_Role: Chỉ chọn: "AI Researcher", "Data Scientist", "Cybersecurity Analyst", "Software Engineer".
    6. Projects_Count: Số lượng dự án (số nguyên).
    7. Salary_Expectation: Kỳ vọng lương (số nguyên).
    
    Văn bản CV:
    {pdf_text}
    """
    
    last_exception = None
    for candidate in MODEL_CANDIDATES:
        print(f"Trying model candidate: {candidate}")
        try:
            model = genai.GenerativeModel(candidate)
            response = model.generate_content(prompt)
            # Làm sạch chuỗi trả về tránh lỗi JSON
            json_data = response.text.replace('```json', '').replace('```', '').strip()
            print(f"Model succeeded: {candidate}")
            return json.loads(json_data)
        except Exception as e:
            last_exception = e
            msg = str(e).lower()
            # If model not found/unsupported, try next candidate; otherwise re-raise
            if 'not found' in msg or 'unsupported' in msg or 'is not found' in msg:
                continue
            raise

    # Nếu không model nào thành công, nâng lỗi rõ hơn
    raise RuntimeError(f"Không thể gọi API với các model thử nghiệm. Lỗi cuối: {last_exception}")