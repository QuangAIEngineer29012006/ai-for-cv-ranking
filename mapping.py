import pandas as pd
import joblib

def transform_to_numeric(extracted_json, feature_columns, scaler):
    """
    Chuyển đổi dữ liệu JSON trích xuất từ LLM sang định dạng DataFrame số hóa
    phù hợp với đầu vào của mô hình Machine Learning.
    """
    # Tạo 1 Dataframe trắng với các cột đã có trong file tien_xu_li.ipynb
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Ánh xạ các giá trị trực tiếp
    input_df['Experience (Years)'] = extracted_json['Experience']
    input_df['Salary Expectation ($)'] = extracted_json['Salary_Expectation']
    input_df['Projects Count'] = extracted_json['Projects_Count']
    
    # Ánh xạ education (giống mapping trong notebook)
    edu_map = {
        'B.Sc': 1,
        'B.Tech': 2,
        'M.Tech': 3,
        'MBA': 4,        
        'PhD': 5
    }
    input_df['Education'] = edu_map.get(extracted_json['Education'], 1)
    
    # Ánh xạ Skills (One-Hot)
    # LLM trả về danh sách, ta duyệt danh sách đó và đánh dấu 1 vào cột tương ứng
    for skill in extracted_json['Skills']:
        if skill in input_df.columns:
            input_df[skill] = 1
        for col in input_df.columns:
            if skill.lower() == col.lower():
                input_df[col] = 1

    # Ánh xạ Job Role và Certifications (Dạng Role_... và Cert_...)
    role_col = f"Role_{extracted_json['Job_Role']}"
    if role_col in input_df.columns:
        input_df[role_col] = 1
    
    cert_col = f"Cert_{extracted_json['Certifications']}"
    if cert_col in input_df.columns:
        input_df[cert_col] = 1
        
    # Chuẩn hóa (Scaling) dùng bộ scaler bạn đã fit lúc train
    num_cols = ['Experience (Years)', 'Salary Expectation ($)', 'Projects Count']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    return input_df