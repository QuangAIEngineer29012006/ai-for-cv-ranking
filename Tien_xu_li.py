import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import joblib

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv('data/AI_Resume_Screening.csv')

# 2. Loại bỏ các cột không cần thiết cho mô hình
df = df.drop(columns=['Resume_ID', 'Name'])

# 3. Xử lý giá trị thiếu (missing value) trong cột Certifications
df['Certifications'] = df['Certifications'].fillna('None')

# 4. Xử lý cột Skills bằng MultiLabelBinarizer (One-hot encoding cho danh sách)
# Tách các kỹ năng thành danh sách dựa trên dấu phẩy
df['Skills'] = df['Skills'].apply(lambda x: [s.strip() for s in x.split(',')])
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df['Skills'])
skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

# 5. Sử dụng Ordinal Encoding cho cột Education_Level (theo thứ tự học vấn)
education_mapping = {
    'B.Sc': 1,
    'B.Tech': 2,
    'M.Tech': 3,
    'MBA': 4,
    'PhD': 5
}
df['Education'] = df['Education'].map(education_mapping)

# 6. Mã hóa One-Hot cho các biến phân loại không theo thứ tự (Job Role và Certifications)
df = pd.get_dummies(df, columns=['Job Role', 'Certifications'], prefix=['Role', 'Cert'])

# 7. Chuyển đổi nhãn mục tiêu 'Recruiter Decision' thành dạng số (1: Hire, 0: Reject)
df['Recruiter Decision'] = df['Recruiter Decision'].map({'Hire': 1, 'Reject': 0})

# 8. Chuẩn hóa các đặc trưng số (Scaling) về khoảng [0, 1]
scaler = MinMaxScaler()
numerical_cols = ['Experience (Years)', 'Salary Expectation ($)', 'Projects Count']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 9. Loại bỏ cột Skills cũ và nối với DataFrame kỹ năng đã mã hóa
df = pd.concat([df.drop(columns=['Skills']), skills_df], axis=1)

# 10. Xuất file CSV đã xử lý xong
df.to_csv('Cleaned_Data.csv', index=False)
print("--- 5 hàng đầu của dữ liệu đã xử lý ---")
print(df.head())

# 11. Lưu các đối tượng cần thiết cho quá trình dự báo (Inference) sau này
# Lưu bộ chuẩn hóa để dùng lại cho dữ liệu mới
joblib.dump(scaler, 'scaler.pkl')

# Lưu danh sách các cột theo đúng thứ tự để đảm bảo dữ liệu đầu vào mô hình luôn khớp
joblib.dump(df.columns.tolist(), 'feature_columns.pkl')

print("\nĐã lưu file Cleaned_Data.csv, scaler.pkl và feature_columns.pkl")