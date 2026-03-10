import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Đọc dữ liệu đã làm sạch từ bước trước
df = pd.read_csv('Cleaned_Data.csv')

# 2. Chia dữ liệu thành Features (X) và Target (y)
# Mục tiêu của chúng ta là dự đoán 'Score (0-100)'
X = df.drop(columns=['Score (0-100)', 'Recruiter Decision']) 
y = df['Score (0-100)']

# 3. Chia tập Train và Test (80% học, 20% để kiểm tra độ giỏi của AI)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Khởi tạo và huấn luyện mô hình
print("Đang huấn luyện mô hình... Vui lòng đợi trong giây lát.")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Đánh giá mô hình
y_pred = model.predict(X_test)
print(f"Độ chính xác (R2 Score): {r2_score(y_test, y_pred):.2f}")
print(f"Sai số trung bình: {mean_absolute_error(y_test, y_pred):.2f} điểm")

# 6. LƯU QUAN TRỌNG: Lưu model và danh sách cột để dùng cho phần LLM
joblib.dump(model, 'cv_scoring_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("Đã lưu mô hình thành công tại 'cv_scoring_model.pkl'!")