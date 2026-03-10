import pyodbc

# In ra tất cả driver mà Python THỰC SỰ nhìn thấy
print("Các driver Python tìm thấy là:")
for d in pyodbc.drivers():
    print(f"- {d}")

# Thử kết nối bằng cái tên tìm được đầu tiên có chữ 'SQL Server'
target_driver = [d for d in pyodbc.drivers() if 'SQL Server' in d]

if target_driver:
    chosen = target_driver[0]
    print(f"\nĐang thử kết nối với: {chosen}")
    conn_str = f"Driver={{{chosen}}};Server=.\SQLEXPRESS;Database=ResumeAI;Trusted_Connection=yes;TrustServerCertificate=yes;"
    try:
        cnxn = pyodbc.connect(conn_str)
        print("KẾT NỐI THÀNH CÔNG RỒI!")
    except Exception as e:
        print(f"Vẫn lỗi: {e}")
else:
    print("\nPython KHÔNG nhìn thấy bất kỳ SQL Driver nào. Có thể do lệch pha 32/64 bit.")