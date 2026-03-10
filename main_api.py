from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from fastapi import UploadFile, File, BackgroundTasks
import shutil
import os
import pyodbc
import socket

# Import pipeline xử lý CV
from main import main as process_cv_logic

app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi máy truy cập
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= DATABASE CONFIG =================

def get_server_name():
    """
    Tự động lấy tên máy hiện tại để tránh lỗi login
    """
    computer_name = socket.gethostname()
    return f"{computer_name}\\SQLEXPRESS"

CONNECTION_STRING = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=COMPUTER;"
    "Database=Resume_AI;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

def get_connection():
    try:
        return pyodbc.connect(CONNECTION_STRING)
    except Exception as e:
        print("❌ DATABASE CONNECTION ERROR:", e)
        raise


# ================= UPLOAD CV =================
@app.post("/upload-multiple")
async def upload_multiple_cvs(files: List[UploadFile] = File(...)):
    results = []
    temp_folder = "temp_uploads"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder, exist_ok=True)
    
    for file in files:
        safe_filename = file.filename.replace(" ", "_")
        file_path = os.path.join(temp_folder, safe_filename)
        # Lưu file tạm
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            score = process_cv_logic(file_path)
            results.append({
                "candidate_name": file.filename,
                "score": round(float(score), 2) if score else 0,
                "status": "success"
            })

            # Chạy model
            score = process_cv_logic(file_path)

            # Xoá file tạm
            if os.path.exists(file_path):
                os.remove(file_path)

            results.append({
                "candidate_name": file.filename,
                "score": round(float(score), 2) if score else 0,
                "status": "success"
            })

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e), "status": "failed"})
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=str(e))

    return results


# ================= HISTORY =================
@app.get("/history")
async def get_history():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT Resume_ID, Candidate_Name, Upload_Date, Score
            FROM Resumes
            ORDER BY Upload_Date DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        result = []
        for row in rows:
            result.append({
                "id": row[0],
                "name": row[1],
                "date": row[2].strftime("%d/%m/%Y %H:%M") if row[2] else "",
                "score": row[3]
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================= DELETE =================
@app.delete("/delete-cv/{cv_id}")
async def delete_cv(cv_id: int):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM Resumes WHERE Resume_ID = ?", (cv_id,))
        conn.commit()
        conn.close()

        return {"message": "Deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
