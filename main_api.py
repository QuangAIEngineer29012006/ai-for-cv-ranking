from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import shutil
import os
import pyodbc
import socket
import asyncio

# Import pipeline xử lý CV
from main import main as process_cv_logic

app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= DATABASE CONFIG =================

def get_sql_driver():
    drivers = pyodbc.drivers()
    for driver in drivers:
        if "ODBC Driver 17 for SQL Server" in driver:
            return driver
    return "ODBC Driver 17 for SQL Server"

def get_connection():
    try:
        driver = get_sql_driver()
        connection_string = (
            f"Driver={{{driver}}};"
            f"Server=COMPUTER;"
            f"Database=Resume_AI;"
            f"Trusted_Connection=yes;"
            f"TrustServerCertificate=yes;"
        )
        return pyodbc.connect(connection_string)
    except Exception as e:
        print("DATABASE CONNECTION ERROR:", e)
        raise


# ================= UPLOAD MULTIPLE CVS =================

@app.post("/upload-multiple")
async def upload_multiple_cvs(files: List[UploadFile] = File(...)):

    results = []
    temp_folder = "temp_uploads"

    os.makedirs(temp_folder, exist_ok=True)
    
    for file in files:

        if not file.filename.lower().endswith(".pdf"):
            results.append({
                "candidate_name": file.filename,
                "error": "Not a PDF file",
                "status": "skipped"
            })
            continue

        # Extract just the filename to handle nested folder uploads from browser
        basename = file.filename.split("/")[-1].split("\\")[-1]
        safe_filename = basename.replace(" ", "_")
        
        file_path = os.path.join(temp_folder, safe_filename)

        try:

            # ===== Save file =====
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # ===== Run CV pipeline =====
            score = process_cv_logic(file_path)

            if score is None:
                results.append({
                    "candidate_name": file.filename,
                    "error": "Fail",
                    "status": "failed"
                })
            else:
                results.append({
                    "candidate_name": file.filename,
                    "score": round(float(score), 1),
                    "status": "success"
                })

        except Exception as e:

            results.append({
                "candidate_name": file.filename,
                "error": str(e),
                "status": "failed"
            })

        finally:

            # Xóa file tạm
            if os.path.exists(file_path):
                os.remove(file_path)
                
        # Delay để tránh lỗi rate limit của Gemini API Free Tier
        await asyncio.sleep(16)

    return results


# ================= SORT CVS =================

@app.post("/sort-cvs")
async def sort_cvs_endpoint(cv_list: List[Dict[str, Any]], order: str = "desc"):
    try:
        def get_score(item):
            val = item.get("score", 0)
            try:
                return float(val) if val is not None else 0.0
            except:
                return 0.0
        
        sorted_list = sorted(cv_list, key=get_score, reverse=(order == "desc"))
        return sorted_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================= HISTORY =================

@app.get("/history")
async def get_history():

    try:

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ID, File_Name, Score
            FROM Resumes
            ORDER BY ID DESC
        """)

        rows = cursor.fetchall()

        conn.close()

        result = []

        for row in rows:

            result.append({
                "id": row[0],
                "name": row[1],
                "date": "N/A",
                "score": round(float(row[2]), 1) if row[2] is not None else 0.0
            })

        return result

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


# ================= DELETE CV =================

@app.delete("/delete-cv/{cv_id}")
async def delete_cv(cv_id: int):

    try:

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM Resumes WHERE ID = ?", (cv_id,)
        )

        conn.commit()
        conn.close()

        return {"message": "Deleted successfully"}

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


# ================= RUN SERVER =================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080
    )