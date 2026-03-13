import pyodbc


def get_sql_driver():
    drivers = pyodbc.drivers()
    for driver in drivers:
        if "ODBC Driver 17 for SQL Server" in driver:
            return driver
    return "ODBC Driver 17 for SQL Server"


def get_connection():
    driver = get_sql_driver()

    connection_string = (
        f"Driver={{{driver}}};"
        f"Server=COMPUTER;"
        f"Database=Resume_AI;"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )

    return pyodbc.connect(connection_string)


def save_cv_to_sql(extracted_json, ai_score, file_name):

    try:

        connection = get_connection()
        cursor = connection.cursor()

        # ===== mapping dữ liệu từ Gemini JSON =====

        skills = extracted_json.get("skills", "")
        if isinstance(skills, list):
            skills = ", ".join(skills)

        experience = extracted_json.get("experience_years", 0)

        education = extracted_json.get("education", "None")

        certification = extracted_json.get("certifications", "None")

        job_role = extracted_json.get("job_role", "None")

        projects = extracted_json.get("projects", 0)


        # ===== SQL INSERT =====

        sql_query = """
        INSERT INTO Resumes (
            Skill,
            Experience_Years,
            Education,
            Certification,
            Job_Role,
            Project_Count,
            Score,
            File_Name
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        cursor.execute(sql_query, (
            skills,
            experience,
            education,
            certification,
            job_role,
            projects,
            ai_score,
            file_name
        ))

        connection.commit()

        cursor.close()
        connection.close()

        print("Đã lưu CV vào database thành công.")

    except Exception as e:
        print(f"DATABASE ERROR: {e}")