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


def save_cv_to_sql(extracted_json, ai_score, candidate_name):

    try:
        connection = get_connection()
        cursor = connection.cursor()

        name = extracted_json.get('Name') or candidate_name
        skills_str = ", ".join(extracted_json.get('Skills', []))

        input_data = (
            name,
            skills_str,
            extracted_json.get('Experience', 0),
            extracted_json.get('Education', 'None'),
            extracted_json.get('Certifications', 'None'),
            extracted_json.get('Job_Role', 'None'),
            extracted_json.get('Salary_Expectation', 0),
            extracted_json.get('Projects_Count', 0),
            ai_score
        )

        sql_query = """
            INSERT INTO Resumes (
                Candidate_Name, Skill, Experience_Years, Education,
                Certification, Job_Role, Salary_Expectation,
                Project_Count, Score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        cursor.execute(sql_query, input_data)
        connection.commit()
        connection.close()

        print("Đã lưu vào database thành công.")

    except Exception as e:
        print(f"DATABASE ERROR: {e}")
