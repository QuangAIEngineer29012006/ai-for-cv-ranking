CREATE TABLE Resumes (

    ID INT IDENTITY(1,1) PRIMARY KEY,

    Skill NVARCHAR(500),

    Experience_Years INT,

    Education NVARCHAR(50),

    Certification NVARCHAR(100),

    Job_Role NVARCHAR(100),

    Project_Count INT,

    Score FLOAT,

    File_Name NVARCHAR(200)

);

select * from Resumes