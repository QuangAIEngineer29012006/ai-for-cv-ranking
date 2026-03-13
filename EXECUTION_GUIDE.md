# AI CV Screening - Execution Guide

## Phase 1: Model Training Setup

### Step 1.1: Data Cleaning
**File:** `data_cleaning.ipynb`

```python
# Expected execution flow:

1. Load raw data
   df = pd.read_csv('data/AI_Resume_Screening.csv')

2. Drop unnecessary columns
   cols_to_drop = ["Resume_ID", "Name", "Score (0-100)", "Salary Expectation ($)"]
   df = df.drop(columns=cols_to_drop)

3. Handle missing values
   text_cols = ['Skills', 'Education', 'Certifications', 'Job Role']
   for col in text_cols:
       df[col] = df[col].fillna('')

4. Encode target label
   df['Recruiter Decision'] = df['Recruiter Decision'].map({
       'Reject': 0,
       'Hire': 1
   })

5. Create TF-IDF vectorizers
   tfidf_skills = TfidfVectorizer(max_features=300, ngram_range=(1,2))
   X_skills = tfidf_skills.fit_transform(df['Skills'])
   (repeat for education, certifications, job role)

6. Scale numeric features
   scaler = StandardScaler()
   X_num = scaler.fit_transform(df[['Experience (Years)', 'Projects Count']])

7. Combine all features
   X_all = hstack([X_skills, X_edu, X_cert, X_job, csr_matrix(X_num)])

8. Save everything
   joblib.dump(tfidf_skills, 'pkl/tfidf_skills.pkl')
   joblib.dump(tfidf_edu, 'pkl/tfidf_edu.pkl')
   joblib.dump(tfidf_cert, 'pkl/tfidf_cert.pkl')
   joblib.dump(tfidf_job, 'pkl/tfidf_job.pkl')
   joblib.dump(scaler, 'pkl/scaler.pkl')
   joblib.dump(X_all, 'pkl/X_sparse.pkl')
   joblib.dump(y, 'pkl/y.pkl')
   df.to_csv('cleaned_data.csv', index=False)
```

**Validation Checklist:**
- [ ] No missing values in critical columns
- [ ] Target label correctly encoded (0 and 1 only)
- [ ] All .pkl files created in pkl/ folder
- [ ] cleaned_data.csv file generated

**Expected Output:**
```
✓ cleaned_data.csv (size ~MB)
✓ pkl/tfidf_skills.pkl
✓ pkl/tfidf_edu.pkl
✓ pkl/tfidf_cert.pkl
✓ pkl/tfidf_job.pkl
✓ pkl/scaler.pkl
✓ pkl/X_sparse.pkl
✓ pkl/y.pkl
```

---

### Step 1.2: Data Visualization & Analysis
**File:** `visualize.ipynb`

```python
# Use cleaned_data.csv output from Step 1.1

1. Load cleaned data
   df = pd.read_csv('cleaned_data.csv')

2. Exploratory Data Analysis
   df.info()              # Check data types and nulls
   df['Recruiter Decision'].value_counts()  # Class balance
   df.describe()          # Numeric statistics

3. Visualizations
   # Class distribution
   plt.figure(figsize=(8, 4))
   df['Recruiter Decision'].value_counts().plot(kind='bar')
   plt.title('Hire vs Reject Distribution')
   
   # Feature distributions
   df['Experience (Years)'].hist(bins=30)
   df['Projects Count'].hist(bins=30)
   
   # Skill frequency analysis
   from collections import Counter
   all_skills = ' '.join(df['Skills']).split(', ')
   skill_counts = Counter(all_skills)
   top_skills = dict(sorted(skill_counts.items(), 
                             key=lambda x: x[1], reverse=True)[:20])
   
   # Correlation with hiring decision
   hire_data = df[df['Recruiter Decision'] == 1]
   reject_data = df[df['Recruiter Decision'] == 0]
   
   print("Average Experience (Hired):", hire_data['Experience (Years)'].mean())
   print("Average Experience (Rejected):", reject_data['Experience (Years)'].mean())

4. Key Insights to Document
   - Class balance (ratio of Hire:Reject)
   - Most important skills appearing in hired candidates
   - Experience levels correlating with hiring
   - Education patterns for successful candidates
   
5. Decision: Choose Model
   ✓ Random Forest: 
     - Good for mixed feature types (text + numeric)
     - Handles non-linear relationships
     - Feature importance interpretable
     - Robust to outliers
```

**Validation Checklist:**
- [ ] All visualizations generated without errors
- [ ] Class balance assessed
- [ ] Top features identified
- [ ] Decision documented (Random Forest selected)

**Key Findings to Record:**
- [ ] Hire/Reject ratio
- [ ] Top 5 skills for hired candidates
- [ ] Average experience for hired vs rejected
- [ ] Any data anomalies

---

### Step 1.3: Model Training
**File:** `random_forest.ipynb`

```python
# Input: X_sparse.pkl, y.pkl (from data_cleaning.ipynb)

1. Load preprocessed data
   X_sparse = joblib.load('pkl/X_sparse.pkl')
   y = joblib.load('pkl/y.pkl')

2. Train-test split
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X_sparse, y, test_size=0.2, random_state=42
   )

3. Train Random Forest
   from sklearn.ensemble import RandomForestClassifier
   
   rf_model = RandomForestClassifier(
       n_estimators=100,
       max_depth=15,
       min_samples_split=5,
       min_samples_leaf=2,
       random_state=42,
       n_jobs=-1
   )
   
   rf_model.fit(X_train, y_train)

4. Evaluate model
   from sklearn.metrics import (
       accuracy_score, precision_score, recall_score, 
       f1_score, confusion_matrix
   )
   
   y_pred = rf_model.predict(X_test)
   y_proba = rf_model.predict_proba(X_test)
   
   print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
   print(f"Precision: {precision_score(y_test, y_pred):.4f}")
   print(f"Recall: {recall_score(y_test, y_pred):.4f}")
   print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
   
   # Confusion matrix
   cm = confusion_matrix(y_test, y_pred)
   print(cm)

5. Load SentenceTransformer
   from sentence_transformers import SentenceTransformer
   
   embed_model = SentenceTransformer("all-MiniLM-L6-v2")
   
   # Build CV texts for centroid
   df = pd.read_csv('cleaned_data.csv')
   
   def build_cv_text(row):
       return (
           str(row['Skills']) + " " +
           str(row['Education']) + " " +
           str(row['Certifications']) + " " +
           str(row['Job Role'])
       )
   
   cv_texts = df.apply(build_cv_text, axis=1).tolist()
   cv_embeddings = embed_model.encode(cv_texts, normalize_embeddings=True)
   
   # Build centroid from hired candidates (class=1)
   hire_embeddings = cv_embeddings[df['Recruiter Decision'] == 1]
   hire_centroid = np.mean(hire_embeddings, axis=0)
   hire_centroid = hire_centroid / np.linalg.norm(hire_centroid)

6. Save models
   joblib.dump(rf_model, 'pkl/random_forest_model.pkl')
   # SentenceTransformer auto-caches, so no need to save separately
   # (loaded via: SentenceTransformer("all-MiniLM-L6-v2"))
```

**Validation Checklist:**
- [ ] Model accuracy > 70%
- [ ] Precision and Recall balanced
- [ ] No overfitting (train vs test scores similar)
- [ ] pkl/random_forest_model.pkl created

**Performance Targets:**
```
✓ Accuracy: >= 0.75
✓ Precision: >= 0.70
✓ Recall: >= 0.70
✓ F1-Score: >= 0.70
```

---

## Phase 2: Production Deployment

### Step 2.1: Set Environment Variables

**File:** `env.txt` or `.env`

```
GEMINI_API_KEY=your_actual_api_key_here
DATABASE_NAME=Resume_AI
DATABASE_HOST=localhost  # or your SQL Server instance
PYTHONPATH=.
```

**Setup:**
```bash
# Windows PowerShell
$env:GEMINI_API_KEY='your_key'
$env:DATABASE_NAME='Resume_AI'

# Or create .env file and load with python-dotenv
```

---

### Step 2.2: Ensure Database Setup

**File:** `CV_save_to_sql.sql`

```sql
-- Create database
CREATE DATABASE Resume_AI;

-- Create table
USE Resume_AI;

CREATE TABLE Resumes (
    ID INT PRIMARY KEY IDENTITY(1,1),
    Skill NVARCHAR(MAX),
    Experience_Years INT,
    Education NVARCHAR(255),
    Certification NVARCHAR(255),
    Job_Role NVARCHAR(255),
    Project_Count INT,
    Score FLOAT,
    File_Name NVARCHAR(255),
    Created_At DATETIME DEFAULT GETDATE()
);

-- Create indexes for performance
CREATE INDEX idx_score ON Resumes(Score DESC);
CREATE INDEX idx_file_name ON Resumes(File_Name);
CREATE INDEX idx_created_at ON Resumes(Created_At DESC);
```

**Validation:**
- [ ] Database `Resume_AI` created
- [ ] Table `Resumes` created with all columns
- [ ] Indexes created for performance

---

### Step 2.3: Install Python Dependencies

```bash
# Core dependencies
pip install fastapi uvicorn python-multipart
pip install pandas scikit-learn joblib numpy scipy

# ML/NLP
pip install sentence-transformers
pip install google-generativeai

# PDF processing
pip install PyPDF2 pdf2image pytesseract

# Database
pip install pyodbc

# Frontend
npm install
```

**Validation:**
- [ ] All pip packages installed
- [ ] npm packages installed
- [ ] No dependency conflicts

---

### Step 2.4: Start Backend API

**File:** `main_api.py`

```bash
# Run FastAPI server
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python -m uvicorn main_api:app --reload
```

**Endpoints Available:**
- `POST /upload-multiple` → Upload multiple PDFs, get scores
- CORS enabled for frontend integration

**Expected Startup:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

---

### Step 2.5: Start Frontend

```bash
# In separate terminal
cd project_root
npm run dev

# Expected output:
# VITE v5.0.0  ready in XXX ms
# ➜  Local:   http://localhost:5173/
```

---

## Phase 3: Testing & Validation

### Step 3.1: Unit Testing

**File:** `TestDriver.py`

```python
# Test individual components

from text_extraction import extract_text_from_pdf
from gemini_api import get_structured_data
from ranking import hybrid_score
from main import convert_to_ranking_format

# Test 1: PDF Extraction
test_pdf = "CV/sample.pdf"
text = extract_text_from_pdf(test_pdf)
assert len(text) > 0, "Text extraction failed"
print("✓ PDF extraction works")

# Test 2: Gemini Parsing
json_data = get_structured_data(text)
assert json_data is not None, "Gemini parsing failed"
assert "skills" in json_data, "Missing skills field"
print("✓ Gemini API works")

# Test 3: Format Conversion
cv_dict = convert_to_ranking_format(json_data)
assert cv_dict is not None, "Format conversion failed"
print("✓ Format conversion works")

# Test 4: Scoring
score = hybrid_score(cv_dict)
assert 0 <= score <= 1, f"Score out of range: {score}"
print(f"✓ Scoring works: {score}")
```

**Validation Checklist:**
- [ ] All test cases pass
- [ ] Scores in valid range [0, 1]
- [ ] No errors in pipeline

---

### Step 3.2: End-to-End Testing

```python
# Full pipeline test

import requests
import json

# Prepare test PDFs
test_files = [
    ("CV/sample1.pdf", open("CV/sample1.pdf", "rb")),
    ("CV/sample2.pdf", open("CV/sample2.pdf", "rb")),
]

# Call API
response = requests.post(
    "http://localhost:8000/upload-multiple",
    files=test_files
)

# Validate response
assert response.status_code == 200
results = response.json()

for result in results["results"]:
    print(f"Candidate: {result['candidate_name']}")
    print(f"Score: {result['score']}")
    print(f"Status: {result['status']}")
    
    # Validate in range
    if result['status'] == 'success':
        assert 0 <= result['score'] <= 1
        print("✓ Valid score")
```

---

## Phase 4: Production Monitoring

### Metrics to Track

```python
# logging_config.py

import logging
from datetime import datetime

logging.basicConfig(
    filename=f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log each stage
logger = logging.getLogger(__name__)

logger.info("PDF extraction: 2.3s")
logger.info("Gemini API: 1.8s")
logger.info("Scoring: 0.5s")
logger.info("DB insert: 0.2s")
logger.info("Total: 4.8s")
```

### Expected Performance

| Stage | Time | Notes |
|-------|------|-------|
| Extract | 1-3s | Depends on PDF size |
| Gemini | 2-5s | API latency |
| Score | 0.2-0.5s | Model inference |
| DB Save | 0.1-0.3s | Network latency |
| **Total** | **4-10s** | Per CV |

---

## Troubleshooting

### Issue: "Cannot find GEMINI_API_KEY"
**Solution:**
```bash
# Set environment variable
$env:GEMINI_API_KEY='sk-...'

# Or create .env and use python-dotenv
python -c "from dotenv import load_dotenv; load_dotenv()"
```

### Issue: "Models not found in pkl/"
**Solution:**
```bash
# Ensure training phase completed
# Run all cells in: data_cleaning.ipynb, random_forest.ipynb

# Verify files exist
ls -la pkl/
# Should show: *.pkl files
```

### Issue: "SQL Connection failed"
**Solution:**
```bash
# Check SQL Server is running
# Verify database exists
sqlcmd -S . -d Resume_AI -Q "SELECT 1"

# Check ODBC driver
python -c "import pyodbc; print(pyodbc.drivers())"
```

### Issue: "PDF extraction returns empty text"
**Solution:**
```bash
# Ensure pytesseract installed
pip install pytesseract pdf2image

# Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Set path in code
import pytesseract
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

---

## Success Checklist

- [ ] Phase 1: Data Cleaning ✓
- [ ] Phase 1: Visualization & Analysis ✓
- [ ] Phase 1: Model Training ✓
- [ ] Phase 2: Environment Setup ✓
- [ ] Phase 2: Database Ready ✓
- [ ] Phase 2: Backend API Running ✓
- [ ] Phase 2: Frontend Running ✓
- [ ] Phase 3: Unit Tests Passing ✓
- [ ] Phase 3: E2E Tests Passing ✓
- [ ] Phase 4: Monitoring in Place ✓

🎉 **System Ready for Production**

