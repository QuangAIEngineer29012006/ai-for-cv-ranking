# 🚀 AI CV Screening Pipeline

## 📊 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                 AI CV SCREENING PIPELINE ARCHITECTURE               │
└─────────────────────────────────────────────────────────────────────┘

STAGE 1: MODEL PREPARATION (ONE-TIME SETUP)
═════════════════════════════════════════════════════════════════════

  STEP 1️⃣: DATA CLEANING
  ────────────────────────
  📥 Input: data/AI_Resume_Screening.csv
  
  File: data_cleaning.ipynb
  Process:
    • Load raw data
    • Drop unnecessary columns (Resume_ID, Name, Score, Salary)
    • Handle missing values in text fields (Skills, Education, Certifications, Job Role)
    • Encode target label (Recruiter Decision: Reject→0, Hire→1)
    • Create TF-IDF vectorizers:
      - tfidf_skills (max_features=300, ngram_range=(1,2))
      - tfidf_edu (max_features=20)
      - tfidf_cert (max_features=50)
      - tfidf_job (max_features=30)
    • Normalize numeric features (Experience Years, Projects Count) with StandardScaler
    
  📤 Output: cleaned DataFrame + feature transformers
  💾 Saved: pkl/
     - tfidf_skills.pkl
     - tfidf_edu.pkl
     - tfidf_cert.pkl
     - tfidf_job.pkl
     - scaler.pkl
     - X_sparse.pkl (all features combined)
     - y.pkl (target labels)


  STEP 2️⃣: DATA VISUALIZATION & ANALYSIS
  ───────────────────────────────────────
  📥 Input: cleaned_data.csv (output from Step 1)
  
  File: visualize.ipynb
  Process:
    • Exploratory Data Analysis (EDA):
      - Check class distribution (Hire vs Reject)
      - Feature distributions
      - Missing data analysis
    • Visualization:
      - Skill frequency analysis
      - Education level distribution
      - Certification patterns
      - Experience vs hiring rate
      - Feature importance indicators
    • Correlation analysis
    • Data quality assessment
    
  📊 Output: Visual insights to guide model selection
  📌 Decision Point: Choose Random Forest as primary model
     (Based on feature importance, non-linear relationships, 
      ability to handle categorical text data)


  STEP 3️⃣: MODEL TRAINING
  ───────────────────────
  📥 Input: X_sparse.pkl, y.pkl, transformed features, scaler
  
  File: random_forest.ipynb
  Process:
    • Load training data and preprocessed features
    • Train Random Forest Classifier:
      - n_estimators=100 (or tuned value)
      - max_depth optimized via cross-validation
      - Feature importance analysis
    • Model evaluation:
      - Train/test split
      - Cross-validation scores
      - Confusion matrix
      - Precision, Recall, F1 scores
    • Load SentenceTransformer for semantic embeddings:
      - Model: "all-MiniLM-L6-v2"
      - Build hire_centroid from positive examples
      
  💾 Output: pkl/random_forest_model.pkl
  🎯 Model ready for inference


═════════════════════════════════════════════════════════════════════

STAGE 2: REAL-TIME INFERENCE (PRODUCTION)
═════════════════════════════════════════════════════════════════════

  PIPELINE ENTRY: FastAPI Backend (main_api.py)
  ───────────────────────────────────────────
  
  POST /upload-multiple
    ├─ Accepts: List of PDF files
    ├─ Database: SQL Server (Resume_AI)
    └─ Returns: JSON with scores
    
  
  STEP 1️⃣: PDF TEXT EXTRACTION
  ──────────────────────────────
  Module: text_extraction.py
  Function: extract_text_from_pdf(file_path)
  
  Process:
    • Try PyPDF2 text extraction (fast)
    • If empty → Fallback to OCR:
      - pdf2image (convert pages to images)
      - pytesseract (OCR text from images)
    • Convert to lowercase
    • Handle errors gracefully
    
  📤 Output: raw_text (string)
  ⚠️ Validation: Check if text is empty


  STEP 2️⃣: STRUCTURED DATA PARSING
  ─────────────────────────────────
  Module: gemini_api.py
  Function: get_structured_data(pdf_text)
  
  Process:
    • Send CV text to Google Gemini API
    • Prompt: Extract structured JSON:
      {
        "name": "...",
        "skills": "Python, TensorFlow, SQL",
        "education": "B.Tech | B.Sc | M.Tech | MBA | PhD",
        "certifications": "Google ML | AWS Certified | ...",
        "job_role": "AI Researcher | Data Scientist | ...",
        "experience_years": number,
        "projects": number
      }
    • Model fallback chain:
      1. gemini-2.5-flash (latest)
      2. gemini-2.0-flash
      3. gemini-2.0-flash-001
    • Clean JSON response (remove markdown formatting)
    
  📤 Output: extracted_json (dict)
  ⚠️ Validation: Ensure required fields present


  STEP 3️⃣: DATA FORMAT CONVERSION
  ───────────────────────────────
  Module: main.py
  Function: convert_to_ranking_format(data)
  
  Process:
    • Map Gemini JSON to ranking format:
      {
        "Skills": "...",
        "Education": "...",
        "Certifications": "...",
        "Job Role": "...",
        "Experience (Years)": number,
        "Projects Count": number
      }
    • Handle list→string conversion for skills
    
  📤 Output: cv_dict (pandas-compatible format)


  STEP 4️⃣: HYBRID SCORING
  ──────────────────────
  Module: ranking.py
  Function: hybrid_score(cv_dict, alpha=0.6, beta=0.4)
  
  Process:
    Load pre-trained models:
    • TF-IDF transformers (all 4 types)
    • Random Forest model
    • StandardScaler
    • SentenceTransformer
    • Hire centroid embeddings
    
    Feature transformation:
    • X_skills = tfidf_skills.transform(cv_dict["Skills"])
    • X_edu = tfidf_edu.transform(cv_dict["Education"])
    • X_cert = tfidf_cert.transform(cv_dict["Certifications"])
    • X_job = tfidf_job.transform(cv_dict["Job Role"])
    • X_num = scaler.transform([Experience, Projects])
    
    Combine features:
    • Concatenate all features into X_combined (sparse matrix)
    
    Scoring (hybrid approach):
    ├─ Method 1: Random Forest probability
    │  score_rf = random_forest_model.predict_proba(X_combined)[0][1]
    │
    └─ Method 2: Semantic similarity
       • Embed CV text using SentenceTransformer
       • Calculate cosine_similarity(cv_embedding, hire_centroid)
       score_semantic = cosine_similarity_value
    
    Final score:
    score = α × score_rf + β × score_semantic
    (default: 60% RF, 40% semantic)
    
  📤 Output: final_score (0-1 range)


  STEP 5️⃣: DATABASE PERSISTENCE
  ──────────────────────────────
  Module: save_data_to_sql.py
  Function: save_cv_to_sql(extracted_json, ai_score, file_name)
  
  Process:
    • Connect to SQL Server:
      - Database: Resume_AI
      - Table: Resumes
      - Auth: Trusted_Connection (Windows Auth)
    
    • Prepare data from extracted_json:
      - skills, experience_years, education
      - certifications, job_role, projects
      
    • INSERT query:
      INSERT INTO Resumes (
        Skill, Experience_Years, Education, Certification,
        Job_Role, Project_Count, Score, File_Name
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      
    • Commit transaction
    
  💾 Output: Record stored in database


  STEP 6️⃣: API RESPONSE
  ──────────────────────
  Module: main_api.py
  Endpoint: POST /upload-multiple
  
  Process:
    FOR EACH file in files:
      1. Validate file extension (.pdf)
      2. Save to temp_uploads/
      3. Execute full pipeline (Steps 1-5)
      4. Record result with status
      5. Clean up temp file
    
    Compile results:
    {
      "results": [
        {
          "candidate_name": "file.pdf",
          "score": 0.875,
          "status": "success"
        },
        ...
      ]
    }
    
  📤 Output: JSON response to frontend


═════════════════════════════════════════════════════════════════════

STAGE 3: FRONTEND & VISUALIZATION (PRODUCTION)
═════════════════════════════════════════════════════════════════════

  React/Vite Application (src/)
  
  Components:
  ├─ Home.jsx
  │  └─ Multiple file upload interface
  │     └─ Call POST /upload-multiple
  │
  ├─ Dashboard.jsx
  │  └─ Display real-time scores
  │     └─ Table: Candidate Name, Score, Status
  │     └─ Chart: Score distribution
  │     └─ Sort/Filter options
  │
  ├─ History.jsx
  │  └─ Query SQL database
  │     └─ Show all processed CVs
  │     └─ Filter by date, score range
  │     └─ Export results
  │
  └─ Sidebar.jsx
     └─ Navigation between pages
     └─ Statistics/Analytics view


═════════════════════════════════════════════════════════════════════
```

---

## 📋 Execution Flow Summary

### **Training Phase** (One-time)
```
DATA
  ↓
[1] CLEAN (data_cleaning.ipynb)
  ↓
CLEANED_DATA + VECTORIZERS
  ↓
[2] VISUALIZE (visualize.ipynb) → Choose Random Forest
  ↓
[3] TRAIN (random_forest.ipynb)
  ↓
MODEL + SCALER (pkl/)
```

### **Inference Phase** (Real-time)
```
PDF FILES
  ↓
[1] TEXT EXTRACT → raw_text
  ↓
[2] GEMINI PARSE → JSON
  ↓
[3] FORMAT CONVERT → cv_dict
  ↓
[4] HYBRID SCORE → AI_score
  ↓
[5] DB SAVE → Resume_AI table
  ↓
[6] API RESPONSE → Frontend
  ↓
[7] DISPLAY → Dashboard
```

---

## 🛠️ File Dependencies

```
Training Dependencies:
├─ data/AI_Resume_Screening.csv
│  ├─ data_cleaning.ipynb
│  ├─ visualize.ipynb  
│  └─ random_forest.ipynb
└─ Output: cleaned_data.csv, pkl/*

Inference Dependencies:
├─ main_api.py
│  ├─ main.py
│  │  ├─ text_extraction.py
│  │  └─ gemini_api.py
│  ├─ ranking.py
│  │  └─ pkl/*
│  └─ save_data_to_sql.py
├─ PyPDF2, google.generativeai, sentence_transformers
├─ sklearn, pandas, numpy
└─ SQL Server connection (pyodbc)

Frontend Dependencies:
├─ package.json (React, Vite, React Router)
├─ src/App.jsx
├─ src/components/*
└─ src/pages/*
```

---

## ✅ Quality Checkpoints

| Stage | Validation |
|-------|-----------|
| **Clean** | No missing values, correct encoding |
| **Visualize** | Data quality confirmed, patterns identified |
| **Train** | Cross-validation scores, model accuracy verified |
| **Extract** | Text length > 0, fallback to OCR tested |
| **Parse** | Valid JSON, required fields present |
| **Score** | Score in range [0, 1] |
| **Store** | SQL Insert success, no duplicates |
| **Display** | Frontend renders correct scores |

---

## 🔍 Environment Setup

```bash
# Python dependencies
pip install pandas scikit-learn joblib sentence-transformers google-generativeai
pip install PyPDF2 pdf2image pytesseract pyodbc

# Tesseract installation (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

# Environment variables
set GEMINI_API_KEY=your_key_here

# Frontend
npm install
npm run dev
```

---

## 📞 Support & Troubleshooting

| Issue | Solution |
|-------|----------|
| PDF text extraction fails | Ensure OCR installed (pytesseract) |
| Gemini API timeout | Check GEMINI_API_KEY, internet connection |
| SQL Server connection error | Verify database name, Windows Auth enabled |
| Model not found | Run training notebooks first, check pkl/ dir |
| Score always 0 | Check TF-IDF vectorizer fitting, numeric scaling |

