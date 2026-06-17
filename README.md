# AI-Powered CV Screening & Ranking Pipeline

An end-to-end, enterprise-grade CV screening solution that combines traditional Machine Learning, Semantic Embeddings, and Generative AI to automate, score, and rank candidate resumes in real-time. 

The system features a **Hybrid Scoring Engine** (Random Forest + SentenceTransformers) alongside an LLM-powered parser using Google Gemini, wrapped in a high-performance FastAPI backend and an intuitive React/Vite dashboard.

---

## 📊 System Architecture & Workflow

The pipeline is split into three main stages: **Data Preparation & Model Training**, **Real-Time Production Inference**, and **Frontend Analytics**.

              AI CV SCREENING PIPELINE ARCHITECTURE             
          STAGE 1: MODEL PREPARATION (ONE-TIME SETUP)
═════════════════════════════════════════════════════════════════════
STEP 1️⃣: DATA CLEANING (data_cleaning.ipynb)
• Load raw CSV data, drop unnecessary columns, encode target labels.
• Fit 4 distinct TF-IDF vectorizers (Skills, Edu, Cert, Job).
• Standardize numerical features using StandardScaler.  STEP 2️⃣: DATA VISUALIZATION & ANALYSIS (visualize.ipynb)
• Perform Exploratory Data Analysis (EDA) on class distributions.
• Select Random Forest based on feature importance indicators.  STEP 3️⃣: MODEL TRAINING (random_forest.ipynb)
• Train Random Forest Classifier with cross-validation optimization.
• Load SentenceTransformer ("all-MiniLM-L6-v2") & bake hire_centroid.  STAGE 2: REAL-TIME INFERENCE (PRODUCTION FLOW)
═════════════════════════════════════════════════════════════════════
[FastAPI Entry: POST /upload-multiple]
↓
STEP 1️⃣: PDF TEXT EXTRACTION (text_extraction.py)
• PyPDF2 Text Extraction ──(If empty)──> Fallback to OCR (pytesseract).
↓
STEP 2️⃣: STRUCTURED DATA PARSING (gemini_api.py)
• Prompt Gemini API (gemini-2.5-flash / 2.0-flash) to extract structured JSON.
↓
STEP 3️⃣: FORMAT CONVERSION (main.py)
• Map Gemini JSON outputs into pandas-compatible ranking format.
↓
STEP 4️⃣: HYBRID SCORING ENGINE (ranking.py)
• ML Score: Random Forest prediction probability.
• Semantic Score: Cosine similarity against hire_centroid matrix.
• Final Score = (α × score_rf) + (β × score_semantic).
↓
STEP 5️⃣: DATABASE PERSISTENCE (save_data_to_sql.py)
• Commit features and computed scores into MS SQL Server (Resume_AI).
↓
STEP 6️⃣: API RESPONSE ──> JSON metrics returned to frontend client.  STAGE 3: FRONTEND & VISUALIZATION (PRODUCTION REACT UI)
═════════════════════════════════════════════════════════════════════
• Home.jsx      : Dynamic multi-file drag-and-drop PDF upload interface.
• Dashboard.jsx : Real-time scoring monitors & score distribution charts.
• History.jsx   : Query SQL database records with sorting & filtering tools.  
---

## 🛠️ Features & Methodology

### 1. Training & Core Machine Learning (`STAGE 1`)
- **Data Preprocessing:** Standardizes raw resumes, handles categorical text fields, and applies advanced vectorization (`TF-IDF` with tuned `max_features` and `ngram_ranges` for skills, education, certifications, and roles) alongside `StandardScaler` for experience metrics.
- **Model Selection:** Utilizes a custom-tuned **Random Forest Classifier** selected based on feature importance indicators and its robust capacity to handle complex, non-linear text-categorical relationships.
- **Semantic Clustering:** Leverages `SentenceTransformer` (`all-MiniLM-L6-v2`) to build a high-dimensional `hire_centroid` embedding matrix from historically successful candidates.

### 2. Real-Time Production Inference (`STAGE 2`)
- **Dual-Layer Text Extraction:** Fast extraction via `PyPDF2` with an automated fallback mechanism to an **OCR Pipeline** using `pdf2image` and `pytesseract` for scanned documents.
- **LLM-Powered JSON Parsing:** Integrates a robust fallback chain of Google Gemini models (`gemini-2.5-flash` $\rightarrow$ `gemini-2.0-flash`) using strict prompt engineering to parse unstructured text into standardized JSON schemas.
- **Hybrid Scoring Engine:** Calculates a highly calibrated score bounded within $[0, 1]$ using a weighted hybrid formula:
  $$Score = \alpha \cdot Score_{RandomForest} + \beta \cdot Score_{SemanticSimilarity}$$
  *(Default configuration: 60% Machine Learning probability, 40% Semantic Cosine Similarity)*.

---

## 📁 Repository Structure

```text
├── data/                      # Raw datasets (AI_Resume_Screening.csv)
├── pkl/                       # Serialized model checkpoints & transformers (.pkl)
├── notebooks/                 # Development & Experimental Notebooks
│   ├── data_cleaning.ipynb    # Data cleaning & feature vectorization pipeline
│   ├── visualize.ipynb        # Exploratory Data Analysis (EDA) & Model selection
│   └── random_forest.ipynb    # Random forest classifier training routines
├── src/                       # Production Backend Modules
│   ├── main_api.py            # Main FastAPI server and routing entry point
│   ├── main.py                # Data format mapping and processing orchestration
│   ├── text_extraction.py     # Multi-engine PDF/OCR extraction logic
│   ├── gemini_api.py          # Google Gemini structured inference agent
│   ├── ranking.py             # Hybrid Scoring & Vector transformation core
│   └── save_data_to_sql.py    # pyodbc database driver pipeline
├── frontend/                  # React/Vite Single-Page Application source
│   ├── package.json           # Frontend dependency manifest
│   └── src/components/        # Home, Dashboard, History, Sidebar UI components
├── requirements.txt           # Python application dependencies
└── README.md                  # Project documentation
```
## 📚 Technical Interfaces1. Extracted JSON Schema (Gemini Agent Output)
```
JSON{
  "name": "Candidate Full Name",
  "skills": "Python, TensorFlow, SQL, ...",
  "education": "B.Tech | B.Sc | M.Tech | MBA | PhD",
  "certifications": "Google ML | AWS Certified | ...",
  "job_role": "AI Researcher | Data Scientist | ...",
  "experience_years": 5,
  "projects": 3
}
```
## 2. Combined Pipeline Vector Format (Pandas Compatible)
```
Pythoncv_dict = {
    "Skills": str,
    "Education": str,
    "Certifications": str,
    "Job Role": str,
    "Experience (Years)": float,
    "Projects Count": float
}
```
## 🚀 Environment Setup & InstallationBackend Server SetupEnsure Tesseract OCR is installed locally on your operating system, then build dependencies:  Bash# Install mandatory Python requirements
```
pip install pandas scikit-learn joblib sentence-transformers google-generativeai
pip install PyPDF2 pdf2image pytesseract pyodbc
```
# Configure production environment credentials
set GEMINI_API_KEY=your_production_key_here
Frontend Client SetupBash# Navigate to the react app and install packages
```
cd frontend
npm install
```
# Launch production-ready local dev server
```
npm run dev
```
📞 Troubleshooting & Support
PDF extraction failure: Verify local binary installations of pytesseract and system environment paths[cite: 1].

Gemini API timeout: Check GEMINI_API_KEY configurations and internet connections[cite: 1].

SQL Driver connection errors: Ensure the database targeted matches the configured Trusted_Connection permissions[cite: 1].

Blank models error: Execute all training notebooks completely inside the notebooks/ directory before starting main_api.py[cite: 1].

