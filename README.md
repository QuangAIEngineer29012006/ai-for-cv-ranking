# AI-Powered CV Screening & Ranking Pipeline

An end-to-end, enterprise-grade CV screening solution that combines traditional Machine Learning, Semantic Embeddings, and Generative AI to automate, score, and rank candidate resumes in real-time. 

The system features a **Hybrid Scoring Engine** (Random Forest + SentenceTransformers) alongside an LLM-powered parser using Google Gemini, wrapped in a high-performance FastAPI backend and an intuitive React/Vite dashboard.

---
# AI-Powered CV Screening & Ranking Pipeline

An end-to-end, enterprise-grade CV screening solution that combines traditional Machine Learning, Semantic Embeddings, and Generative AI to automate, score, and rank candidate resumes in real-time. 

The system features a **Hybrid Scoring Engine** (Random Forest + SentenceTransformers) alongside an LLM-powered parser using Google Gemini, wrapped in a high-performance FastAPI backend and an intuitive React/Vite dashboard.

---

## 📊 System Architecture & Workflow

The pipeline is explicitly decoupled into an **Offline Training Pipeline** (for feature extraction and model checkpoint serialization) and an **Online Real-Time Inference Pipeline** (for high-throughput CV processing and scoring).

### 1. Stage 1: Model Preparation Pipeline (Offline Matrix)
* **Data Cleaning & Engineering (`data_cleaning.ipynb`):** Ingests raw training arrays (`AI_Resume_Screening.csv`), drops diagnostic metadata metrics (Resume_ID, Name, Score, Salary), and encodes the binary recruitment ground truth (`Reject ➔ 0`, `Hire ➔ 1`). It instantiates and fits **four independent TF-IDF vectorizers** specializing in localized categorical vocabularies:
  * `tfidf_skills`: Optimized with `max_features=300` and `ngram_range=(1,2)`.
  * `tfidf_edu`: Optimized with `max_features=20`.
  * `tfidf_cert`: Optimized with `max_features=50`.
  * `tfidf_job`: Optimized with `max_features=30`.
  * Numeric features (`Experience Years`, `Projects Count`) are dynamically normalized using `StandardScaler`.
* **Exploratory Data Analysis (`visualize.ipynb`):** Assesses class imbalances, feature correlations, and missing data density to validate algorithm selections.
* **Model Checkpoint Generation (`random_forest.ipynb`):** Trains a robust **Random Forest Classifier** optimized via cross-validation. Concurrently, it passes positive historical vectors through a `SentenceTransformer` (`all-MiniLM-L6-v2`) to derive and bake a high-dimensional reference matrix known as the `hire_centroid`. All structural transformers and final weights are dumped into the `pkl/` registry.

### 2. Stage 2: Real-Time Inference Loop (Production Engine)
* **Ingestion & Dual-Layer Extraction (`text_extraction.py`):** The operational boundary triggers via a `POST /upload-multiple` request at the FastAPI gateway (`main_api.py`). Raw stream buffers are converted to layout strings using `PyPDF2`. If zero-length text or an image-only scanned pattern is detected, the pipeline automatically spins up a parallel **OCR Fallback Worker** leveraging `pdf2image` and `pytesseract`.
* **Generative NLP Standardization (`gemini_api.py`):** Raw text configurations are processed by Google Gemini models utilizing a strict production failover chain (`gemini-2.5-flash` $\rightarrow$ `gemini-2.0-flash` $\rightarrow$ `gemini-2.0-flash-001`). Controlled prompt engineering enforces a zero-markdown deterministic JSON extraction matching the targeted schema.
* **Vector Alignment & Hybrid Inference (`ranking.py`):** The engine converts parsed profiles into unified dictionary models. It reloads the pre-trained `.pkl` weights to transform current applicant strings into identical dimensional spaces. The custom inference loop calculates:
  * **Algorithmic Classification:** The Random Forest probability prediction of landing a successful interview.
  * **Semantic Similarity:** The Cosine Similarity coefficient of the applicant's raw embedding against the calculated `hire_centroid`.
  
  The final ranking score is computed via a weighted ensemble formulation:
  $$Score = \alpha \cdot Score_{RandomForest} + \beta \cdot Score_{SemanticSimilarity}$$
  *(Default configuration benchmarks: $\alpha = 0.6$ for Machine Learning probability, $\beta = 0.4$ for Semantic Cosine Similarity)*.

### 3. Stage 3: Persistence & Interface Distribution
* **Transactional Persistence (`save_data_to_sql.py`):** Validated objects, operational tracking IDs, and calculated hybrid scores are bound inside an atomic insertion query and committed securely to the `Resume_AI` database instance within **Microsoft SQL Server** via authenticated Windows Connections (`pyodbc`).
* **Client-Side Analytics (React Workspace):** The API framework outputs a payload array to the single-page application layer. The **React + Vite** frontend parses data models across specialized layouts (`Home.jsx` for async uploads, `Dashboard.jsx` for metrics visualizations, and `History.jsx` for querying SQL records).

---

## 🛠️ Technical Stack & Infrastructure

- **Backend Architecture:** FastAPI (High-performance API routing)
- **Frontend Dashboard:** React + Vite (State management & components modularization)
- **Machine Learning & NLP Core:** Scikit-Learn, Pandas, NumPy, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Generative AI Agent:** Google GenAI SDK (`gemini-2.5-flash`, `gemini-2.0-flash`)
- **Database Engine:** Microsoft SQL Server (Connected via native `pyodbc` drivers)
- **Document Extractors:** PyPDF2, pdf2image, pytesseract (OCR)

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
└─ README.md                  # Project documentation
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

