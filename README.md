# AI-Powered CV Screening & Ranking Pipeline

An end-to-end, enterprise-grade CV screening solution that combines traditional Machine Learning, Semantic Embeddings, and Generative AI to automate, score, and rank candidate resumes in real-time. 

The system features a **Hybrid Scoring Engine** (Random Forest + SentenceTransformers) alongside an LLM-powered parser using Google Gemini, wrapped in a high-performance FastAPI backend and an intuitive React/Vite dashboard.

---

## 📊 System Architecture & Workflow

The pipeline is split into three main stages: **Data Preparation & Model Training**, **Real-Time Production Inference**, and **Frontend Analytics**.
---

## 🛠️ Features & Methodology

### 1. Training & Core Machine Learning (`STAGE 1`)
- **Data Preprocessing:** Standardizes raw resumes, handles categorical text fields, and applies advanced vectorization (`TF-IDF` with tuned `max_features` and `ngram_ranges` for skills, education, certifications, and roles) alongside `StandardScaler` for experience metrics.
- **Model Selection:** Utilizes a custom-tuned **Random Forest Classifier** selected based on feature importance indicators and its robust capacity to handle complex, non-linear text-categorical relationships.
- **Semantic Clustering:** Leverages `SentenceTransformer` (`all-MiniLM-L6-v2`) to build a high-dimensional `hire_centroid` embedding matrix from historically successful candidates.

### 2. Real-Time Production Inference (`STAGE 2`)
- **Dual-Layer Text Extraction:** Fast extraction via `PyPDF2` with an automated fallback mechanism to an **OCR Pipeline** using `pdf2image` and `pytesseract` for scanned documents.
- **LLM-Powered JSON Parsing:** Integrates a robust fallback chain of Google Gemini models (`gemini-2.5-flash` $\rightarrow$ `gemini-2.0-flash`) using strict prompt engineering to parse unstructured text into clean, standardized JSON objects.
- **Hybrid Scoring Engine:** Calculates a highly calibrated score bounded within $[0, 1]$ using a weighted hybrid formula:
  $$Score = \alpha \cdot Score_{RandomForest} + \beta \cdot Score_{SemanticSimilarity}$$
  *(Default configuration: 60% Machine Learning probability, 40% Semantic Cosine Similarity)*
- **Data Persistence:** Automatically commits validated candidate features and computed scores into a **Microsoft SQL Server** database instance via trusted Windows Authentication connection strings.

### 3. Frontend & Analytics Dashboard (`STAGE 3`)
- Built on top of **React + Vite** providing an asynchronous interface for multiple file uploads.
- Displays comprehensive data analytics tables, real-time status monitors, filterable score distributions, and comprehensive processing history extracted straight from the SQL database.

---

## 📁 Repository Structure

```text
├── data/                      # Raw datasets (AI_Resume_Screening.csv)
├── pkl/                       # Serialized model checkpoints & transformers (.pkl)
├── notebooks/                 # Development & Experimental Workflows
│   ├── data_cleaning.ipynb    # Data cleaning & feature vectorization pipeline
│   ├── visualize.ipynb        # Exploratory Data Analysis (EDA) & Model selection
│   └── random_forest.ipynb    # Random forest classifier training routines
├── src/                       # Production Backend & Processing Modules
│   ├── main_api.py            # Main FastAPI server and routing entry point
│   ├── text_extraction.py     # Multi-engine PDF/OCR extraction logic
│   ├── gemini_api.py          # Google Gemini structured inference agent
│   ├── ranking.py             # Hybrid Scoring & Vector transformation core
│   └── save_data_to_sql.py    # pyodbc database driver pipeline
├── frontend/                  # React/Vite Single-Page Application source
│   ├── package.json           # Frontend dependency manifest
│   └── src/components/        # Home, Dashboard, History UI components
├── requirements.txt           # Python application dependencies
└── README.md                  # Project documentation
```
## 🚀 Environment Setup & Installation
Backend Setup
Clone the repository and install the mandatory Python frameworks:
```
Bash
   pip install pandas scikit-learn joblib sentence-transformers google-generativeai
   pip install PyPDF2 pdf2image pytesseract pyodbc
```
Ensure you have Tesseract OCR installed locally on your operating system and append it to your system PATH variables.

## Configure your API environment keys:
```
Bash
   set GEMINI_API_KEY=your_production_api_key_here
```
## Frontend Setup
Navigate to the frontend directory, install npm packages, and run the client-side server:
```
Bash
   cd frontend
   npm install
   npm run dev
```
