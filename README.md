# 🚀 AI CV Screening Pipeline

An intelligent resume screening and candidate ranking system powered by Machine Learning, Semantic Similarity, and Large Language Models (LLMs). The platform automates CV processing, extracts structured candidate information, evaluates applicant suitability, and provides recruitment insights through an interactive dashboard.

---

## 📌 Overview

Recruiters often spend significant time reviewing resumes manually. This project streamlines the hiring process by combining:

- OCR-based PDF text extraction
- Gemini-powered information parsing
- Machine Learning candidate evaluation
- Semantic similarity matching
- Automated ranking and scoring
- Real-time analytics dashboard

The system supports batch CV uploads and produces objective candidate scores to assist recruitment decision-making.

---

## 🏗️ System Architecture

```text
PDF Resume
    │
    ▼
Text Extraction (PyPDF2 / OCR)
    │
    ▼
Gemini Information Extraction
    │
    ▼
Structured Candidate Data
    │
    ▼
Feature Engineering (TF-IDF + Numeric Features)
    │
    ▼
Random Forest Prediction
    │
    ▼
Semantic Similarity Scoring
    │
    ▼
Hybrid AI Score
    │
    ▼
SQL Server Storage
    │
    ▼
React Dashboard Visualization
```

---

## ✨ Features

### 📄 Resume Processing
- PDF resume upload
- Multi-file batch processing
- OCR fallback for scanned documents
- Automatic text extraction

### 🤖 AI Information Extraction
- Candidate name extraction
- Skills detection
- Education parsing
- Certification identification
- Job role classification
- Experience calculation
- Project counting

### 🧠 Candidate Evaluation
- Random Forest classification
- TF-IDF feature representation
- Sentence Transformer embeddings
- Semantic similarity analysis
- Hybrid scoring strategy

### 📊 Analytics Dashboard
- Candidate ranking
- Score visualization
- Historical records
- Filtering and sorting
- Recruitment statistics

---

## 🛠️ Technology Stack

### Backend
- Python
- FastAPI
- Scikit-Learn
- Sentence Transformers
- Google Gemini API

### Data Processing
- Pandas
- NumPy
- TF-IDF Vectorization
- StandardScaler

### OCR & Document Processing
- PyPDF2
- Tesseract OCR
- pdf2image

### Database
- SQL Server
- pyodbc

### Frontend
- React
- Vite
- React Router

---

## 📂 Project Structure

```text
AI-CV-Screening/
│
├── backend/
│   ├── main_api.py
│   ├── ranking.py
│   ├── text_extraction.py
│   ├── gemini_api.py
│   ├── save_data_to_sql.py
│   └── pkl/
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── visualize.ipynb
│   └── random_forest.ipynb
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── data/
│   └── AI_Resume_Screening.csv
│
└── README.md
```

---

## ⚙️ Model Training Pipeline

### 1. Data Cleaning
- Remove irrelevant columns
- Handle missing values
- Encode target labels
- Generate TF-IDF features

### 2. Data Analysis
- Exploratory Data Analysis (EDA)
- Distribution visualization
- Correlation analysis
- Feature importance assessment

### 3. Model Training
- Random Forest Classifier
- Cross-validation
- Hyperparameter tuning
- Performance evaluation

### 4. Semantic Modeling
- SentenceTransformer
- Candidate embedding generation
- Hire centroid construction

---

## 🎯 Hybrid Scoring Formula

The final candidate score combines machine learning predictions and semantic similarity:

```text
Final Score =
α × RandomForest Score
+
β × Semantic Similarity Score
```

Default configuration:

```text
α = 0.6
β = 0.4
```

Where:

- Random Forest captures structured feature patterns
- Semantic Similarity measures alignment with successful candidates

---

## 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/your-username/AI-CV-Screening.git
cd AI-CV-Screening
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

```env
GEMINI_API_KEY=YOUR_API_KEY
```

### Run Backend

```bash
uvicorn main_api:app --reload
```

### Run Frontend

```bash
npm install
npm run dev
```

---

## 📈 Workflow

```text
Upload CV
     ↓
Extract Text
     ↓
Gemini Parsing
     ↓
Feature Transformation
     ↓
AI Evaluation
     ↓
Score Generation
     ↓
Database Storage
     ↓
Dashboard Display
```

---

## 🔮 Future Improvements

- Job description matching
- Explainable AI recommendations
- Resume quality feedback
- Multi-language CV support
- Advanced ranking algorithms
- RAG-enhanced candidate retrieval

---

## 👨‍💻 Authors

Developed as an AI-powered recruitment automation project for intelligent resume screening and candidate ranking.

---

## 📄 License

This project is released under the MIT License.
