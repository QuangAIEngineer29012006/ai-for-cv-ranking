# Training & Inference Pipeline Architecture

## Stage 1: Model Training (One-time)

```mermaid
graph TD
    A["📥 Input: AI_Resume_Screening.csv"] --> B["🔧 Step 1: Data Cleaning<br/>data_cleaning.ipynb<br/>- Drop columns<br/>- Handle missing values<br/>- Encode labels"]
    
    B --> C["💾 Output: cleaned_data.csv<br/>TF-IDF + Scaler (pkl/)"]
    
    C --> D["📊 Step 2: Data Visualization<br/>visualize.ipynb<br/>- EDA<br/>- Feature analysis<br/>- Decision: Choose Random Forest"]
    
    D --> E["🤖 Step 3: Model Training<br/>random_forest.ipynb<br/>- Train RF classifier<br/>- Load embeddings<br/>- Build centroid"]
    
    E --> F["💾 Output: pkl/<br/>- random_forest_model.pkl<br/>- tfidf_*.pkl<br/>- scaler.pkl<br/>- SentenceTransformer"]
    
    style A fill:#e1f5ff
    style C fill:#fff9c4
    style D fill:#f3e5f5
    style E fill:#e8f5e9
    style F fill:#fff3e0
```

## Stage 2: Real-time Inference (Production)

```mermaid
graph TD
    A["📤 API: POST /upload-multiple<br/>Multiple PDF files"] --> B["🔄 For each PDF:"]
    
    B --> C["1️⃣ TEXT EXTRACT<br/>text_extraction.py<br/>PyPDF2 → OCR fallback<br/>Output: raw_text"]
    
    C --> D["2️⃣ GEMINI PARSE<br/>gemini_api.py<br/>LLM structured extraction<br/>Output: JSON"]
    
    D --> E["3️⃣ FORMAT CONVERT<br/>main.py<br/>Map JSON → ranking format<br/>Output: cv_dict"]
    
    E --> F["4️⃣ HYBRID SCORE<br/>ranking.py<br/>TF-IDF + RF + Embeddings<br/>score = 0.6×RF + 0.4×semantic"]
    
    F --> G["5️⃣ DB SAVE<br/>save_data_to_sql.py<br/>INSERT to Resume_AI<br/>Output: stored"]
    
    G --> H["6️⃣ API RESPONSE<br/>main_api.py<br/>Return JSON scores<br/>status: success/failed"]
    
    H --> I["📊 Display Results<br/>React Dashboard<br/>Real-time score display"]
    
    style A fill:#e1f5ff
    style C fill:#fff9c4
    style D fill:#fce4ec
    style E fill:#f3e5f5
    style F fill:#e8f5e9
    style G fill:#ede7f6
    style H fill:#fff3e0
    style I fill:#c8e6c9
```

## Detailed Feature Engineering

```mermaid
graph LR
    A["CV Record"] --> B["Skills"]
    A --> C["Education"]
    A --> D["Certifications"]
    A --> E["Job Role"]
    A --> F["Experience<br/>Years"]
    A --> G["Projects<br/>Count"]
    
    B --> H["TF-IDF<br/>Skills<br/>300 features"]
    C --> I["TF-IDF<br/>Education<br/>20 features"]
    D --> J["TF-IDF<br/>Certifications<br/>50 features"]
    E --> K["TF-IDF<br/>Job Role<br/>30 features"]
    F --> L["Normalize<br/>via Scaler"]
    G --> L
    
    H --> M["Combine via<br/>hstack"]
    I --> M
    J --> M
    K --> M
    L --> M
    
    M --> N["Sparse Matrix<br/>X_combined"]
    
    N --> O["Random Forest<br/>Predict Proba"]
    
    A --> P["Embed Text<br/>SentenceTransformer"]
    P --> Q["Cosine Similarity<br/>vs hire_centroid"]
    
    O --> R["Final Score<br/>= 0.6×RF + 0.4×Semantic"]
    Q --> R
    
    style A fill:#e3f2fd
    style H fill:#fff9c4
    style I fill:#fff9c4
    style J fill:#fff9c4
    style K fill:#fff9c4
    style L fill:#f3e5f5
    style M fill:#fce4ec
    style N fill:#e0f2f1
    style O fill:#e8f5e9
    style Q fill:#e8f5e9
    style R fill:#fff3e0
```

## Folder Structure

```mermaid
graph TD
    A["📁 AI for CV_Score"]
    
    A --> B["🎯 Training Phase"]
    B --> B1["data/<br/>AI_Resume_Screening.csv"]
    B --> B2["data_cleaning.ipynb"]
    B --> B3["visualize.ipynb"]
    B --> B4["random_forest.ipynb"]
    B --> B5["logistic_regression.ipynb<br/>(alternative)"]
    B --> B6["cleaned_data.csv<br/>(output)"]
    
    A --> C["💾 Models"]
    C --> C1["pkl/"]
    C1 --> C1A["random_forest_model.pkl"]
    C1 --> C1B["tfidf_skills.pkl"]
    C1 --> C1C["tfidf_edu.pkl"]
    C1 --> C1D["tfidf_cert.pkl"]
    C1 --> C1E["tfidf_job.pkl"]
    C1 --> C1F["scaler.pkl"]
    C1 --> C1G["X_sparse.pkl, y.pkl"]
    
    A --> D["⚙️ Inference Pipeline"]
    D --> D1["main_api.py<br/>(FastAPI backend)"]
    D --> D2["main.py<br/>(main logic)"]
    D --> D3["text_extraction.py"]
    D --> D4["gemini_api.py"]
    D --> D5["ranking.py"]
    D --> D6["save_data_to_sql.py"]
    
    A --> E["🌐 Frontend"]
    E --> E1["src/"]
    E1 --> E1A["App.jsx"]
    E1 --> E1B["components/"]
    E1B --> E1B1["Sidebar.jsx"]
    E1 --> E1C["pages/"]
    E1C --> E1C1["Home.jsx"]
    E1C --> E1C2["Dashboard.jsx"]
    E1C --> E1C3["History.jsx"]
    E --> E2["package.json"]
    E --> E3["vite.config.js"]
    
    A --> F["🗄️ Database"]
    F --> F1["SQL Server<br/>Resume_AI"]
    F --> F2["CV_save_to_sql.sql<br/>(schema)"]
    
    A --> G["📚 Config"]
    G --> G1["env.txt<br/>(environment)"]
    G --> G2["PIPELINE.md<br/>(this doc)"]
    
    A --> H["🧪 Testing"]
    H --> H1["TestDriver.py"]
    
    style A fill:#fff9c4
    style B fill:#e1f5ff
    style C fill:#fff3e0
    style D fill:#fce4ec
    style E fill:#e8f5e9
    style F fill:#ede7f6
