import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix


# ================================
# Load models
# ================================

log_model = joblib.load("pkl/random_forest_model.pkl")

tfidf_skills = joblib.load("pkl/tfidf_skills.pkl")
tfidf_edu = joblib.load("pkl/tfidf_edu.pkl")
tfidf_cert = joblib.load("pkl/tfidf_cert.pkl")
tfidf_job = joblib.load("pkl/tfidf_job.pkl")

scaler = joblib.load("pkl/scaler.pkl")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ================================
# Build text for embedding
# ================================

def build_cv_text(row):

    return (
        str(row["Skills"]) + " " +
        str(row["Education"]) + " " +
        str(row["Certifications"]) + " " +
        str(row["Job Role"])
    )


# ================================
# Build hire centroid
# ================================

df = pd.read_csv("cleaned_data.csv")

cv_texts = df.apply(build_cv_text, axis=1).tolist()

cv_embeddings = embed_model.encode(
    cv_texts,
    normalize_embeddings=True
)

hire_embeddings = cv_embeddings[df["Recruiter Decision"] == 1]

hire_centroid = np.mean(hire_embeddings, axis=0)
hire_centroid = hire_centroid / np.linalg.norm(hire_centroid)


# ================================
# Main scoring function
# ================================

def hybrid_score(cv_dict, alpha=0.6, beta=0.4):

    row = pd.DataFrame([cv_dict])

    # TF-IDF
    X_sk = tfidf_skills.transform(row["Skills"])
    X_ed = tfidf_edu.transform(row["Education"])
    X_ce = tfidf_cert.transform(row["Certifications"])
    X_jb = tfidf_job.transform(row["Job Role"])

    X_num = scaler.transform(
        row[["Experience (Years)", "Projects Count"]].values
    )

    X_all = hstack([
        X_sk,
        X_ed,
        X_ce,
        X_jb,
        csr_matrix(X_num)
    ])

    rf_prob = log_model.predict_proba(X_all)[0][1]

    # ===== semantic score =====

    text = build_cv_text(row.iloc[0])

    new_embedding = embed_model.encode(
        [text],
        normalize_embeddings=True
    )[0]

    semantic_score = cosine_similarity(
        [new_embedding],
        [hire_centroid]
    )[0][0]

    # ===== final score =====

    final_score = alpha * rf_prob + beta * semantic_score

    # ===== explanation =====

    reasons = []

    exp = cv_dict["Experience (Years)"]
    projects = cv_dict["Projects Count"]
    skills = cv_dict["Skills"]

    if exp >= 3:
        reasons.append(f"Có kinh nghiệm {exp} năm")

    if projects >= 3:
        reasons.append(f"Có {projects} dự án thực tế")

    if "Machine Learning" in skills or "Deep Learning" in skills:
        reasons.append("Có kỹ năng Machine Learning")

    if semantic_score > 0.8:
        reasons.append("CV tương đồng với các ứng viên đã được tuyển")

    return {
        "final_score": float(final_score * 100),
        "rf_score": float(rf_prob * 100),
        "semantic_score": float(semantic_score * 100),
        "reasons": reasons
    }