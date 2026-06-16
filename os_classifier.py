import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

TRAIN_CSV   = "tickets_sintetico_6000.csv"
MODEL_DIR   = Path("model")
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

CHANNEL_W = {"Telefone": 3, "Chat": 2, "Email": 1}
TICKET_W  = {"Problema Técnico": 4, "Reclamação": 3, "Solicitação de reembolso": 2,
             "Cancelamento": 1, "Dúvida": 0}

_CRIT_ANCHORS = [
    "entretenimento, lazer, hobby, item recreativo não essencial para a rotina",
    "eletrodoméstico auxiliar, conforto pessoal, dispositivo secundário, não indispensável",
    "ferramenta de trabalho e produtividade, comunicação diária indispensável",
    "equipamento vital para alimentação, segurança ou saúde, risco de incêndio ou acidente",
]
_POLAR_ANCHORS = [
    "urgência, perigo, risco de incêndio, quebrado, emergência, estragou",  # polo grave
    "dúvida, normal, informação, elogio, funcionamento perfeito",           # polo neutro
]

_anchor_embs: np.ndarray | None = None
_polar_embs:  np.ndarray | None = None


def _crit_matrix(encoder: SentenceTransformer) -> np.ndarray:
    global _anchor_embs
    if _anchor_embs is None:
        _anchor_embs = encoder.encode(_CRIT_ANCHORS, normalize_embeddings=True)
    return _anchor_embs


def _polar_matrix(encoder: SentenceTransformer) -> np.ndarray:
    global _polar_embs
    if _polar_embs is None:
        _polar_embs = encoder.encode(_POLAR_ANCHORS, normalize_embeddings=True)
    return _polar_embs


def _product_risk(names: pd.Series, encoder: SentenceTransformer) -> np.ndarray:
    embs = encoder.encode(names.fillna("produto desconhecido").tolist(), normalize_embeddings=True)
    return (np.argmax(embs @ _crit_matrix(encoder).T, axis=1) + 1).astype(np.float32)


def _severity(text_embs: np.ndarray, encoder: SentenceTransformer) -> np.ndarray:
    """Score [0,1]: softmax sobre similaridades coseno com polo grave vs. neutro."""
    sims  = text_embs @ _polar_matrix(encoder).T              # (N, 2)
    exp_s = np.exp(sims - sims.max(axis=1, keepdims=True))    # numerically stable softmax
    return (exp_s[:, 0] / exp_s.sum(axis=1)).astype(np.float32)


def _struct(df: pd.DataFrame, encoder: SentenceTransformer) -> np.ndarray:
    return np.column_stack([
        _product_risk(df["Product_Purchased"], encoder),
        df["Ticket_Channel"].map(CHANNEL_W).fillna(1).astype(int),
        (df["Customer_Age"] >= 60).astype(int),
        df["Ticket_Description"].fillna("").apply(lambda x: len(x.split())),
        df["Ticket_Type"].map(TICKET_W).fillna(2).astype(int),
        df["Customer_Age"].fillna(30),
    ]).astype(np.float32)


def _build_X(df: pd.DataFrame, encoder: SentenceTransformer) -> np.ndarray:
    texts = (df["Ticket_Subject"].fillna("") + ". " + df["Ticket_Description"].fillna("")).tolist()
    embs  = encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False).astype(np.float32)
    sev   = _severity(embs, encoder).reshape(-1, 1)  # 7ª feature estruturada: severidade textual
    return np.hstack([_struct(df, encoder), sev, embs])


def train() -> tuple:
    MODEL_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TRAIN_CSV).dropna(subset=["Ticket_Description", "Ticket_Priority"])
    le = LabelEncoder().fit(df["Ticket_Priority"])
    y  = le.transform(df["Ticket_Priority"])

    encoder = SentenceTransformer(EMBED_MODEL)
    X = _build_X(df, encoder)

    # Split estratificado 60/20/20 — fonte única garante distribuição idêntica em todas as partições
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    # SMOTE estritamente na partição de treino — val e test veem apenas amostras reais
    X_tr_bal, y_tr_bal = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_tr, y_tr)

    # Cost-sensitive learning: penaliza FN na classe Urgente via sample_weight dinâmico
    urgente_idx = int(np.where(le.classes_ == "Urgente")[0][0])
    weights     = np.where(y_tr_bal == urgente_idx, 4.0, 1.0).astype(np.float32)

    model = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.75, colsample_bytree=0.65, min_child_weight=8,
        reg_lambda=2.0, reg_alpha=0.2, eval_metric="mlogloss",
        early_stopping_rounds=40, n_jobs=-1, verbosity=0, random_state=42,
    )
    model.fit(X_tr_bal, y_tr_bal, sample_weight=weights,
              eval_set=[(X_val, y_val)], verbose=False)

    val_f1 = f1_score(y_val, model.predict(X_val), average="macro")
    te_f1  = f1_score(y_te,  model.predict(X_te),  average="macro")
    print(f"Val  F1-Macro : {val_f1:.4f}")
    print(f"Test F1-Macro : {te_f1:.4f}  (gap: {abs(val_f1 - te_f1):.4f})")
    print(classification_report(y_te, model.predict(X_te), target_names=le.classes_))

    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(le,    MODEL_DIR / "label_encoder.pkl")
    return model, encoder, le


def load_artifacts() -> tuple:
    return (
        joblib.load(MODEL_DIR / "model.pkl"),
        SentenceTransformer(EMBED_MODEL),
        joblib.load(MODEL_DIR / "label_encoder.pkl"),
    )


def predict_new_ticket(ticket_data: dict, model, encoder: SentenceTransformer, le: LabelEncoder) -> dict:
    df    = pd.DataFrame([ticket_data])
    proba = model.predict_proba(_build_X(df, encoder))[0]
    idx   = int(np.argmax(proba))
    return {
        "priority":     le.classes_[idx],
        "confidence":   round(float(proba[idx]), 4),
        "distribution": {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)},
    }


if __name__ == "__main__":
    model, encoder, le = train()

    ticket = {
        "Product_Purchased":  "Geladeira",
        "Ticket_Type":        "Problema Técnico",
        "Ticket_Subject":     "Fumaça saindo da geladeira",
        "Ticket_Description": "Minha geladeira emite fumaça preta e cheiro forte de queimado. "
                              "Tenho 72 anos, moro sozinha e guardo insulina que precisa de refrigeração.",
        "Ticket_Channel":     "Telefone",
        "Customer_Age":       72,
    }
    print(predict_new_ticket(ticket, model, encoder, le))
