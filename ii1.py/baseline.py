"""
baseline_tfidf_logreg_v2.py
─────────────────────────────────────────────────────────────
TF‑IDF (word + char n‑grams) + LogisticRegression
результат F1 (класс 1) ≈ 0.93–0.95 после подбора порога.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from sklearn.metrics import f1_score, precision_recall_curve, classification_report

# зачистим лейблы

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

train = train.dropna(subset=["label"])
train["label"] = train["label"].astype(int)

#  готовим данные

def combine(row):
    return f"{row['title'] or ''} {row['url'] or ''}".lower()

train["text"] = train.apply(combine, axis=1)
test ["text"] = test .apply(combine, axis=1)

# TF‑IDF векторизаторы (сразу fit на всём train)

tfidf_word = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2),
    max_features=30_000, min_df=2
)
tfidf_char = TfidfVectorizer(
    analyzer="char", ngram_range=(3, 5),
    max_features=10_000, min_df=2
)

tfidf_word.fit(train["text"])
tfidf_char.fit(train["text"])

X_word = tfidf_word.transform(train["text"])
X_char = tfidf_char.transform(train["text"])
X      = hstack([X_word, X_char])

X_test_word = tfidf_word.transform(test["text"])
X_test_char = tfidf_char.transform(test["text"])
X_test      = hstack([X_test_word, X_test_char])

y = train["label"].values

# отделяем выборку на valid с аналогичной консистенцией 0:1

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# обучаем

clf = LogisticRegression(
    class_weight="balanced",
    max_iter=600,
    solver="lbfgs",
    n_jobs=-1
)
clf.fit(X_tr, y_tr)

# подбор оптимального порога

probs_val = clf.predict_proba(X_val)[:, 1]

prec, rec, thr = precision_recall_curve(y_val, probs_val, pos_label=1)
f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
best_idx  = np.argmax(f1_scores)
best_thr  = thr[best_idx]
best_f1   = f1_scores[best_idx]

print(f"ЛУЧШИЙ порог = {best_thr:.3f}, F1_val = {best_f1:.4f}")

# Отчёт
y_val_pred = (probs_val >= best_thr).astype(int)
print(classification_report(y_val, y_val_pred, digits=3))

clf.fit(X, y)   # переобучим на всем train

test_probs = clf.predict_proba(X_test)[:, 1]
test_pred  = (test_probs >= best_thr).astype(int)

submission = pd.DataFrame({"ID": test["ID"], "label": test_pred})
submission.to_csv("submission_logreg_tfidf.csv", index=False)
