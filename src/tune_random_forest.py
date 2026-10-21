from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce",
).fillna(0)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

configs = [
    {"name": "A_depth10", "max_depth": 10, "min_samples_leaf": 2},
    {"name": "B_depth20", "max_depth": 20, "min_samples_leaf": 2},
    {"name": "C_leaf4", "max_depth": None, "min_samples_leaf": 4},
]

results = []

for config in configs:
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)

    results.append({
        "model": config["name"],
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_precision": precision_score(
            y_val, val_pred, zero_division=0
        ),
        "val_recall": recall_score(
            y_val, val_pred, zero_division=0
        ),
        "val_f1": f1_score(y_val, val_pred, zero_division=0),
        "val_roc_auc": roc_auc_score(y_val, val_prob),
    })

results_df = pd.DataFrame(results).sort_values(
    "val_roc_auc", ascending=False
)

print("\n===== 验证集参数比较 =====")
print(results_df.to_string(index=False))

# 根据验证集 ROC-AUC 选择的参数
best_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

# 使用完整训练集重新训练
best_model.fit(X_train_full, y_train_full)

# 测试集只做最终评估
test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.5).astype(int)

print("\n===== 最终测试集评估：A_depth10 =====")
print(f"Accuracy : {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, test_pred, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, test_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, test_prob):.4f}")