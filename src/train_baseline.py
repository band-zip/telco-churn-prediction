from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# 1. 加载数据
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

# 2. 数据清洗
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce",
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 3. 构造特征X和目标y
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No":0, "Yes":1}) # y 要么 yes 要么 no

# 4. 自动区分数值特征和类别特征
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

print("数值特征：", numeric_features)
print("类别特征：", categorical_features)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("训练集大小：", X_train.shape)
print("数据集大小：", X_test.shape)

# 6. 定义预处理
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

# 7. 建立 Pipeline，避免预处理时发生数据泄漏
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)
# 8. 训练模型
model.fit(X_train, y_train)

# 9.预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

print("\n===== 分类阈值实验 =====")

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_test, y_pred_threshold, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred_threshold, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred_threshold, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred_threshold, labels=[0, 1]
    ).ravel()

    print(f"\n阈值：{threshold}")
    print(f"Precision：{precision:.4f}")
    print(f"Recall：   {recall:.4f}")
    print(f"F1：       {f1:.4f}")
    print(f"误报 FP：  {fp}")
    print(f"漏报 FN：  {fn}")


# 输出评价指标
# 10. 输出评价指标
print("\n===== 基线模型评估 =====")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

print("\n分类报告：")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["未流失", "已流失"],
        zero_division=0,
    )
)

print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))