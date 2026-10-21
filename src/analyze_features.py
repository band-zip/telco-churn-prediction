
from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "Telco-Customer-Churn.csv"
)

df = pd.read_csv(DATA_PATH)

# 修复累计费用
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce",
).fillna(0)

# 1. 按入网时长分组，计算每组流失率
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 6, 12, 24, 48, 72],
    labels=["0-6个月", "7-12个月", "13-24个月", "25-48个月", "49-72个月"],
)

tenure_churn = (
    df.groupby("TenureGroup", observed=False)
    .agg(
        客户数量=("Churn", "size"),
        流失人数=("Churn", lambda s: (s == "Yes").sum()),
        流失率=("Churn", lambda s: (s == "Yes").mean()),
    )
)

tenure_churn["流失率"] = (tenure_churn["流失率"] * 100).round(2)

print("=== 不同入网时长的流失情况 ===")
print(tenure_churn)

# 2. 查看其他类别特征与流失率的关系
features = [
    "InternetService",
    "TechSupport",
    "OnlineSecurity",
    "PaymentMethod",
    "PaperlessBilling",
]

for feature in features:
    result = (
        df.groupby(feature, dropna=False, observed=False)["Churn"]
        .apply(lambda s: (s == "Yes").mean())
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    print(f"\n=== {feature}：各类别流失率（%）===")
    print(result)
