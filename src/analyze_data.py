from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "Telco-Customer-Churn.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# 修复 TotalCharges 的数据类型与空白值
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce",
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial"
]
plt.rcParams["axes.unicode_minus"] = False

# 1. 查看目标变量分布
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)
plt.show()

# 2. 比较流失与未流失客户的入网时长
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    element="step",
    stat="density",
    common_norm=False,
)
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure (months)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tenure_distribution.png", dpi=150)
plt.show()

# 3. 计算不同合同类型的流失率
contract_churn = (
    df.groupby("Contract", observed=True)["Churn"]
    .apply(lambda s: (s == "Yes").mean())
    .sort_values(ascending=False)
)

print("\n不同合同类型的流失率：")
print((contract_churn * 100).round(2).astype(str) + "%")

plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="Contract",
    y=df["Churn"].eq("Yes").astype(int),
    errorbar=None,
)
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "contract_churn_rate.png", dpi=150)
plt.show()

# 4. 查看流失与未流失客户的月费用
print("\n不同流失状态的月费用统计：")
print(
    df.groupby("Churn")["MonthlyCharges"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print(f"\n图表已保存至：{OUTPUT_DIR}")