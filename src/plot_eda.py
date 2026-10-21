from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 路径配置
# =========================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "Telco-Customer-Churn.csv"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 绘图风格
# =========================
sns.set_theme(style="whitegrid", context="notebook")

# 优先使用常见中文字体；若系统没有安装，会自动回退
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200

# =========================
# 读取和清洗数据
# =========================
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce",
).fillna(0)

df["ChurnLabel"] = df["Churn"].map({
    "No": "未流失",
    "Yes": "已流失",
})

def save_figure(filename):
    """保存当前图表并关闭，避免重复占用内存。"""
    output_path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"已保存：{output_path.relative_to(ROOT_DIR)}")


# =========================
# 图 1：客户流失比例
# =========================
plt.figure(figsize=(7, 5))

churn_counts = df["ChurnLabel"].value_counts().reindex(
    ["未流失", "已流失"]
)

ax = sns.barplot(
    x=churn_counts.index,
    y=churn_counts.values,
)

ax.set_title("电信客户流失情况", fontsize=15)
ax.set_xlabel("客户状态")
ax.set_ylabel("客户数量")

for i, value in enumerate(churn_counts.values):
    ax.text(i, value + 50, f"{value:,}", ha="center")

save_figure("01_churn_distribution.png")


# =========================
# 图 2：合同类型与流失率
# =========================
contract_order = [
    "Month-to-month",
    "One year",
    "Two year",
]

contract_churn = (
    df.groupby("Contract", observed=False)
    .agg(
        customer_count=("Churn", "size"),
        churn_rate=("Churn", lambda s: (s == "Yes").mean() * 100),
    )
    .reindex(contract_order)
    .reset_index()
)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=contract_churn,
    x="Contract",
    y="churn_rate",
)

ax.set_title("不同合同类型的客户流失率", fontsize=15)
ax.set_xlabel("合同类型")
ax.set_ylabel("流失率（%）")
ax.set_ylim(0, max(contract_churn["churn_rate"]) * 1.2)

for i, row in contract_churn.iterrows():
    ax.text(
        i,
        row["churn_rate"] + 0.8,
        f'{row["churn_rate"]:.2f}%',
        ha="center",
    )

save_figure("02_churn_by_contract.png")


# =========================
# 图 3：入网时长与流失率
# =========================
tenure_bins = [-1, 6, 12, 24, 48, 72]
tenure_labels = [
    "0-6个月",
    "7-12个月",
    "13-24个月",
    "25-48个月",
    "49-72个月",
]

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=tenure_bins,
    labels=tenure_labels,
)

tenure_churn = (
    df.groupby("TenureGroup", observed=False)
    .agg(
        customer_count=("Churn", "size"),
        churn_count=("Churn", lambda s: (s == "Yes").sum()),
    )
    .reset_index()
)

tenure_churn["churn_rate"] = (
    tenure_churn["churn_count"]
    / tenure_churn["customer_count"]
    * 100
)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=tenure_churn,
    x="TenureGroup",
    y="churn_rate",
)

ax.set_title("不同入网时长的客户流失率", fontsize=15)
ax.set_xlabel("入网时长")
ax.set_ylabel("流失率（%）")
ax.set_ylim(0, max(tenure_churn["churn_rate"]) * 1.2)

for i, row in tenure_churn.iterrows():
    ax.text(
        i,
        row["churn_rate"] + 0.8,
        f'{row["churn_rate"]:.2f}%',
        ha="center",
    )

save_figure("03_churn_by_tenure.png")


# =========================
# 图 4：流失与月费用的关系
# =========================
plt.figure(figsize=(8, 5))

ax = sns.boxplot(
    data=df,
    x="ChurnLabel",
    y="MonthlyCharges",
    order=["未流失", "已流失"],
)

ax.set_title("不同客户状态的月费用分布", fontsize=15)
ax.set_xlabel("客户状态")
ax.set_ylabel("月费用")

save_figure("04_monthly_charges_by_churn.png")


# =========================
# 图 5：网络类型与流失率
# =========================
internet_order = [
    "DSL",
    "Fiber optic",
    "No",
]

internet_churn = (
    df.groupby("InternetService", observed=False)
    .agg(
        customer_count=("Churn", "size"),
        churn_rate=("Churn", lambda s: (s == "Yes").mean() * 100),
    )
    .reindex(internet_order)
    .reset_index()
)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=internet_churn,
    x="InternetService",
    y="churn_rate",
)

ax.set_title("不同网络服务类型的客户流失率", fontsize=15)
ax.set_xlabel("网络服务类型")
ax.set_ylabel("流失率（%）")
ax.set_ylim(0, max(internet_churn["churn_rate"]) * 1.2)

for i, row in internet_churn.iterrows():
    ax.text(
        i,
        row["churn_rate"] + 0.8,
        f'{row["churn_rate"]:.2f}%',
        ha="center",
    )

save_figure("05_churn_by_internet_service.png")


# =========================
# 图 6：技术支持与流失率
# =========================
support_order = [
    "No",
    "Yes",
    "No internet service",
]

support_churn = (
    df.groupby("TechSupport", observed=False)
    .agg(
        customer_count=("Churn", "size"),
        churn_rate=("Churn", lambda s: (s == "Yes").mean() * 100),
    )
    .reindex(support_order)
    .reset_index()
)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=support_churn,
    x="TechSupport",
    y="churn_rate",
)

ax.set_title("不同技术支持状态的客户流失率", fontsize=15)
ax.set_xlabel("技术支持")
ax.set_ylabel("流失率（%）")
ax.set_ylim(0, max(support_churn["churn_rate"]) * 1.2)

for i, row in support_churn.iterrows():
    ax.text(
        i,
        row["churn_rate"] + 0.8,
        f'{row["churn_rate"]:.2f}%',
        ha="center",
    )

save_figure("06_churn_by_tech_support.png")

print("\n全部 EDA 图表生成完毕！")
