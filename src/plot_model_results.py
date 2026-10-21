
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. 项目路径与绘图设置
# =========================
ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT_DIR / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def save_figure(filename):
    """保存当前图表并关闭画布。"""
    plt.tight_layout()
    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存：{output_path}")


# =========================
# 2. 三个模型的测试集结果
# =========================
metrics = pd.DataFrame({
    "模型": [
        "逻辑回归",
        "原始随机森林",
        "优化随机森林",
    ],
    "Accuracy": [0.8055, 0.7601, 0.7523],
    "Precision": [0.6572, 0.5366, 0.5229],
    "Recall": [0.5588, 0.7059, 0.7647],
    "F1-score": [0.6040, 0.6097, 0.6211],
    "ROC-AUC": [0.7268, 0.8343, 0.8388],
})

metrics_long = metrics.melt(
    id_vars="模型",
    var_name="评估指标",
    value_name="得分",
)

plt.figure(figsize=(11, 6))
sns.barplot(
    data=metrics_long,
    x="评估指标",
    y="得分",
    hue="模型",
)
plt.ylim(0, 1)
plt.title("不同模型的测试集性能对比")
plt.xlabel("评估指标")
plt.ylabel("得分")
plt.legend(title="模型")
save_figure("07_model_comparison.png")


# =========================
# 3. 随机森林内置特征重要性
# =========================
feature_importance = pd.DataFrame({
    "特征": [
        "tenure",
        "TotalCharges",
        "MonthlyCharges",
        "Contract_Month-to-month",
        "Contract_Two year",
        "OnlineSecurity_No",
        "TechSupport_No",
        "InternetService_Fiber optic",
        "PaymentMethod_Electronic check",
        "Contract_One year",
        "OnlineBackup_No",
        "InternetService_DSL",
        "OnlineSecurity_Yes",
        "gender_Male",
        "PaperlessBilling_Yes",
    ],
    "重要性": [
        0.130294,
        0.122898,
        0.098068,
        0.096225,
        0.047299,
        0.046826,
        0.037687,
        0.036638,
        0.032555,
        0.018088,
        0.017440,
        0.013878,
        0.013861,
        0.013755,
        0.013509,
    ],
})

feature_importance = feature_importance.sort_values(
    "重要性", ascending=True
)

plt.figure(figsize=(10, 8))
sns.barplot(
    data=feature_importance,
    x="重要性",
    y="特征",
)
plt.title("随机森林内置特征重要性 Top 15")
plt.xlabel("特征重要性")
plt.ylabel("特征")
save_figure("08_feature_importance.png")


# =========================
# 4. 置换特征重要性
# =========================
permutation_importance = pd.DataFrame({
    "特征": [
        "Contract",
        "tenure",
        "InternetService",
        "TotalCharges",
        "OnlineSecurity",
        "TechSupport",
        "Dependents",
        "PaperlessBilling",
        "PaymentMethod",
        "Partner",
        "SeniorCitizen",
        "MultipleLines",
        "PhoneService",
        "StreamingMovies",
        "OnlineBackup",
    ],
    "平均重要性": [
        0.057078,
        0.027656,
        0.016629,
        0.014514,
        0.003539,
        0.002548,
        0.001092,
        0.000854,
        0.000833,
        0.000734,
        0.000479,
        -0.000472,
        -0.000588,
        -0.000602,
        -0.000801,
    ],
    "标准差": [
        0.003709,
        0.001339,
        0.006883,
        0.001635,
        0.002647,
        0.001517,
        0.000848,
        0.002557,
        0.002490,
        0.000401,
        0.000827,
        0.000473,
        0.000361,
        0.002478,
        0.003143,
    ],
})

permutation_importance = permutation_importance.sort_values(
    "平均重要性", ascending=True
)

plt.figure(figsize=(10, 8))
plt.barh(
    permutation_importance["特征"],
    permutation_importance["平均重要性"],
    xerr=permutation_importance["标准差"],
    capsize=3,
)
plt.axvline(x=0, linewidth=1)
plt.title("随机森林置换特征重要性 Top 15")
plt.xlabel("ROC-AUC 平均变化（含标准差）")
plt.ylabel("特征")
save_figure("09_permutation_importance.png")


print("\n模型对比与特征重要性图表生成完毕！")

