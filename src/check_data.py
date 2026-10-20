from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "Telco-Customer-Churn.csv"
)

df = pd.read_csv(DATA_PATH)

print("数据集形状：", df.shape)
print("数据前五行：", df.head())
print("数据字段类型：", df.dtypes)
print("缺失值数量：", df.isnull().sum())
print("目标变量分布：", df["Churn"].value_counts(dropna=False))


