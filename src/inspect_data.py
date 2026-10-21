from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "Telco-Customer-Churn.csv"
)

df = pd.read_csv(DATA_PATH)

# 1. 将累计费用转换成数值
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce",
)

# 2. 查看转换产生的缺失值
print("转换后的缺失值数量：", df["TotalCharges"].isna().sum())

# 3. 填补缺失值
mask = df["TotalCharges"].isna()
print("\n待填补记录：")
print(df.loc[mask, ["tenure", "TotalCharges", "Churn"]])

df.loc[mask, "TotalCharges"] = 0

# 4. 检查处理结果
print("\n处理后的字段类型：")
print(df["TotalCharges"].dtype)

print("\n处理后的缺失值数量：")
print(df["TotalCharges"].isna().sum())

print("\n处理后的累计费用描述统计：")
print(df["TotalCharges"].describe())