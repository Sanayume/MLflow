
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# 定义输出目录
output_dir = '/sandbox/outputs/complex_test/'
os.makedirs(output_dir, exist_ok=True)

print("--- 开始复杂沙盒测试 ---")

# 1. 生成模拟数据
print("1. 正在生成模拟数据...")
np.random.seed(42) # 为了结果可复现

num_samples = 1000
num_features = 5

# 生成特征数据
X = np.random.rand(num_samples, num_features) * 10

# 生成目标变量，与特征有线性关系并加入噪声
true_coefficients = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
y = X @ true_coefficients + np.random.randn(num_samples) * 2 + 50 # 加上截距和噪声

# 转换为DataFrame
feature_names = [f'feature_{i+1}' for i in range(num_features)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"   生成了 {num_samples} 样本，{num_features} 特征的数据集。")
print(f"   数据集前5行：\n{df.head()}")

# 保存原始数据到输出目录
data_output_path = os.path.join(output_dir, 'simulated_data.csv')
df.to_csv(data_output_path, index=False)
print(f"   原始模拟数据已保存到: {data_output_path}")

# 2. 数据预处理 - 分割训练集和测试集
print("\n2. 正在分割训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df['target'], test_size=0.2, random_state=42)
print(f"   训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

# 2. 数据预处理 - 特征标准化
print("   正在对特征进行标准化处理...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   特征标准化完成。")

# 3. 模型训练 - 线性回归
print("\n3. 正在训练线性回归模型...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("   模型训练完成。")
print(f"   模型截距: {model.intercept_:.2f}")
print(f"   模型系数: {np.round(model.coef_, 2)}")

# 4. 模型评估
print("\n4. 正在评估模型性能...")
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"   均方误差 (MSE): {mse:.4f}")
print(f"   R-squared (R2) 分数: {r2:.4f}")

# 5. 结果保存
print("\n5. 正在保存模型和评估结果...")

# 保存模型
model_output_path = os.path.join(output_dir, 'linear_regression_model.pkl')
joblib.dump(model, model_output_path)
print(f"   模型已保存到: {model_output_path}")

# 保存评估结果
results_df = pd.DataFrame({
    'Metric': ['Mean Squared Error', 'R-squared'],
    'Value': [mse, r2]
})
results_output_path = os.path.join(output_dir, 'evaluation_results.csv')
results_df.to_csv(results_output_path, index=False)
print(f"   评估结果已保存到: {results_output_path}")

print("\n--- 复杂沙盒测试完成 ---")
