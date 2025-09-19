import os
import numpy as np
from sklearn.datasets import load_wine,load_iris,load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 定义要处理的数据集
datasets = {
    'Iris': load_iris,
    'Wine': load_wine,
    'Breast Cancer': load_breast_cancer
}

# 存储归一化结果
normalized_results = {}
standardized_results = {}

#对数据进行归一化处理
for name, loader in datasets.items():
    # 加载数据(特征数据矩阵，特征名称列表，目标标签向量，目标标签列表)
    datasets = loader()
    data = datasets.data
    x = datasets.feature_names
    y = datasets.target
    z = datasets.target_names

    # 创建新的归一化器实例
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # 最小-最大归一化
    data_minmax = minmax_scaler.fit_transform(data)
    # 标准化
    data_standard = standard_scaler.fit_transform(data)

    # 存储结果
    normalized_results[name] = {
        'data': data_minmax,
        'feature_names': x,
        'target': y,
        'target_names': z
    }

    standardized_results[name] = {
        'data': data_standard,
        'feature_names': x,
        'target': y,
        'target_names': z
    }

print("\n所有归一化处理完成！")

# # 输出特定数据集的归一化结果
# print("\nIris数据集归一化后的前3个样本:")
# print(normalized_results['Iris']['data'][:3])

#用于输出所有数据集的最大-最小归一化结果
def print_normalized_results():
    for name, data_dict in normalized_results.items():
        data = data_dict['data']
        print(f"\n{name} 数据集:")
        print(f"数据形状: {data.shape}")
        print(data,"\n")

# 创建文件夹（如果不存在）
filepath = "DataResults"
os.makedirs(filepath, exist_ok=True)
#用于保存所有数据集的归一化结果
def save_data_minmax(results, method_name):
    for name, data_results in results.items():
        filename = f"{name}_{method_name}_data.txt"
        np.savetxt(f"{filepath}/{filename}", data_results['data'])
        print(f"已保存{name}的{method_name}归一化数据: {filename}")
    print(f"\n所有{method_name}归一化数据已保存完毕！")

#输出有数据集的归一化结果
print_normalized_results()
#保存所有数据集的归一化结果
save_data_minmax(normalized_results, "minmax")
#save_data_minmax(standardized_results, "standard")