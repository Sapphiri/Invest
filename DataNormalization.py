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
minmax_results = {}
standardized_results = {}
# 创建输出目录（如果不存在）
filepath = "DataResults"
os.makedirs(filepath, exist_ok=True)

def normalization_process(dataset):
    # 对数据进行归一化处理
    for name, loader in dataset.items():
        # 加载数据(特征数据矩阵，特征名称列表，目标标签向量，目标标签列表)
        datasets_data = loader()
        data = datasets_data.data
        x = datasets_data.feature_names
        y = datasets_data.target
        z = datasets_data.target_names

        # 创建归一化器实例
        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        # 最小-最大归一化
        data_minmax = minmax_scaler.fit_transform(data)
        # 标准化
        data_standard = standard_scaler.fit_transform(data)

        # 存储结果
        minmax_results[name] = {
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
    # print(minmax_results['Iris']['data'][:3])


#用于输出所有数据集的最大-最小归一化结果
def print_minmax_results():
    for dataset_name, data_results in minmax_results.items():
        data = data_results['data']
        print(f"\n{dataset_name} 数据集:")
        print(f"数据形状: {data.shape}")
        print(data,"\n")

#用于输出所有数据集的标准化结果
def print_standardized_results():
    for dataset_name, data_results in standardized_results.items():
        data = data_results['data']
        print(f"\n{dataset_name} 数据集:")
        print(f"数据形状: {data.shape}")
        print(data,"\n")

#用于保存所有数据集的归一化结果
def save_data_minmax(results, method_name):
    for dataset_name, data_results in results.items():
        filename = f"{dataset_name}_{method_name}_data.txt"
        np.savetxt(f"{filepath}/{filename}", data_results['data'])
        print(f"已保存{dataset_name}的{method_name}归一化数据: {filename}")
    print(f"\n所有{method_name}归一化数据已保存完毕！")

# 加载储存的归一化数据
def load_normalized_data(dataset_name, method_name="minmax"):
    filename=f"{dataset_name}_{method_name}_data.txt"
    try:
        data=np.loadtxt(f"{filepath}/{filename}")
        print(f"成功加载 {dataset_name}的 {method_name} 归一化数据")
        return data
    except FileNotFoundError:
        print(f"找不到文件: {filename}")
        print("请先运行生成归一化数据")
        return None


"""
根据数据集名称返回数据集的真实类别数
参数：dataset_name：数据集名称
返回：数据集的真实类别数，如果数据集不存在则返回 -1
"""
def get_dataset_classes(dataset_name):
    if dataset_name in datasets:
        # 加载数据集
        dataset=datasets[dataset_name]()
        # 返回类别数
        return len(dataset.target_names)
    else:
        print(f"错误: 数据集 '{dataset_name}' 不存在")
        print(f"可用的数据集：{list(datasets.keys())}")
        return -1
'''
返回样本数量
'''
def get_dataset_numbers(dataset_name):
    if dataset_name in datasets:
        # 加载数据集
        dataset=datasets[dataset_name]()
        # 返回类别数
        return dataset.data.shape[0]  # 样本数量
    else:
        print(f"错误: 数据集 '{dataset_name}' 不存在")
        print(f"可用的数据集：{list(datasets.keys())}")
        return -1

def main():
    #对数据进行归一化处理
    normalization_process(datasets)

    #输出所有数据集的最大最小归一化结果
    print_minmax_results()

    print(type(minmax_results),minmax_results)
    #保存所有数据集的归一化结果
    save_data_minmax(minmax_results, "minmax")
    #save_data_standard(standardized_results, "standard")
    print('Iris数据集的类别数为：',get_dataset_classes('Iris'))
    print('Iris数据集的样本数为：',get_dataset_numbers('Iris'))


if __name__ == "__main__":
    main()