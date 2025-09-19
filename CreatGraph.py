import numpy as np
from DataNormalization import filepath

#加载归一化数据
def load_normalized_data(dataset_name, method_name="minmax"):
    filename =  f"{dataset_name}_{method_name}_data.txt"
    try:
        data = np.loadtxt(f"{filepath}/{filename}")
        print(f"成功加载 {dataset_name}的 {method_name} 归一化数据")
        return data
    except FileNotFoundError:
        print(f"找不到文件: {filename}")
        print("请先运行 DataNormalize.py 生成归一化数据")
        return None

def main():
    """主函数"""
    print("基于归一化数据创建最近邻图")
    # 加载并输出归一化数据集
    datasets = ['Iris', 'Wine', 'Breast Cancer']
    for dataset_name in datasets:
        data=load_normalized_data(dataset_name, method_name="minmax")
        print(f"\n{dataset_name} 数据集:")
        print(f"数据形状: {data.shape}")
        print(data,"\n")

if __name__ == "__main__":
    main()