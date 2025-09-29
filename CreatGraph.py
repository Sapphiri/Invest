import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from DataNormalization import filepath

# 加载归一化数据
def load_normalized_data(dataset_name, method_name="minmax"):
    filename=f"{dataset_name}_{method_name}_data.txt"
    try:
        data=np.loadtxt(f"{filepath}/{filename}")
        print(f"成功加载 {dataset_name}的 {method_name} 归一化数据")
        return data
    except FileNotFoundError:
        print(f"找不到文件: {filename}")
        print("请先运行 DataNormalize.py 生成归一化数据")
        return None

def get_knn_neighbors(data, k):
    n = len(data)
    # 初始化n×k的数组存储近邻索引
    knn_neighbors = np.zeros((n, k), dtype=int)
    data_matrix=distance_matrix(data,data)
    for i in range(n):
        # 获取i到所有节点的距离（含自身）
        distances=data_matrix[i]
        # 2. 生成带索引的距离对 (距离值, 节点索引)
        index_distances = [(idx, distances) for idx, distances in enumerate(distances)]
        # 按距离排序，再筛选掉索引为i的元素，最后取前k个
        sorted_distances = sorted(index_distances, key=lambda x: x[1])
        filtered_distances = [sd for sd in sorted_distances if sd[0]!= i]
        nearest_index=[fd[0] for fd in filtered_distances[:k]]
        # print(f'i: {i}, 近邻索引: {nearest_index}')
        knn_neighbors[i]=nearest_index
    return knn_neighbors

# 创建knn图
def create_knn_graph(data,k):
    data_matrix = distance_matrix(data, data)
    knn_neighbors=get_knn_neighbors(data, k)
    # 无向图 <class 'networkx.classes.graph.Graph'>
    knn_G=nx.Graph()
    # 添加节点
    for i in range(len(data)):
        knn_G.add_node(i)
    # 仅添加双向边
    for i in range(len(data)):
        # 遍历i的所有近邻j
        for j in knn_neighbors[i]:
            # 检查i是否在j的近邻中
            if i in knn_neighbors[j]:
                knn_G.add_edge(i, j,weight=data_matrix[i][j])

    print(f"图创建完成: {len(knn_G.nodes())} 个节点, {len(knn_G.edges())} 条边")
    # 获取所有边
    edges=knn_G.edges.data()
    print("所有边：")
    for u,v,data in edges:
        print(f"节点{u}和节点{v}相连,属性：{data}")
    return knn_G

def main():
    """主函数"""
    print("基于归一化数据创建最近邻图")
    # 加载并输出归一化数据集
    datasets=['Iris']
    for dataset_name in datasets:
        data=load_normalized_data(dataset_name, method_name="minmax")
        print(f"\n{dataset_name} 数据集:")
        print(f"数据形状: {data.shape}",type(data))
        print(data,"\n")
        create_knn_graph(data,3)

if __name__ == "__main__":
    main()