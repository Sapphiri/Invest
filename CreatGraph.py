import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from DataNormalization import load_normalized_data

# 返回一个数据集最近k个节点的索引矩阵（n维数组）
def get_knn_neighbors(data, k):
    n = len(data)
    # 初始化n×k的数组存储近邻索引
    knn_neighbors = np.zeros((n, k), dtype=int)
    data_matrix=distance_matrix(data,data)
    for i in range(n):
        # 获取i到所有节点的距离（含自身）
        distances=data_matrix[i]
        # 生成索引-距离对列表 (节点索引，距离)
        index_distances=[(idx, distances) for idx, distances in enumerate(distances)]
        # 按距离排序，再筛选掉索引为i的元素，最后取前k个索引
        sorted_distances=sorted(index_distances, key=lambda x: x[1])
        filtered_distances=[sd for sd in sorted_distances if sd[0]!= i]
        nearest_index=[fd[0] for fd in filtered_distances[:k]]
        # print(f'i: {i}, 近邻索引: {nearest_index}')
        knn_neighbors[i]=nearest_index
    return knn_neighbors

# 对一个数据集创建并返回knn图
def create_knn_graph(data,k):
    # 计算样本间的欧几里得距离矩阵
    data_matrix=distance_matrix(data, data)
    # 计算样本的最近k个节点的索引矩阵
    knn_neighbors=get_knn_neighbors(data, k)
    # 创建无向图 <class 'networkx.classes.graph.Graph'>
    knn_G=nx.Graph()
    # 添加节点（个数为数据集样本数）
    for i in range(len(data)):
        knn_G.add_node(i)
    # 搜索各节点的KNN近邻，仅添加双向边
    for i in range(len(data)):
        # 遍历i的所有近邻j
        for j in knn_neighbors[i]:
            # 检查i是否在j的近邻中
            if i in knn_neighbors[j]:
                knn_G.add_edge(i, j,weight=data_matrix[i][j])

    print(f"图创建完成: {len(knn_G.nodes())} 个节点, {len(knn_G.edges())} 条边")
    # # 获取所有边
    # edges=knn_G.edges.data()
    # print("所有边：")
    # for u,v,data in edges:
    #     print(f"节点{u}和节点{v}相连,属性：{data}")
    return knn_G

# 对数据集合创建并返回knn图字典
def create_knn_graphs(datasets,k_vals):
    knn_graphs={}  # 用于存储所有图的字典
    for dataset_name in datasets:
        print(f"\n{'=' * 50}")
        print(f"正在处理数据集(KNN_Graph): {dataset_name}")
        data=load_normalized_data(dataset_name, method_name="minmax")
        print(f"数据形状: {data.shape}",type(data))
        # print(data,"\n")
        k = k_vals.get(dataset_name, 3)
        knn_G=create_knn_graph(data,k)
        # 将图存储到字典中，以数据集名为键
        knn_graphs[dataset_name]=knn_G
    return knn_graphs

def main():
    """主函数"""
    print("基于归一化数据创建最近邻图")
    # 加载并输出归一化数据集
    datasets=['Iris']
    k_vals={
        'Iris':3
    }
    knn_graphs=create_knn_graphs(datasets,k_vals)
    for dataset_name in datasets:
        print(f"图创建完成: {len(knn_graphs[dataset_name].nodes())} 个节点, {len(knn_graphs[dataset_name].edges())} 条边")
        #print('临接节点：', list(knn_graphs[dataset_name].neighbors(149)))
        # D = sum(dict(knn_graphs[dataset_name].degree()).values())
        # print(knn_graphs[dataset_name].degree())
        # print(D)
        # DS=2 * knn_graphs[dataset_name].number_of_edges()
        # print(DS)
if __name__ == "__main__":
    main()