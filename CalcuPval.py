import os
os.environ["OMP_NUM_THREADS"] = "1"
from scipy.stats import hypergeom

import CreatGraph
import ClusterResults

"""
计算单个节点与给定聚类的关联p值（基于Fisher's exact test）
返回p_value: 节点与聚类的关联p值
"""
def calculate_p_value_for_node(graph, cluster_result, node,cluster_id):
    clustering_dict = cluster_result['clustering_dict']
    # 获取指定的聚类(list)
    cluster_nodes = clustering_dict.get(cluster_id, [])
    print(f"聚类{cluster_id}包含样本:{cluster_nodes}")
    if not cluster_nodes:
        raise ValueError(f"聚类 {cluster_id} 不存在或为空")
    # 节点是否在聚类中，计算方法不同
    if node in cluster_nodes:
        return calculate_p_value_for_node_in_cluster(graph, cluster_nodes, node)
    else:
        return calculate_p_value_for_node_outside_cluster(graph, cluster_nodes, node)

"""
计算节点不在聚类中时与聚类的关联p值
"""
def calculate_p_value_for_node_outside_cluster(graph, cluster, node):
    # 获取节点与聚类内部的连p接数
    k_in=sum(1 for neighbor in graph.neighbors(node) if neighbor in cluster)
    # 获取节点与聚类外部的连接数
    k_out=sum(1 for neighbor in graph.neighbors(node) if neighbor not in cluster)
    # 节点总度数
    k_i=k_in+k_out
    # 聚类内样本的总度数
    D_S=sum(graph.degree(n) for n in cluster)
    # 整个图的总度数
    D=2*graph.number_of_edges()

    # 使用超几何分布计算p值
    M=D-k_i  # 总体数量（总半边数-该样本半边数）
    n=D_S  # 成功总体数（聚类中样本的总半边数）
    N=k_i  # 抽样数量（该样本半边数）
    k=k_in  # 抽样中的成功数（节点与聚类的连接数）

    # p值：计算观察到至少k_in个连接的概率
    p_value=hypergeom.sf(k-1,M,n,N)

    return p_value

"""
计算节点在聚类中时与聚类的关联p值
"""
def calculate_p_value_for_node_in_cluster(graph, cluster, node):
    # 获取节点与聚类内部的连p接数
    k_in=sum(1 for neighbor in graph.neighbors(node) if neighbor in cluster)
    # 获取节点与聚类外部的连接数
    k_out=sum(1 for neighbor in graph.neighbors(node) if neighbor not in cluster)
    # 节点总度数
    k_i=k_in+k_out
    # 聚类内样本的总度数
    D_S=sum(graph.degree(n) for n in cluster)
    # 整个图的总度数
    D=2*graph.number_of_edges()

    # 使用超几何分布计算p值
    M=D-k_i  # 调整后的总体大小
    n=D_S-k_i  # 成功总体数
    N=k_i  # 抽样数量
    k=k_in  # 抽样中的成功数

    # p值：计算观察到至少k_in个连接的概率
    p_value=hypergeom.sf(k-1,M,n,N)

    return p_value

"""
    计算整个聚类的统计显著性
"""
def calculate_cluster_significance(graph, cluster, alpha=0.01, method='FWER'):
    n_nodes = graph.number_of_nodes()
    cluster_p_values = []

    # 计算聚类中所有节点的p值
    for node in cluster:
        p_val = calculate_p_value_for_node(graph, cluster, node)
        cluster_p_values.append((node, p_val))
def main():
    # 数据集
    datasets=['Iris']
    # 自定义KNN图K值
    knn_vals={
        'Iris':3
    }
    # 自定义聚类K值
    k_vals={
        'Iris': 3
    }
    knn_graphs = CreatGraph.create_knn_graphs(datasets, knn_vals)
    cluster_results = ClusterResults.perform_kmeans(datasets, k_vals)
    for dataset_name in datasets:
        print(calculate_p_value_for_node(knn_graphs[dataset_name],cluster_results[dataset_name],0,0))
if __name__ == "__main__":
    main()