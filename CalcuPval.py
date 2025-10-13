from scipy.stats import hypergeom

import CreatGraph

# 计算一个数据集的Pval
#$def cacul_pval(data,cluster_result):
    # 求Pval需要：数据集KNN图的总度数，某类别内样本的总度数，该样本的总度数，该样本与该类别内样本关联的半边数

    # 1：获取所有样本节点的度（即关联边数）
    # knn_G=sum(dict(knn_G.degree()).values())

    # 2：获取单个节点的度
    # edge_count = knn_G.degree(node_id)
    # print(f"与节点 {node_id} 相关的边有 {edge_count} 条")
    # 2: 获取聚类得到的类别

    # 3：看被分到该类的样本有哪些

    # 4：看该类样本一共有多少个半边（度的和）

"""
计算单个节点与给定聚类的关联p值（基于Fisher's exact test）
返回:
    p_value: 节点与聚类的关联p值"""
def calculate_p_value_for_node(graph, cluster, node):
    # 节点是否在聚类中，计算方法不同
    if node in cluster:
        return calculate_p_value_for_node_in_cluster(graph, cluster, node)
    else:
        return calculate_p_value_for_node_outside_cluster(graph, cluster, node)


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

