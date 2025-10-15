import os
os.environ["OMP_NUM_THREADS"]="1"
from DataNormalization import get_dataset_numbers
import numpy as np
from scipy.stats import hypergeom, alpha
import CreatGraph
import ClusterResults

"""
分析数据集整体聚类结果的统计显著性
返回：overall_clu_significance：字典：包含数据集整体聚类结果的统计显著性
"""
def overall_clustering_significance(dataset_name,knn_graphs,cluster_results,alpha=0.01):
    print('\n'+'*'*80)
    print(f'正在对 {dataset_name} 数据集聚类结果显著性进行分析：')
    cluster_significances=calculate_cluster_significance(knn_graphs[dataset_name],cluster_results[dataset_name],alpha)
    if not cluster_significances:
        return{
            'total_clusters':0,
            'significant_strict':0,
            'significant_relaxed':0,
            'significance_strict_rate':0,
            'significance_relaxed_rate':0,
            'average_significance_ratio':0
        }
    total_clusters=len(cluster_significances)
    significant_strict=sum(1 for cs in cluster_significances.values() if cs['is_significant_strict'])
    significant_relaxed=sum(1 for cs in cluster_significances.values() if cs['is_significant_relaxed'])
    average_significance_ratio=np.mean([cs['significance_ratio'] for cs in cluster_significances.values()])

    overall_clu_significance={
        'total_clusters':total_clusters,
        'significant_strict':significant_strict,
        'significant_relaxed':significant_relaxed,
        'significance_strict_rate':significant_strict/total_clusters,
        'significance_relaxed_rate':significant_relaxed/total_clusters,
        'average_significance_ratio':average_significance_ratio
    }
    return overall_clu_significance

"""
计算一个数据集整个聚类的统计显著性（基于FWER控制）
返回：cluster_significance: 字典，包含每个聚类的显著性分析结果
"""
def calculate_cluster_significance(graph,cluster_result,alpha=0.01):
    cluster_significances={}
    n_nodes=graph.number_of_nodes()
    clustering_dict = cluster_result['clustering_dict']
    # Bonferroni校正
    corrected_alpha=alpha/n_nodes
    # 遍历所有聚类
    for cluster_id, cluster_nodes in clustering_dict.items():
        # if len(cluster_nodes) < 2:
        #     # 跳过太小的聚类
        #     continue
        # 用于记录聚类中所有节点的p值（node,p_val)
        node_p_values=[]
        # 用于记录聚类中的显著节点 node
        significant_nodes=[]
        # 遍历该聚类中的所有节点
        for node in cluster_nodes:
            # 计算节点p值（节点在聚类中的情况）
            p_value=calculate_p_value_for_node_in_cluster(graph,cluster_nodes,node)
            node_p_values.append((node,p_value))
            # 判断节点是否显著（通过FWER校正）
            if p_value<=corrected_alpha:
                significant_nodes.append(node)
        # 计算该聚类节点的最小p值及该聚类的节点显著率
        min_p_value=min([p_val for _,p_val in node_p_values]) if node_p_values else 1.0
        significance_ratio=len(significant_nodes)/len(cluster_nodes)

        # 判断该聚类是否显著
        # 严格标准：所有节点都显著
        is_significant_strict=len(significant_nodes)==len(cluster_nodes)
        # 宽松标准：70%的节点显著
        is_significant_relaxed=significance_ratio>=0.7

        cluster_significances[cluster_id] = {
            'cluster_nodes':cluster_nodes,
            'corrected_alpha':corrected_alpha,
            'node_p_values':node_p_values,
            'min_p_value':min_p_value,
            'significant_nodes':significant_nodes,
            'significance_ratio':significance_ratio,
            'is_significant_strict':is_significant_strict,  # 聚类显著性严格标准
            'is_significant_relaxed':is_significant_relaxed    # 聚类显著性宽松标准
        }
    print_cluster_significance(cluster_significances)
    return cluster_significances

"""
计算单个节点与给定聚类的关联p值（基于Fisher's exact test）
输入：graph:图对象    cluster_result：聚类结果字典   node：节点ID   cluster_id：指定的聚类ID
输出：p_value: 节点与聚类的关联p值
"""
def calculate_p_value_for_node(graph,cluster_result,node,cluster_id):
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
计算节点在聚类中时与聚类的关联p值
输入：graph:图对象    cluster：一个聚类的节点列表   node：节点ID
输出：p_value
"""
def calculate_p_value_for_node_in_cluster(graph,cluster,node):
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
计算节点不在聚类中时与聚类的关联p值
输入：graph:图对象    cluster：一个聚类的节点列表   node：节点ID
输出：p_value
"""
def calculate_p_value_for_node_outside_cluster(graph,cluster,node):
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
打印一个数据集聚类整体显著性分析结果
"""
def print_overall_clustering_significance(overall_stats,dataset_name=""):
    print("="*80)
    if dataset_name:
        print(f"数据集:{dataset_name}-聚类统计显著性分析")
    else:
        print("聚类统计显著性分析")
    print("="*80)

    # 打印整体统计
    print(f"整体统计：")
    print(f"总聚类数：{overall_stats['total_clusters']}")
    print(f"严格显著聚类数：{overall_stats['significant_strict']} ({overall_stats['significance_strict_rate']:.1%})")
    print(f"宽松显著聚类数：{overall_stats['significant_relaxed']} ({overall_stats['significance_relaxed_rate']:.1%})")
    print(f"平均显著性比例：{overall_stats['average_significance_ratio']:.1%}")
    print()

"""
打印一个数据集具体聚类显著性分析结果
"""
def print_cluster_significance(cluster_significances):
    # 打印每个聚类的详细信息
    print("各聚类显著性详细信息:")
    for cluster_id, cs in cluster_significances.items():
        print("-"*80)
        print(f"聚类 {cluster_id}:")
        print(f"Bonferroni校正后的显著性水平：{cs['corrected_alpha']}:")
        print(f"聚类各节点p值：{cs['node_p_values']}:")
        print(f"最小p值：{cs['min_p_value']}")
        print(f"显著节点：{len(cs['significant_nodes'])}/{len(cs['cluster_nodes'])} ({cs['significance_ratio']:.1%})")
        print(f"严格显著：{'是' if cs['is_significant_strict'] else '否'}")
        print(f"宽松显著：{'是' if cs['is_significant_relaxed'] else '否'}")
        # 显示p值分布
        p_values=[p_val for _,p_val in cs['node_p_values']]
        print(f"p值范围: [{min(p_values)} : {max(p_values)}]")

def main():
    # 数据集
    datasets=['Iris','Wine','Breast Cancer']
    # 自定义KNN图K值
    knn_vals={
        'Iris':int(np.sqrt(get_dataset_numbers('Iris'))),
        'Wine': int(np.sqrt(get_dataset_numbers('Wine'))),
        'Breast Cancer':int(np.sqrt(get_dataset_numbers('Breast Cancer')))
    }
    # 自定义聚类K值
    k_vals={
        'Iris':3,
        'Wine':3,
        'Breast Cancer':2
    }
    knn_graphs=CreatGraph.create_knn_graphs(datasets,knn_vals)
    cluster_results=ClusterResults.perform_kmeans(datasets,k_vals) # k-means聚类结果
    # cluster_results=ClusterResults.perform_hierarchical(datasets,k_vals) # 层次聚类结果
    # cluster_results=ClusterResults.perform_spectral(datasets,k_vals) # 谱聚类结果
    alpha_val=0.05
    for dataset_name in datasets:
        # print(calculate_p_value_for_node(knn_graphs[dataset_name],cluster_results[dataset_name],0,0))
        overall_stats=overall_clustering_significance(dataset_name,knn_graphs,cluster_results,alpha_val)
        print_overall_clustering_significance(overall_stats,dataset_name)
if __name__=="__main__":
    main()