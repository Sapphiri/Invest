import os
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from DataNormalization import load_normalized_data,get_dataset_classes

# 对所有数据集进行Kmeans聚类，返回聚类结果(dict)
def perform_kmeans(datasets,k_values):
    cluster_results={}
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"正在处理数据集(Kmeans)：{dataset_name}")
        # 获取归一化后的数据
        data = load_normalized_data(dataset_name,method_name="minmax")
        # 获取用户指定的K值，如果没有指定则使用默认值
        k = k_values.get(dataset_name,3)
        print(f"使用K值：{k}")
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
        y_pred = kmeans.fit_predict(data)
        # print(type(y_pred),y_pred)

        # 将标签数组转换为聚类字典格式
        cluster_dict={}
        for cluster_id in range(k):
            node_idx=np.where(y_pred==cluster_id)[0].tolist()
            cluster_dict[cluster_id]=node_idx
        # 存储结果
        cluster_results[dataset_name] ={
            'k':k,
            'labels':y_pred,  # 保持原始标签数组
            'clustering_dict':cluster_dict,  # 新增：聚类字典格式
            'cluster_sizes':[len(idx) for idx in cluster_dict.values()],  # 各聚类大小
            'kmeans_model':kmeans
        }
        print(f'{dataset_name}数据集Kmeans聚类已完成')
    return cluster_results

# 对所有数据集进行层次聚类，返回聚类结果(dict)
def perform_hierarchical(datasets,k_values):
    cluster_results={}
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"正在处理数据集(层次聚类)：{dataset_name}")
        # 获取归一化后的数据
        data=load_normalized_data(dataset_name,method_name="minmax")
        # 获取用户指定的K值，如果没有指定则使用默认值
        k=k_values.get(dataset_name,3)
        print(f"使用K值：{k}")
        # 执行层次聚类
        hierarchical=AgglomerativeClustering(n_clusters=k)
        y_pred=hierarchical.fit_predict(data)

        # 将标签数组转换为聚类字典格式
        cluster_dict={}
        for cluster_id in range(k):
            node_idx=np.where(y_pred==cluster_id)[0].tolist()
            cluster_dict[cluster_id]=node_idx
        # 存储结果
        cluster_results[dataset_name]={
            'k':k,
            'labels':y_pred,  # 保持原始标签数组
            'clustering_dict':cluster_dict,  # 聚类字典格式
            'cluster_sizes':[len(idx) for idx in cluster_dict.values()],  # 各聚类大小
            'hierarchical_model':hierarchical
        }
        print(f'{dataset_name}数据集层次聚类已完成')
    return cluster_results

# 对所有数据集进行谱聚类，返回聚类结果(dict)
def perform_spectral(datasets, k_values):
    cluster_results={}
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"正在处理数据集(谱聚类)：{dataset_name}")
        # 获取归一化后的数据
        data=load_normalized_data(dataset_name,method_name="minmax")
        # 获取用户指定的K值，如果没有指定则使用默认值
        k=k_values.get(dataset_name,3)
        print(f"使用K值: {k}")
        # 执行谱聚类
        spectral=SpectralClustering(n_clusters=k, random_state=42, affinity='rbf')
        y_pred=spectral.fit_predict(data)

        # 将标签数组转换为聚类字典格式
        cluster_dict={}
        for cluster_id in range(k):
            node_idx=np.where(y_pred==cluster_id)[0].tolist()
            cluster_dict[cluster_id]=node_idx
        # 存储结果
        cluster_results[dataset_name]={
            'k':k,
            'labels':y_pred,  # 保持原始标签数组
            'clustering_dict':cluster_dict,  # 聚类字典格式
            'cluster_sizes':[len(idx) for idx in cluster_dict.values()],  # 各聚类大小
            'spectral_model':spectral  # 存储模型用于后续分析
        }
        print(f'{dataset_name}数据集谱聚类已完成')
    return cluster_results

# 用于输出所有数据集的聚类标签
def print_cluster_results(cluster_results):
    for dataset_name,data_cluster_result in cluster_results.items():
        y_pred=data_cluster_result['labels']
        print(f"\n{dataset_name} 数据集聚类结果：")
        print(y_pred,"\n")

def main():
    # 数据集
    datasets=['Iris','Wine','Breast Cancer']
    # 自定义K值
    k_values={
        'Iris': get_dataset_classes('Iris'),
        'Wine': 3,
        'Breast Cancer': 2
    }
    print("进行K-means聚类分析")
    cluster_results=perform_kmeans(datasets,k_values)
    print_cluster_results(cluster_results)
    print("进行层次聚类分析")
    cluster_results=perform_hierarchical(datasets,k_values)
    print_cluster_results(cluster_results)
    print("进行谱聚类分析")
    cluster_results=perform_spectral(datasets,k_values)
    print_cluster_results(cluster_results)
if __name__=="__main__":
    main()