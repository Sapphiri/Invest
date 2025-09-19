import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import load_wine,load_iris,load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#创建最小-最大归一化缩分器实例
scaler = MinMaxScaler()
#对iris数据集归一化
iris = load_wine()
x = iris.feature_names
y = iris.target
data= iris.data
Z = scaler.fit_transform(data)
# print("\n归一化后的前3个样本:")
# print(Z[:3])

#对wine数据集归一化
wine = load_wine()
x = wine.feature_names
y = wine.target
data= wine.data
Z = scaler.fit_transform(data)
# print("\n归一化后的前3个样本:")
# print(Z[:3])