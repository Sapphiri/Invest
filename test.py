# import math
# import numpy as np
# from scipy.stats import hypergeom
#
# def comb(n, k):
#     """计算组合数 C(n, k)"""
#     if k < 0 or k > n:
#         return 0
#     return math.comb(n, k)
#
# def hypergeom_pmf(k, M, n, N):
#     """
#     手动计算超几何分布的概率质量函数
#     参数:
#     k: 抽样中的成功数
#     M: 总体大小
#     n: 总体中的成功数
#     N: 抽样数量
#     返回:
#     P(X = k)
#     """
#     if k < max(0, N - (M - n)) or k > min(n, N):
#         return 0.0
#
#     numerator = comb(n, k) * comb(M - n, N - k)
#     denominator = comb(M, N)
#
#     return numerator / denominator
#
#
# def hypergeom_sf_manual(k, M, n, N):
#     """
#     手动计算超几何分布的生存函数 (SF)
#     SF(k) = P(X >= k) = 1 - CDF(k-1)
#     参数:
#     k: 阈值
#     M, n, N: 超几何分布参数
#     返回:
#     P(X >= k)
#     """
#     if k > min(n, N):
#         return 0.0
#     # N - (M - n)：抽样中必须包含的最少成功数
#     if k <= max(0, N - (M - n)):
#         return 1.0
#
#     # 计算 P(X >= k) = sum_{i=k}^{min(n,N)} P(X = i)
#     sf_value = 0.0
#     for i in range(k, min(n, N) + 1):
#         sf_value += hypergeom_pmf(i, M, n, N)
#
#     return sf_value
#
#
# def hypergeom_cdf_manual(k, M, n, N):
#     """
#     手动计算超几何分布的累积分布函数 (CDF)
#     CDF(k) = P(X <= k)
#     """
#     if k < max(0, N - (M - n)):
#         return 0.0
#     if k >= min(n, N):
#         return 1.0
#
#     cdf_value = 0.0
#     for i in range(max(0, N - (M - n)), k + 1):
#         cdf_value += hypergeom_pmf(i, M, n, N)
#
#     return cdf_value
#
#
# def verify_hypergeom_functions():
#     """验证手动实现与scipy的一致性"""
#
#     test_cases = [
#         # (k, M, n, N, description)
#         (3, 20, 10, 5, "典型情况"),
#         (0, 20, 10, 5, "k=0边界"),
#         (5, 20, 10, 5, "k=最大值"),
#         (2, 50, 25, 10, "中等规模"),
#         (1, 10, 5, 3, "小规模")
#     ]
#
#     print("超几何分布函数验证")
#     print("=" * 80)
#
#     for k, M, n, N, desc in test_cases:
#         print(f"\n测试案例: {desc}")
#         print(f"参数: k={k}, M={M}, n={n}, N={N}")
#
#         # Scipy计算结果
#         try:
#             scipy_sf = hypergeom.sf(k - 1, M, n, N)  # 注意：scipy.sf(k-1) 对应 P(X >= k)
#             scipy_pmf = hypergeom.pmf(k, M, n, N)
#             scipy_cdf = hypergeom.cdf(k, M, n, N)
#         except:
#             scipy_sf = scipy_pmf = scipy_cdf = np.nan
#
#         # 手动计算结果
#         manual_sf = hypergeom_sf_manual(k, M, n, N)
#         manual_pmf = hypergeom_pmf(k, M, n, N)
#         manual_cdf = hypergeom_cdf_manual(k, M, n, N)
#
#         print(f"SF (P(X >= {k})):")
#         print(f"  Scipy:  {scipy_sf:.8f}")
#         print(f"  手动:   {manual_sf:.8f}")
#         print(f"  差异:   {abs(scipy_sf - manual_sf):.2e}")
#
#         print(f"PMF (P(X = {k})):")
#         print(f"  Scipy:  {scipy_pmf:.8f}")
#         print(f"  手动:   {manual_pmf:.8f}")
#         print(f"  差异:   {abs(scipy_pmf - manual_pmf):.2e}")
#
#         print(f"CDF (P(X <= {k})):")
#         print(f"  Scipy:  {scipy_cdf:.8f}")
#         print(f"  手动:   {manual_cdf:.8f}")
#         print(f"  差异:   {abs(scipy_cdf - manual_cdf):.2e}")
#
#         # 验证关系: SF(k) + CDF(k-1) = 1
#         if k > 0:
#             relation_check = manual_sf + manual_cdf - manual_pmf  # 应该是1
#             print(f"关系验证 SF(k)+CDF(k-1): {relation_check:.8f} (应该≈1)")
#
# verify_hypergeom_functions()