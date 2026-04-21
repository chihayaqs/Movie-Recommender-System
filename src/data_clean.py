import pandas as pd
import os

base_path = os.path.dirname(os.path.abspath(__file__)) 
data_path1 = os.path.join(base_path, "../ml-latest-small/ratings.csv")
data_path2 = os.path.join(base_path, "../ml-latest-small/movies.csv")

# 1. 加载评分数据
ratings = pd.read_csv(data_path1)
movies = pd.read_csv(data_path2)

# 2. 检查基本信息
print("评分表基本情况：")
print(ratings.info())

# 3. 创建评分矩阵 (示例提取前1000条，否则太大不好看)
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# 4. 看看这个矩阵
print("\n生成的评分矩阵（部分）：")
print(user_movie_matrix.head())

# 看看有多少评分是缺失的
sparsity = user_movie_matrix.isnull().sum().sum() / user_movie_matrix.size
print(f"矩阵稀疏度: {sparsity:.2%}")