"""
本文件为数据预处理代码
主要流程：
1. 从清洗后的ratings数据中构建用户评分矩阵
2. 计算矩阵的稀疏度
3. 将矩阵保存到cleaned_data目录,供后续使用
"""

from pathlib import Path
import pandas as pd

def load_cleaned_ratings(project_root: Path) -> pd.DataFrame:
    """读取清洗后的ratings数据"""
    ratings_path = project_root / "data" / "cleaned_data" / "cleaned_ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"未找到清洗后的评分文件: {ratings_path}")

    ratings = pd.read_csv(ratings_path)
    required_cols = {"userId", "movieId", "rating"}
    if not required_cols.issubset(ratings.columns):
        raise ValueError(f"ratings 缺少必要列，至少需要: {required_cols}")

    return ratings

def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    构建用户评分矩阵：
    行: userId
    列: movieId
    值: rating
    """
    matrix = ratings.pivot(index="userId", columns="movieId", values="rating")
    return matrix

def save_user_item_matrix(matrix: pd.DataFrame, project_root: Path) -> Path:
    """保存用户评分矩阵到 data/cleaned_data 目录"""
    output_path = project_root / "data" / "cleaned_data" / "user_item_matrix.csv"
    matrix.to_csv(output_path, encoding="utf-8")
    return output_path

def matrix_sparsity(matrix: pd.DataFrame) -> float:
    """计算评分矩阵稀疏度"""
    return float(matrix.isna().sum().sum() / matrix.size)

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    ratings = load_cleaned_ratings(project_root)
    matrix = build_user_item_matrix(ratings)
    output_path = save_user_item_matrix(matrix, project_root)
    print(f"用户评分矩阵已构建至: {output_path}")
    print(f"矩阵形状: {matrix.shape}")
    print(f"矩阵稀疏度: {matrix_sparsity(matrix):.2%}")

if __name__ == "__main__":
    main()
