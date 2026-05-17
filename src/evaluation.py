from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
	from .cf_recommender import (
		compute_item_similarity,
		predict_ratings_item_based,
		recommend_items_item_based,
	)
except ImportError:
	from cf_recommender import (
		compute_item_similarity,
		predict_ratings_item_based,
		recommend_items_item_based,
	)


DEFAULT_RELEVANCE_THRESHOLD = 4.0


def load_filtered_matrix(file_path: Optional[Path] = None) -> pd.DataFrame:
	"""读取过滤后的用户-电影评分矩阵，供评估流程使用。"""
	if file_path is None:
		file_path = Path(__file__).resolve().parents[1] / "data" / "cleaned_data" / "user_item_matrix_sparse_filtered.csv"
	if not file_path.exists():
		raise FileNotFoundError(f"矩阵文件不存在: {file_path}")

	matrix = pd.read_csv(file_path, index_col=0)
	try:
		matrix.index = matrix.index.astype(int)
	except Exception:
		pass
	try:
		matrix.columns = matrix.columns.astype(int)
	except Exception:
		pass
	return matrix.apply(pd.to_numeric, errors="coerce")


def train_test_split_matrix(
	matrix: pd.DataFrame,
	test_size: float = 0.2,
	random_state: int = 42,
	min_ratings_per_user: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""按用户随机划分训练集和测试集，并保持原始矩阵结构。

	每个满足最少评分数的用户都会随机抽出一部分已评分电影作为测试集；
	训练矩阵中对应位置置为 NaN，测试矩阵只保留被抽出的真实评分。
	"""
	if not 0 < test_size < 1:
		raise ValueError("test_size 必须在 (0, 1) 之间")

	matrix = matrix.apply(pd.to_numeric, errors="coerce")
	rng = np.random.default_rng(random_state)
	train = matrix.copy()
	test = pd.DataFrame(np.nan, index=matrix.index, columns=matrix.columns)

	for user_id in matrix.index:
		user_ratings = matrix.loc[user_id].dropna()
		n_ratings = len(user_ratings)
		if n_ratings < min_ratings_per_user:
			continue

		n_test = int(round(n_ratings * test_size))
		n_test = max(1, n_test)
		n_test = min(n_test, n_ratings - 1)
		if n_test <= 0:
			continue

		test_items = rng.choice(user_ratings.index.to_numpy(), size=n_test, replace=False)
		train.loc[user_id, test_items] = np.nan
		test.loc[user_id, test_items] = matrix.loc[user_id, test_items]

	return train, test


def _top_k_list(recommendations):
	"""从 Top-N 推荐结果中提取 movieId，便于计算命中数。"""
	return [item_id for item_id, _ in recommendations]


def _evaluate_one_user(
	train_matrix: pd.DataFrame,
	test_row: pd.Series,
	sim_matrix: pd.DataFrame,
	top_k_neighbors: int = 20,
	top_n: int = 10,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> Dict[str, float]:
	"""评估单个用户的评分预测误差和 Top-N 推荐命中效果。"""
	user_id = test_row.name
	true_ratings = test_row.dropna()
	if true_ratings.empty:
		return {}

	preds = predict_ratings_item_based(train_matrix, sim_matrix, user_id, top_k=top_k_neighbors)
	recs = recommend_items_item_based(train_matrix, user_id, n=top_n, top_k=top_k_neighbors)

	preds = preds.reindex(true_ratings.index)
	common = preds.dropna().index.intersection(true_ratings.index)
	if len(common) > 0:
		diff = preds.loc[common].astype(float) - true_ratings.loc[common].astype(float)
		rmse = float(np.sqrt(np.mean(np.square(diff.to_numpy()))))
		mae = float(np.mean(np.abs(diff.to_numpy())))
	else:
		rmse = np.nan
		mae = np.nan

	relevant_items = true_ratings[true_ratings >= relevance_threshold].index
	recommended_items = _top_k_list(recs)
	if len(relevant_items) == 0:
		precision_at_k = 0.0
		recall_at_k = 0.0
	else:
		hits = len(set(recommended_items) & set(relevant_items))
		precision_at_k = hits / max(1, min(top_n, len(recommended_items)))
		recall_at_k = hits / len(relevant_items)

	return {
		"userId": user_id,
		"RMSE": rmse,
		"MAE": mae,
		"Precision@K": precision_at_k,
		"Recall@K": recall_at_k,
	}


def evaluate_algorithm(
	train_matrix: pd.DataFrame,
	test_matrix: pd.DataFrame,
	algorithm: str = "item",
	top_k_neighbors: int = 20,
	top_n: int = 10,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
	user_ids: Optional[Sequence[object]] = None,
) -> pd.DataFrame:
	"""评估基于物品的协同过滤模型，并返回一行汇总指标。

	保留 algorithm 参数是为了兼容旧 notebook；当前只支持 algorithm="item"。
	"""
	if algorithm != "item":
		raise ValueError("当前版本只保留基于物品的协同过滤，algorithm 必须是 'item'")

	train_matrix = train_matrix.apply(pd.to_numeric, errors="coerce")
	test_matrix = test_matrix.apply(pd.to_numeric, errors="coerce")

	if user_ids is None:
		user_ids = [uid for uid in test_matrix.index if test_matrix.loc[uid].notna().any()]

	sim_matrix = compute_item_similarity(train_matrix)
	detail_rows = []
	for user_id in user_ids:
		if user_id not in test_matrix.index:
			continue
		row = _evaluate_one_user(
			train_matrix=train_matrix,
			test_row=test_matrix.loc[user_id],
			sim_matrix=sim_matrix,
			top_k_neighbors=top_k_neighbors,
			top_n=top_n,
			relevance_threshold=relevance_threshold,
		)
		if row:
			detail_rows.append(row)

	if not detail_rows:
		result = pd.DataFrame(
			[
				{
					"algorithm": "item",
					"top_k_neighbors": top_k_neighbors,
					"top_n": top_n,
					"n_users": 0,
					"n_test_ratings": 0,
					"RMSE": np.nan,
					"MAE": np.nan,
					"Precision@K": np.nan,
					"Recall@K": np.nan,
				}
			]
		)
		result.attrs["per_user_results"] = pd.DataFrame(columns=["userId", "RMSE", "MAE", "Precision@K", "Recall@K"])
		return result

	detail_df = pd.DataFrame(detail_rows)
	summary = detail_df[["RMSE", "MAE", "Precision@K", "Recall@K"]].mean(numeric_only=True).to_frame().T
	summary.insert(0, "algorithm", "item")
	summary.insert(1, "top_k_neighbors", top_k_neighbors)
	summary.insert(2, "top_n", top_n)
	summary.insert(3, "n_users", len(detail_df))
	n_test_ratings = int(sum(test_matrix.loc[user_id].notna().sum() for user_id in detail_df["userId"] if user_id in test_matrix.index))
	summary.insert(4, "n_test_ratings", n_test_ratings)
	summary.attrs["per_user_results"] = detail_df
	return summary


def evaluate_item_cf(
	matrix: pd.DataFrame,
	test_size: float = 0.2,
	top_k_neighbors: int = 20,
	top_n: int = 10,
	random_state: int = 42,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
	user_ids: Optional[Sequence[object]] = None,
) -> pd.DataFrame:
	"""完成一次物品 CF 的训练/测试划分和整体评估，返回汇总指标表。"""
	train_matrix, test_matrix = train_test_split_matrix(
		matrix=matrix,
		test_size=test_size,
		random_state=random_state,
	)
	result = evaluate_algorithm(
		train_matrix=train_matrix,
		test_matrix=test_matrix,
		algorithm="item",
		top_k_neighbors=top_k_neighbors,
		top_n=top_n,
		relevance_threshold=relevance_threshold,
		user_ids=user_ids,
	)
	result.attrs["train_matrix"] = train_matrix
	result.attrs["test_matrix"] = test_matrix
	return result


def compare_item_user_cf(
	matrix: pd.DataFrame,
	test_size: float = 0.2,
	top_k_neighbors: int = 20,
	top_n: int = 10,
	random_state: int = 42,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
	user_ids: Optional[Sequence[object]] = None,
) -> pd.DataFrame:
	"""兼容旧函数名：当前只评估并返回物品 CF 结果。"""
	return evaluate_item_cf(
		matrix=matrix,
		test_size=test_size,
		top_k_neighbors=top_k_neighbors,
		top_n=top_n,
		random_state=random_state,
		relevance_threshold=relevance_threshold,
		user_ids=user_ids,
	)


def sensitivity_analysis(
	matrix: pd.DataFrame,
	top_k_values: Sequence[int] = (5, 10, 20, 40),
	top_n: int = 10,
	test_size: float = 0.2,
	random_state: int = 42,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
	user_ids: Optional[Sequence[object]] = None,
) -> pd.DataFrame:
	"""遍历不同 top_k 邻居数，返回物品 CF 的敏感性分析表。"""
	train_matrix, test_matrix = train_test_split_matrix(
		matrix=matrix,
		test_size=test_size,
		random_state=random_state,
	)

	rows = []
	for top_k_neighbors in top_k_values:
		result = evaluate_algorithm(
			train_matrix=train_matrix,
			test_matrix=test_matrix,
			algorithm="item",
			top_k_neighbors=top_k_neighbors,
			top_n=top_n,
			relevance_threshold=relevance_threshold,
			user_ids=user_ids,
		)
		if not result.empty:
			rows.append(result.iloc[0].to_dict())

	return pd.DataFrame(rows)


def load_default_matrix(project_root: Optional[Path] = None) -> pd.DataFrame:
	"""定位项目默认矩阵文件，供命令行 demo 使用。"""
	if project_root is None:
		project_root = Path(__file__).resolve().parents[1]
	return load_filtered_matrix(project_root / "data" / "cleaned_data" / "user_item_matrix_sparse_filtered.csv")


def run_demo() -> Dict[str, pd.DataFrame]:
	"""命令行演示：输出物品 CF 汇总评估和 top_k 敏感性分析。"""
	matrix = load_default_matrix()
	result = evaluate_item_cf(matrix=matrix)
	sensitivity = sensitivity_analysis(matrix=matrix)

	print("=== Item-based CF evaluation ===")
	print(result.to_string(index=False))
	print("\n=== top_k sensitivity ===")
	print(sensitivity.to_string(index=False))

	return {
		"result": result,
		"sensitivity": sensitivity,
	}


def main() -> None:
	"""直接运行 evaluation.py 时执行快速评估 demo。"""
	run_demo()


if __name__ == "__main__":
	main()
