from __future__ import annotations

from pathlib import Path
from typing import Hashable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


RecommendationList = List[Tuple[Hashable, float]]

# 相似度置信度参数：共同评分用户越少，余弦相似度越容易偶然偏高，因此用 shrinkage 收缩。
SIMILARITY_SHRINKAGE_LAMBDA = 25.0
# 共评人数硬约束：低于该阈值的电影对不参与邻居贡献。
MIN_CO_RATINGS = 3
# 排序层权重：只影响 Top-N 推荐排序，不改变评分预测本身。
POPULARITY_RERANK_WEIGHT = 0.08
CONFIDENCE_RERANK_WEIGHT = 0.04


def _validate_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
	"""检查评分矩阵格式，并统一转换为数值型 DataFrame。"""
	if not isinstance(matrix, pd.DataFrame):
		raise TypeError("matrix 必须是 pandas.DataFrame")
	if matrix.empty:
		raise ValueError("matrix 不能为空")
	if all(pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
		return matrix
	return matrix.apply(pd.to_numeric, errors="coerce")


def _global_mean(matrix: pd.DataFrame) -> float:
	"""计算全局平均评分，作为冷启动或缺少邻居时的兜底值。"""
	value = float(matrix.stack().mean())
	return 0.0 if np.isnan(value) else value


def _user_means(matrix: pd.DataFrame) -> pd.Series:
	"""计算每个用户的平均评分，用于用户均值中心化。"""
	return matrix.mean(axis=1, skipna=True)


def _item_means(matrix: pd.DataFrame) -> pd.Series:
	"""计算每部电影的平均评分，用于冷启动和孤立电影兜底。"""
	return matrix.mean(axis=0, skipna=True)


def _rating_bounds(matrix: pd.DataFrame) -> Tuple[float, float]:
	"""从历史评分估计评分上下界，用于裁剪预测评分。"""
	values = matrix.to_numpy(dtype=float)
	values = values[~np.isnan(values)]
	if values.size == 0:
		return 0.0, 5.0
	return float(np.nanmin(values)), float(np.nanmax(values))


def _clip_prediction(value: float, matrix: pd.DataFrame) -> float:
	"""把预测评分限制在数据集中已有评分范围内。"""
	low, high = _rating_bounds(matrix)
	return float(np.clip(value, low, high))


def _center_by_user(matrix: pd.DataFrame) -> pd.DataFrame:
	"""对评分矩阵做用户均值中心化，降低不同用户打分尺度差异。"""
	return matrix.sub(_user_means(matrix), axis=0)


def _fallback_item_ranking(matrix: pd.DataFrame) -> pd.Series:
	"""冷启动用户没有历史评分时，直接按电影平均分生成推荐排序。"""
	global_mean = _global_mean(matrix)
	return _item_means(matrix).fillna(global_mean).sort_values(ascending=False)


def _top_k_similar_indices(similarities: np.ndarray, candidate_indices: np.ndarray, top_k: int) -> np.ndarray:
	"""从候选邻居中保留相似度最高的 top_k 个位置索引。"""
	if candidate_indices.size == 0:
		return candidate_indices
	ordered = candidate_indices[np.argsort(similarities[candidate_indices])[::-1]]
	return ordered[:top_k]


def _normalized_series(values: pd.Series) -> pd.Series:
	"""把辅助排序特征归一化到 0-1，便于和预测评分小权重融合。"""
	values = values.astype(float)
	min_value = float(values.min()) if not values.empty else 0.0
	max_value = float(values.max()) if not values.empty else 0.0
	if np.isclose(max_value, min_value):
		return pd.Series(0.0, index=values.index)
	return (values - min_value) / (max_value - min_value)


def load_user_item_matrix(project_root: Path) -> pd.DataFrame:
	"""加载过滤后的用户-电影评分矩阵，行是 userId，列是 movieId。"""
	path = project_root / "data" / "cleaned_data" / "user_item_matrix_sparse_filtered.csv"
	if not path.exists():
		raise FileNotFoundError(f"user_item_matrix_sparse_filtered.csv not found: {path}")

	matrix = pd.read_csv(path, index_col=0)
	try:
		matrix.index = matrix.index.astype(int)
	except Exception:
		pass
	try:
		matrix.columns = matrix.columns.astype(int)
	except Exception:
		pass
	return _validate_matrix(matrix)


def load_movie_title_map(project_root: Path) -> dict[Hashable, str]:
	"""读取 movieId 到电影名称的映射，供 notebook 展示推荐结果时使用。"""
	candidates = [
		project_root / "data" / "cleaned_data" / "cleaned_movies.csv",
		project_root / "data" / "ml-latest-small" / "movies.csv",
	]
	movie_path = next((path for path in candidates if path.exists()), None)
	if movie_path is None:
		raise FileNotFoundError("没有找到 cleaned_movies.csv 或 movies.csv")

	movies = pd.read_csv(movie_path)
	if not {"movieId", "title"}.issubset(movies.columns):
		raise ValueError(f"{movie_path} 缺少 movieId/title 列")

	movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
	movies = movies.dropna(subset=["movieId", "title"])
	movies["movieId"] = movies["movieId"].astype(int)
	return dict(zip(movies["movieId"], movies["title"].astype(str)))


def attach_movie_titles(
	recommendations: RecommendationList,
	movie_titles: Mapping[Hashable, str],
) -> pd.DataFrame:
	"""把推荐列表转换为 DataFrame，并可选补充电影名称。"""
	result = pd.DataFrame(recommendations, columns=["movieId", "pred_rating"])
	result["title"] = result["movieId"].map(movie_titles)
	return result[["movieId", "title", "pred_rating"]]


def compute_item_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
	"""计算经过共评置信度约束的物品相似度矩阵。

	主要步骤：
	1. 校验并数值化用户-电影评分矩阵；
	2. 按用户均值做中心化，降低不同用户打分尺度差异；
	3. 计算电影列向量之间的余弦相似度；
	4. 统计每对电影的共同评分用户数；
	5. 用 shrinkage 收缩共评人数较少的相似度，并过滤低共评电影对。
	"""
	matrix = _validate_matrix(matrix)
	centered = _center_by_user(matrix).fillna(0.0)
	sim = cosine_similarity(centered.to_numpy(dtype=float).T)
	sim = np.clip(sim, -1.0, 1.0)

	# 共评人数越少，相似度可信度越低；shrinkage 会把这类相似度向 0 拉回。
	observed_mask = matrix.notna().astype(int).to_numpy()
	co_rating_counts = observed_mask.T @ observed_mask
	shrinkage = co_rating_counts / (co_rating_counts + SIMILARITY_SHRINKAGE_LAMBDA)
	sim = sim * shrinkage
	sim[co_rating_counts < MIN_CO_RATINGS] = 0.0
	np.fill_diagonal(sim, 1.0)

	return pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)


def predict_ratings_item_based(
	matrix: pd.DataFrame,
	sim_matrix: pd.DataFrame,
	user_id: Hashable,
	top_k: int = 20,
) -> pd.Series:
	"""基于物品协同过滤预测用户未评分电影的评分。

	预测评分仍保持纯评分语义：只使用中心化评分和收缩后的物品相似度。
	热门度与置信度重排序只在推荐列表生成阶段使用，不混入 RMSE/MAE 评估。
	"""
	matrix = _validate_matrix(matrix)
	item_means = _item_means(matrix)
	global_mean = _global_mean(matrix)

	if user_id not in matrix.index:
		return _fallback_item_ranking(matrix)

	item_ids = list(sim_matrix.columns)
	user_ratings = matrix.loc[user_id].reindex(item_ids)
	user_mean = float(user_ratings.mean(skipna=True))
	if np.isnan(user_mean):
		user_mean = global_mean

	user_values = user_ratings.to_numpy(dtype=float)
	rated_positions = np.flatnonzero(~np.isnan(user_values))
	unrated_positions = np.flatnonzero(np.isnan(user_values))
	if rated_positions.size == 0:
		return item_means.loc[[item_ids[pos] for pos in unrated_positions]].fillna(global_mean).sort_values(ascending=False)

	sim_values = sim_matrix.to_numpy(dtype=float)
	item_means_values = item_means.reindex(item_ids).fillna(global_mean).to_numpy(dtype=float)
	neighbor_sims = sim_values[np.ix_(unrated_positions, rated_positions)]
	rating_deviations = user_values[rated_positions] - user_mean
	k = len(rated_positions) if top_k <= 0 else min(top_k, len(rated_positions))

	# 对所有候选电影一次性取 top_k 邻居，避免全量评估时逐电影 Python 循环。
	if k < len(rated_positions):
		selected = np.argpartition(-neighbor_sims, kth=k - 1, axis=1)[:, :k]
		row_index = np.arange(neighbor_sims.shape[0])[:, None]
		selected_sims = neighbor_sims[row_index, selected]
		selected_deviations = rating_deviations[selected]
	else:
		selected_sims = neighbor_sims
		selected_deviations = np.broadcast_to(rating_deviations, neighbor_sims.shape)

	denom = np.sum(np.abs(selected_sims), axis=1)
	numer = np.sum(selected_sims * selected_deviations, axis=1)
	fallback_values = item_means_values[unrated_positions]
	scores = fallback_values.copy()
	valid_score_mask = denom > 0
	scores[valid_score_mask] = user_mean + numer[valid_score_mask] / denom[valid_score_mask]
	low, high = _rating_bounds(matrix)
	scores = np.clip(scores, low, high)

	preds = pd.Series(scores, index=[item_ids[pos] for pos in unrated_positions])
	return preds.sort_values(ascending=False)


def rerank_predictions_item_based(
	matrix: pd.DataFrame,
	sim_matrix: pd.DataFrame,
	user_id: Hashable,
	predictions: pd.Series,
	top_k: int = 20,
) -> pd.Series:
	"""在预测评分基础上做轻量重排序，用于提升 Top-N 推荐命中。

	排序层加入两个辅助信号：
	- popularity_score：电影评分人数，代表统计支撑和大众接受度；
	- confidence_score：候选电影与用户历史电影的相似度证据强度。
	"""
	if predictions.empty:
		return predictions

	matrix = _validate_matrix(matrix)
	item_ids = list(sim_matrix.columns)
	sim_values = sim_matrix.to_numpy(dtype=float)
	item_to_pos = {item_id: pos for pos, item_id in enumerate(item_ids)}

	item_rating_counts = matrix.notna().sum(axis=0).reindex(predictions.index).fillna(0.0)
	popularity_score = _normalized_series(np.log1p(item_rating_counts))

	if user_id in matrix.index:
		user_ratings = matrix.loc[user_id].reindex(item_ids)
		rated_positions = np.flatnonzero(~np.isnan(user_ratings.to_numpy(dtype=float)))
	else:
		rated_positions = np.array([], dtype=int)

	candidate_positions = np.array([item_to_pos.get(item_id, -1) for item_id in predictions.index], dtype=int)
	confidence_values = np.zeros(len(predictions), dtype=float)
	valid_mask = (candidate_positions >= 0) & (rated_positions.size > 0)
	if valid_mask.any():
		candidate_sims = np.abs(sim_values[np.ix_(candidate_positions[valid_mask], rated_positions)])
		k = len(rated_positions) if top_k <= 0 else min(top_k, len(rated_positions))
		# 置信度也批量取 top_k 相似度绝对值求和，表示候选电影邻居证据强度。
		if k < len(rated_positions):
			selected = np.argpartition(-candidate_sims, kth=k - 1, axis=1)[:, :k]
			selected_sims = np.take_along_axis(candidate_sims, selected, axis=1)
			confidence_values[valid_mask] = selected_sims.sum(axis=1)
		else:
			confidence_values[valid_mask] = candidate_sims.sum(axis=1)

	confidence_score = _normalized_series(pd.Series(confidence_values, index=predictions.index))
	ranking_scores = (
		predictions
		+ POPULARITY_RERANK_WEIGHT * popularity_score
		+ CONFIDENCE_RERANK_WEIGHT * confidence_score
	)
	return ranking_scores.sort_values(ascending=False)


def recommend_items_item_based(
	matrix: pd.DataFrame,
	user_id: Hashable,
	n: int = 10,
	top_k: int = 20,
) -> RecommendationList:
	"""计算物品相似度，并返回重排序后的 Top-N 推荐电影。"""
	sim_matrix = compute_item_similarity(matrix)
	preds = predict_ratings_item_based(matrix, sim_matrix, user_id, top_k=top_k)
	ranked = rerank_predictions_item_based(matrix, sim_matrix, user_id, preds, top_k=top_k)
	return list(ranked.dropna().head(n).items())


def main_demo(project_root: Path = Path(__file__).resolve().parent.parent) -> None:
	"""命令行演示：加载矩阵并输出一个样例用户的物品 CF 推荐结果。"""
	matrix = load_user_item_matrix(project_root)
	sample_user = matrix.index[0]
	recs = recommend_items_item_based(matrix, sample_user, n=10, top_k=20)

	print(f"Matrix shape: {matrix.shape}")
	print(f"Sample user: {sample_user}")
	print("Item-based CF recommendations:")
	print(recs)


if __name__ == "__main__":
	main_demo()
