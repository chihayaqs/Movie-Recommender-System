from __future__ import annotations

from pathlib import Path
from typing import Hashable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


RecommendationList = List[Tuple[Hashable, float]]


def _validate_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
	"""检查评分矩阵格式，并统一转换为数值型 DataFrame。"""
	if not isinstance(matrix, pd.DataFrame):
		raise TypeError("matrix 必须是 pandas.DataFrame")
	if matrix.empty:
		raise ValueError("matrix 不能为空")
	return matrix.apply(pd.to_numeric, errors="coerce")


def _global_mean(matrix: pd.DataFrame) -> float:
	"""计算全局平均评分，作为冷启动或缺少邻居时的兜底值。"""
	value = float(matrix.stack().mean())
	return 0.0 if np.isnan(value) else value


def _user_means(matrix: pd.DataFrame) -> pd.Series:
	"""计算每个用户的平均评分，用于按用户中心化，削弱用户打分尺度差异。"""
	return matrix.mean(axis=1, skipna=True)


def _item_means(matrix: pd.DataFrame) -> pd.Series:
	"""计算每部电影的平均评分，用于冷启动用户和孤立电影的回退推荐。"""
	return matrix.mean(axis=0, skipna=True)


def _rating_bounds(matrix: pd.DataFrame) -> Tuple[float, float]:
	"""从历史评分中估计评分上下界，用于裁剪预测评分。"""
	values = matrix.to_numpy(dtype=float)
	values = values[~np.isnan(values)]
	if values.size == 0:
		return 0.0, 5.0
	return float(np.nanmin(values)), float(np.nanmax(values))


def _clip_prediction(value: float, matrix: pd.DataFrame) -> float:
	"""把预测评分限制在数据集已有评分范围内。"""
	low, high = _rating_bounds(matrix)
	return float(np.clip(value, low, high))


def _center_by_user(matrix: pd.DataFrame) -> pd.DataFrame:
	"""对评分矩阵做用户均值中心化，再用于计算电影之间的相似度。"""
	return matrix.sub(_user_means(matrix), axis=0)


def _top_k_neighbors(similarities: pd.Series, top_k: int) -> pd.Series:
	"""从候选相似电影中取相似度绝对值最高的 top_k 个邻居。"""
	similarities = similarities.dropna()
	similarities = similarities[similarities != 0]
	if top_k <= 0 or similarities.empty:
		return similarities.iloc[0:0]
	ordered = similarities.abs().sort_values(ascending=False).head(top_k).index
	return similarities.loc[ordered]


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
	"""使用 scikit-learn 计算物品相似度。

	主要步骤：
	1. 校验并数值化用户-电影评分矩阵；
	2. 按用户平均分做中心化，降低不同用户评分习惯的影响；
	3. 用 sklearn.metrics.pairwise.cosine_similarity 计算电影列向量之间的余弦相似度；
	4. 返回以 movieId 为行列索引的相似度矩阵。
	"""
	matrix = _validate_matrix(matrix)
	centered = _center_by_user(matrix).fillna(0.0)
	sim = cosine_similarity(centered.to_numpy(dtype=float).T)
	sim = np.clip(sim, -1.0, 1.0)
	np.fill_diagonal(sim, 1.0)
	return pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)


def predict_ratings_item_based(
	matrix: pd.DataFrame,
	sim_matrix: pd.DataFrame,
	user_id: Hashable,
	top_k: int = 20,
) -> pd.Series:
	"""基于物品协同过滤预测用户未评分电影的评分。

	主要步骤：
	1. 找到目标用户已评分电影和未评分电影；
	2. 对每部未评分电影，取它与用户已评分电影中最相似的 top_k 个邻居；
	3. 用邻居相似度对用户历史评分偏差做加权平均；
	4. 若用户或电影缺少足够信息，则回退到电影平均分或全局平均分。
	"""
	matrix = _validate_matrix(matrix)
	item_means = _item_means(matrix)
	global_mean = _global_mean(matrix)

	if user_id not in matrix.index:
		return _fallback_item_ranking(matrix)

	user_ratings = matrix.loc[user_id]
	user_mean = float(user_ratings.mean(skipna=True))
	if np.isnan(user_mean):
		user_mean = global_mean

	item_ids = list(matrix.columns)
	user_values = user_ratings.to_numpy(dtype=float)
	rated_positions = np.flatnonzero(~np.isnan(user_values))
	unrated_positions = np.flatnonzero(np.isnan(user_values))
	if rated_positions.size == 0:
		return item_means.loc[[item_ids[pos] for pos in unrated_positions]].fillna(global_mean).sort_values(ascending=False)

	sim_values = sim_matrix.to_numpy(dtype=float)
	item_means_values = item_means.reindex(item_ids).fillna(global_mean).to_numpy(dtype=float)

	preds: dict[Hashable, float] = {}
	for pos in unrated_positions:
		item_id = item_ids[pos]
		sims = sim_values[pos, rated_positions]
		selected_rel = _top_k_similar_indices(sims, np.arange(rated_positions.size), top_k)
		if selected_rel.size == 0:
			preds[item_id] = float(item_means_values[pos])
			continue

		selected_positions = rated_positions[selected_rel]
		selected_sims = sims[selected_rel]
		denom = np.sum(np.abs(selected_sims))
		if denom <= 0:
			preds[item_id] = float(item_means_values[pos])
			continue

		numer = np.dot(selected_sims, user_values[selected_positions])
		score = user_mean + float(numer / denom)
		preds[item_id] = _clip_prediction(score, matrix)

	return pd.Series(preds).sort_values(ascending=False)


def recommend_items_item_based(
	matrix: pd.DataFrame,
	user_id: Hashable,
	n: int = 10,
	top_k: int = 20,
) -> RecommendationList:
	"""计算物品相似度并返回目标用户预测评分最高的 Top-N 电影。"""
	sim_matrix = compute_item_similarity(matrix)
	preds = predict_ratings_item_based(matrix, sim_matrix, user_id, top_k=top_k)
	return list(preds.dropna().head(n).items())


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
