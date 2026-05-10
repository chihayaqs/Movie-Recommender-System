from pathlib import Path
from typing import Tuple, List

import pandas as pd  # type: ignore[import-not-found]


def load_user_item_matrix(project_root: Path) -> pd.DataFrame:
	"""读取已经处理好的用户-电影评分矩阵。"""
	path = project_root / "data" / "cleaned_data" / "user_item_matrix_sparse_filtered.csv"
	if not path.exists():
		raise FileNotFoundError(f"user_item_matrix_sparse_filtered.csv not found: {path}")
	matrix = pd.read_csv(path, index_col=0)
	matrix.index = matrix.index.astype(int)
	matrix.columns = matrix.columns.astype(int)
	return matrix


def _cosine_similarity_by_columns(matrix_values: pd.DataFrame) -> pd.DataFrame:
	"""Compute cosine similarity between columns of a 2D matrix.

	Expects shape (n_users, n_items). Missing values should be filled with 0.
	Returns (n_items, n_items) similarity matrix as DataFrame.
	"""
	X = matrix_values
	norms = (X.pow(2).sum(axis=0)) ** 0.5
	norms = norms.mask(norms == 0, 1.0)
	Xn = X.div(norms, axis=1)
	sim = Xn.T.dot(Xn).clip(lower=-1.0, upper=1.0)
	return sim


def compute_item_similarity(matrix: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
	"""Compute item-item cosine similarity.

	Returns (sim_matrix, item_ids) where item_ids matches column order in sim_matrix.
	"""
	item_ids = list(matrix.columns)
	values = matrix.fillna(0).astype(float)  # shape (n_users, n_items)
	sim = _cosine_similarity_by_columns(values)
	return sim, item_ids


def _top_k_similar_indices(similarities: pd.Series, candidate_items: List[int], top_k: int) -> List[int]:
	"""取出最相似的 K 个候选索引。"""
	if not candidate_items:
		return candidate_items
	return list(similarities.loc[candidate_items].sort_values(ascending=False).head(top_k).index)


def predict_ratings_item_based(
	matrix: pd.DataFrame, sim: pd.DataFrame, item_ids: List[int], user_id: int, top_k: int = 20
) -> pd.Series:
	"""Predict ratings for a given `user_id` using item-based CF.

	Returns a Series indexed by itemId with predicted ratings for items the user hasn't rated.
	"""
	if user_id not in matrix.index:
		raise KeyError(f"user {user_id} not found in matrix")

	user_ratings = matrix.loc[user_id]
	rated_mask = user_ratings.notna()
	rated_items = [c for c in matrix.columns if rated_mask.get(c, False)]
	if not rated_items:
		# cold-start: return global mean for all items
		global_mean = matrix.stack().mean()
		return pd.Series(global_mean, index=matrix.columns)

	user_vector = user_ratings.fillna(0)

	preds = {}
	for item in item_ids:
		if rated_mask.get(item, False):
			continue
		# 仅取与目标电影最相似的 top_k 个历史评分电影，降低噪声
		sims = sim.loc[item]
		selected_indices = _top_k_similar_indices(sims, rated_items, top_k)
		if not selected_indices:
			preds[item] = float("nan")
			continue
		numer = float((sims.loc[selected_indices] * user_vector.loc[selected_indices]).sum())
		denom = float(sims.loc[selected_indices].abs().sum())
		if denom > 0:
			preds[item] = numer / denom
		else:
			preds[item] = float("nan")

	preds_series = pd.Series(preds).sort_values(ascending=False)
	return preds_series


def recommend_items_item_based(matrix: pd.DataFrame, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
	sim, item_ids = compute_item_similarity(matrix)
	preds = predict_ratings_item_based(matrix, sim, item_ids, user_id)
	preds = preds.dropna()
	top = preds.head(n)
	return list(top.items())


def main_demo(project_root: Path = Path(__file__).resolve().parent.parent):
	matrix = load_user_item_matrix(project_root)
	sample_user = matrix.index[0]
	print(f"Matrix shape: {matrix.shape}")
	print(f"Computing item-based recommendations for user {sample_user}...")
	recs = recommend_items_item_based(matrix, sample_user, n=10)
	print(recs)


if __name__ == "__main__":
	main_demo()

