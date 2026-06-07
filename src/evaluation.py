from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
	from . import cf_recommender as sklearn_cf
	from . import model as baseline_cf
except ImportError:
	import cf_recommender as sklearn_cf
	import model as baseline_cf


DEFAULT_RELEVANCE_THRESHOLD = 4.0
BASELINE_ALGORITHM = "baseline_numpy_item_cf"
SKLEARN_ALGORITHM = "sklearn_centered_item_cf"
LEGACY_ALGORITHM_ALIASES = {
	"item": SKLEARN_ALGORITHM,
	"sklearn_item_cf": SKLEARN_ALGORITHM,
	"baseline": BASELINE_ALGORITHM,
}
DEFAULT_ALGORITHMS = (BASELINE_ALGORITHM, SKLEARN_ALGORITHM)
SUMMARY_COLUMNS = [
	"algorithm",
	"top_k_neighbors",
	"top_n",
	"n_users",
	"n_test_ratings",
	"n_users_with_relevant_items",
	"RMSE",
	"MAE",
	"Weighted_RMSE",
	"Weighted_MAE",
	"Precision@K",
	"Recall@K",
	"HitRate@K",
	"Precision@K_relevant_users",
	"Recall@K_relevant_users",
	"HitRate@K_relevant_users",
]
PER_USER_COLUMNS = [
	"userId",
	"n_test_ratings",
	"n_predicted_test_ratings",
	"n_relevant_items",
	"n_hits",
	"RMSE",
	"MAE",
	"SSE",
	"SAE",
	"Precision@K",
	"Recall@K",
	"HitRate@K",
]


@dataclass(frozen=True)
class AlgorithmSpec:
	name: str
	compute_similarity: Callable[[pd.DataFrame], Any]
	predict_ratings: Callable[[pd.DataFrame, Any, object, int], pd.Series]
	rank_ratings: Optional[Callable[[pd.DataFrame, Any, object, int], pd.Series]] = None


def _normalize_algorithm_name(algorithm: str) -> str:
	"""Return the canonical algorithm name used in reports."""
	return LEGACY_ALGORITHM_ALIASES.get(algorithm, algorithm)


def _baseline_compute_similarity(matrix: pd.DataFrame) -> Tuple[np.ndarray, list]:
	return baseline_cf.compute_item_similarity(matrix)


def _baseline_predict_ratings(
	matrix: pd.DataFrame,
	similarity_bundle: Tuple[np.ndarray, list],
	user_id: object,
	top_k: int,
) -> pd.Series:
	sim_matrix, item_ids = similarity_bundle
	if user_id not in matrix.index:
		return _fallback_item_ranking(matrix)
	user_ratings = matrix.loc[user_id].reindex(item_ids)
	return _predict_from_similarity(
		user_ratings=user_ratings,
		sim_values=sim_matrix,
		item_ids=item_ids,
		top_k=top_k,
		center_by_user=False,
		clip_bounds=None,
		fallback_scores=_fallback_item_ranking(matrix).reindex(item_ids),
	)


def _sklearn_compute_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
	return sklearn_cf.compute_item_similarity(matrix)


def _sklearn_predict_ratings(
	matrix: pd.DataFrame,
	similarity_bundle: pd.DataFrame,
	user_id: object,
	top_k: int,
) -> pd.Series:
	return sklearn_cf.predict_ratings_item_based(
		matrix=matrix,
		sim_matrix=similarity_bundle,
		user_id=user_id,
		top_k=top_k,
	)


def _sklearn_rank_ratings(
	matrix: pd.DataFrame,
	similarity_bundle: pd.DataFrame,
	user_id: object,
	top_k: int,
) -> pd.Series:
	"""Return model ranking scores used only for Top-N evaluation."""
	predicted_ratings = _sklearn_predict_ratings(matrix, similarity_bundle, user_id, top_k)
	return sklearn_cf.rerank_predictions_item_based(
		matrix=matrix,
		sim_matrix=similarity_bundle,
		user_id=user_id,
		predictions=predicted_ratings,
		top_k=top_k,
	)


ALGORITHM_SPECS = {
	BASELINE_ALGORITHM: AlgorithmSpec(
		name=BASELINE_ALGORITHM,
		compute_similarity=_baseline_compute_similarity,
		predict_ratings=_baseline_predict_ratings,
	),
	SKLEARN_ALGORITHM: AlgorithmSpec(
		name=SKLEARN_ALGORITHM,
		compute_similarity=_sklearn_compute_similarity,
		predict_ratings=_sklearn_predict_ratings,
		rank_ratings=_sklearn_rank_ratings,
	),
}


def load_filtered_matrix(file_path: Optional[Path] = None) -> pd.DataFrame:
	"""Load the filtered user-item rating matrix used by experiments."""
	if file_path is None:
		file_path = Path(__file__).resolve().parents[1] / "data" / "cleaned_data" / "user_item_matrix_sparse_filtered.csv"
	if not file_path.exists():
		raise FileNotFoundError(f"matrix file not found: {file_path}")

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
	"""Split each user's observed ratings into train and test matrices."""
	if not 0 < test_size < 1:
		raise ValueError("test_size must be between 0 and 1")

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


def _global_mean(matrix: pd.DataFrame) -> float:
	value = float(matrix.stack().mean())
	return 0.0 if np.isnan(value) else value


def _fallback_item_ranking(matrix: pd.DataFrame) -> pd.Series:
	global_mean = _global_mean(matrix)
	return matrix.mean(axis=0, skipna=True).fillna(global_mean).sort_values(ascending=False)


def _predict_from_similarity(
	user_ratings: pd.Series,
	sim_values: np.ndarray,
	item_ids: Sequence[object],
	top_k: int,
	center_by_user: bool,
	clip_bounds: Optional[Tuple[float, float]],
	fallback_scores: pd.Series,
) -> pd.Series:
	"""Vectorized item-CF prediction for one user."""
	user_values = user_ratings.to_numpy(dtype=float)
	rated_positions = np.flatnonzero(~np.isnan(user_values))
	unrated_positions = np.flatnonzero(np.isnan(user_values))
	if unrated_positions.size == 0:
		return pd.Series(dtype=float)
	if rated_positions.size == 0:
		return fallback_scores.iloc[unrated_positions].sort_values(ascending=False)

	k = rated_positions.size if top_k <= 0 else min(top_k, rated_positions.size)
	candidate_sims = sim_values[np.ix_(unrated_positions, rated_positions)]
	if k < rated_positions.size:
		selected_rel = np.argpartition(-candidate_sims, kth=k - 1, axis=1)[:, :k]
		selected_sims = np.take_along_axis(candidate_sims, selected_rel, axis=1)
		selected_ratings = user_values[rated_positions][selected_rel]
	else:
		selected_sims = candidate_sims
		selected_ratings = np.broadcast_to(user_values[rated_positions], selected_sims.shape)

	denom = np.sum(np.abs(selected_sims), axis=1)
	if center_by_user:
		global_fallback = float(fallback_scores.mean()) if not fallback_scores.empty else 0.0
		user_mean = float(np.nanmean(user_values))
		if np.isnan(user_mean):
			user_mean = global_fallback
		selected_values = selected_ratings - user_mean
		raw_scores = user_mean + np.divide(
			np.sum(selected_sims * selected_values, axis=1),
			denom,
			out=np.zeros_like(denom, dtype=float),
			where=denom > 0,
		)
	else:
		raw_scores = np.divide(
			np.sum(selected_sims * selected_ratings, axis=1),
			denom,
			out=np.full_like(denom, np.nan, dtype=float),
			where=denom > 0,
		)

	fallback_values = fallback_scores.iloc[unrated_positions].to_numpy(dtype=float)
	scores = np.where(denom > 0, raw_scores, fallback_values)
	if clip_bounds is not None:
		scores = np.clip(scores, clip_bounds[0], clip_bounds[1])

	preds = pd.Series(scores, index=[item_ids[pos] for pos in unrated_positions])
	return preds.sort_values(ascending=False)


def _top_n_from_predictions(preds: pd.Series, top_n: int) -> list:
	return list(preds.dropna().sort_values(ascending=False).head(top_n).index)


def _evaluate_one_user(
	train_matrix: pd.DataFrame,
	test_row: pd.Series,
	similarity_bundle: Any,
	spec: AlgorithmSpec,
	top_k_neighbors: int = 20,
	top_n: int = 10,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> Dict[str, float]:
	"""Evaluate rating error and Top-N hit quality for one user."""
	user_id = test_row.name
	true_ratings = test_row.dropna()
	if true_ratings.empty:
		return {}

	preds = spec.predict_ratings(train_matrix, similarity_bundle, user_id, top_k_neighbors)
	# 评分预测和推荐排序可以分离：增强模型的重排序只服务 Top-N 指标，
	# RMSE/MAE 仍使用原始评分预测，避免把排序加成误当成评分。
	ranking_scores = (
		spec.rank_ratings(train_matrix, similarity_bundle, user_id, top_k_neighbors)
		if spec.rank_ratings is not None
		else preds
	)
	ranked_recommendations = _top_n_from_predictions(ranking_scores, top_n)

	test_preds = preds.reindex(true_ratings.index)
	common = test_preds.dropna().index.intersection(true_ratings.index)
	if len(common) > 0:
		diff = test_preds.loc[common].astype(float) - true_ratings.loc[common].astype(float)
		diff_values = diff.to_numpy()
		squared_errors = np.square(diff_values)
		absolute_errors = np.abs(diff_values)
		rmse = float(np.sqrt(np.mean(squared_errors)))
		mae = float(np.mean(absolute_errors))
		sse = float(np.sum(squared_errors))
		sae = float(np.sum(absolute_errors))
	else:
		rmse = np.nan
		mae = np.nan
		sse = np.nan
		sae = np.nan

	relevant_items = true_ratings[true_ratings >= relevance_threshold].index
	hits = 0
	if len(relevant_items) == 0:
		precision_at_k = 0.0
		recall_at_k = 0.0
		hit_rate_at_k = 0.0
	else:
		hits = len(set(ranked_recommendations) & set(relevant_items))
		precision_at_k = hits / max(1, min(top_n, len(ranked_recommendations)))
		recall_at_k = hits / len(relevant_items)
		hit_rate_at_k = 1.0 if hits > 0 else 0.0

	return {
		"userId": user_id,
		"n_test_ratings": int(len(true_ratings)),
		"n_predicted_test_ratings": int(len(common)),
		"n_relevant_items": int(len(relevant_items)),
		"n_hits": int(hits),
		"RMSE": rmse,
		"MAE": mae,
		"SSE": sse,
		"SAE": sae,
		"Precision@K": precision_at_k,
		"Recall@K": recall_at_k,
		"HitRate@K": hit_rate_at_k,
	}


def _empty_summary(algorithm: str, top_k_neighbors: int, top_n: int) -> pd.DataFrame:
	row = {column: np.nan for column in SUMMARY_COLUMNS}
	row.update(
		{
			"algorithm": algorithm,
			"top_k_neighbors": top_k_neighbors,
			"top_n": top_n,
			"n_users": 0,
			"n_test_ratings": 0,
			"n_users_with_relevant_items": 0,
		}
	)
	result = pd.DataFrame([row]).reindex(columns=SUMMARY_COLUMNS)
	result.attrs["per_user_results"] = pd.DataFrame(columns=PER_USER_COLUMNS)
	return result


def _summarize_detail_results(
	detail_df: pd.DataFrame,
	algorithm: str,
	top_k_neighbors: int,
	top_n: int,
) -> Dict[str, float]:
	"""Aggregate per-user metrics into one report row."""
	relevant_detail = detail_df[detail_df["n_relevant_items"] > 0]
	n_predicted = int(detail_df["n_predicted_test_ratings"].sum())
	total_sse = float(detail_df["SSE"].sum(skipna=True))
	total_sae = float(detail_df["SAE"].sum(skipna=True))

	summary = {
		"algorithm": algorithm,
		"top_k_neighbors": top_k_neighbors,
		"top_n": top_n,
		"n_users": int(len(detail_df)),
		"n_test_ratings": int(detail_df["n_test_ratings"].sum()),
		"n_users_with_relevant_items": int(len(relevant_detail)),
		"RMSE": float(detail_df["RMSE"].mean(skipna=True)),
		"MAE": float(detail_df["MAE"].mean(skipna=True)),
		"Weighted_RMSE": float(np.sqrt(total_sse / n_predicted)) if n_predicted > 0 else np.nan,
		"Weighted_MAE": float(total_sae / n_predicted) if n_predicted > 0 else np.nan,
		"Precision@K": float(detail_df["Precision@K"].mean(skipna=True)),
		"Recall@K": float(detail_df["Recall@K"].mean(skipna=True)),
		"HitRate@K": float(detail_df["HitRate@K"].mean(skipna=True)),
		"Precision@K_relevant_users": float(relevant_detail["Precision@K"].mean(skipna=True)) if not relevant_detail.empty else np.nan,
		"Recall@K_relevant_users": float(relevant_detail["Recall@K"].mean(skipna=True)) if not relevant_detail.empty else np.nan,
		"HitRate@K_relevant_users": float(relevant_detail["HitRate@K"].mean(skipna=True)) if not relevant_detail.empty else np.nan,
	}
	return summary


def evaluate_algorithm(
	train_matrix: pd.DataFrame,
	test_matrix: pd.DataFrame,
	algorithm: str = SKLEARN_ALGORITHM,
	top_k_neighbors: int = 20,
	top_n: int = 10,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
	user_ids: Optional[Sequence[object]] = None,
) -> pd.DataFrame:
	"""Evaluate one item-CF algorithm and return one summary row."""
	algorithm = _normalize_algorithm_name(algorithm)
	if algorithm not in ALGORITHM_SPECS:
		supported = ", ".join(ALGORITHM_SPECS)
		raise ValueError(f"unsupported algorithm '{algorithm}', choose one of: {supported}")

	train_matrix = train_matrix.apply(pd.to_numeric, errors="coerce")
	test_matrix = test_matrix.apply(pd.to_numeric, errors="coerce")

	if user_ids is None:
		user_ids = [uid for uid in test_matrix.index if test_matrix.loc[uid].notna().any()]

	spec = ALGORITHM_SPECS[algorithm]
	similarity_bundle = spec.compute_similarity(train_matrix)

	detail_rows = []
	for user_id in user_ids:
		if user_id not in test_matrix.index:
			continue
		row = _evaluate_one_user(
			train_matrix=train_matrix,
			test_row=test_matrix.loc[user_id],
			similarity_bundle=similarity_bundle,
			spec=spec,
			top_k_neighbors=top_k_neighbors,
			top_n=top_n,
			relevance_threshold=relevance_threshold,
		)
		if row:
			detail_rows.append(row)

	if not detail_rows:
		return _empty_summary(algorithm, top_k_neighbors, top_n)

	detail_df = pd.DataFrame(detail_rows)
	summary = pd.DataFrame([_summarize_detail_results(detail_df, algorithm, top_k_neighbors, top_n)]).reindex(columns=SUMMARY_COLUMNS)
	summary.attrs["per_user_results"] = detail_df
	return summary


def compare_models(
	matrix: pd.DataFrame,
	algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
	test_size: float = 0.2,
	top_k_neighbors: int = 20,
	top_n: int = 10,
	random_state: int = 42,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
	user_ids: Optional[Sequence[object]] = None,
) -> pd.DataFrame:
	"""Run multiple models on the same split and return comparable metrics."""
	train_matrix, test_matrix = train_test_split_matrix(
		matrix=matrix,
		test_size=test_size,
		random_state=random_state,
	)
	if user_ids is None:
		selected_user_ids = [uid for uid in test_matrix.index if test_matrix.loc[uid].notna().any()]
	else:
		selected_user_ids = list(user_ids)

	rows = []
	per_user_results = {}
	for algorithm in algorithms:
		result = evaluate_algorithm(
			train_matrix=train_matrix,
			test_matrix=test_matrix,
			algorithm=algorithm,
			top_k_neighbors=top_k_neighbors,
			top_n=top_n,
			relevance_threshold=relevance_threshold,
			user_ids=selected_user_ids,
		)
		if not result.empty:
			rows.append(result.iloc[0].to_dict())
			per_user_results[result.iloc[0]["algorithm"]] = result.attrs.get("per_user_results")

	comparison = pd.DataFrame(rows)
	comparison.attrs["train_matrix"] = train_matrix
	comparison.attrs["test_matrix"] = test_matrix
	comparison.attrs["per_user_results"] = per_user_results
	return comparison


def recommend_for_user(
	train_matrix: pd.DataFrame,
	user_id: object,
	algorithm: str = SKLEARN_ALGORITHM,
	top_k_neighbors: int = 20,
	top_n: int = 10,
) -> list[tuple[object, float]]:
	"""Return Top-N recommendations for one user with a selected model."""
	algorithm = _normalize_algorithm_name(algorithm)
	if algorithm not in ALGORITHM_SPECS:
		supported = ", ".join(ALGORITHM_SPECS)
		raise ValueError(f"unsupported algorithm '{algorithm}', choose one of: {supported}")

	train_matrix = train_matrix.apply(pd.to_numeric, errors="coerce")
	spec = ALGORITHM_SPECS[algorithm]
	similarity_bundle = spec.compute_similarity(train_matrix)
	# 推荐展示使用排序分数；普通模型排序分数等同预测评分，增强模型会叠加重排序特征。
	preds = (
		spec.rank_ratings(train_matrix, similarity_bundle, user_id, top_k_neighbors)
		if spec.rank_ratings is not None
		else spec.predict_ratings(train_matrix, similarity_bundle, user_id, top_k_neighbors)
	)
	top = preds.dropna().sort_values(ascending=False).head(top_n)
	return list(top.items())


def evaluate_item_cf(
	matrix: pd.DataFrame,
	test_size: float = 0.2,
	top_k_neighbors: int = 20,
	top_n: int = 10,
	random_state: int = 42,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
	user_ids: Optional[Sequence[object]] = None,
	algorithm: str = SKLEARN_ALGORITHM,
) -> pd.DataFrame:
	"""Evaluate one item-CF model with a fresh train/test split."""
	return compare_models(
		matrix=matrix,
		algorithms=[algorithm],
		test_size=test_size,
		top_k_neighbors=top_k_neighbors,
		top_n=top_n,
		random_state=random_state,
		relevance_threshold=relevance_threshold,
		user_ids=user_ids,
	)


def compare_item_user_cf(
	matrix: pd.DataFrame,
	test_size: float = 0.2,
	top_k_neighbors: int = 20,
	top_n: int = 10,
	random_state: int = 42,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
	user_ids: Optional[Sequence[object]] = None,
) -> pd.DataFrame:
	"""Backward-compatible wrapper for the sklearn-centered item-CF model."""
	return evaluate_item_cf(
		matrix=matrix,
		test_size=test_size,
		top_k_neighbors=top_k_neighbors,
		top_n=top_n,
		random_state=random_state,
		relevance_threshold=relevance_threshold,
		user_ids=user_ids,
		algorithm=SKLEARN_ALGORITHM,
	)


def sensitivity_analysis(
	matrix: pd.DataFrame,
	top_k_values: Sequence[int] = (5, 10, 20, 40),
	top_n: int = 10,
	test_size: float = 0.2,
	random_state: int = 42,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
	user_ids: Optional[Sequence[object]] = None,
	algorithms: Sequence[str] = (SKLEARN_ALGORITHM,),
) -> pd.DataFrame:
	"""Evaluate one or more models under different neighborhood sizes."""
	train_matrix, test_matrix = train_test_split_matrix(
		matrix=matrix,
		test_size=test_size,
		random_state=random_state,
	)
	if user_ids is None:
		selected_user_ids = [uid for uid in test_matrix.index if test_matrix.loc[uid].notna().any()]
	else:
		selected_user_ids = list(user_ids)

	rows = []
	for algorithm in algorithms:
		algorithm = _normalize_algorithm_name(algorithm)
		if algorithm not in ALGORITHM_SPECS:
			supported = ", ".join(ALGORITHM_SPECS)
			raise ValueError(f"unsupported algorithm '{algorithm}', choose one of: {supported}")
		spec = ALGORITHM_SPECS[algorithm]
		similarity_bundle = spec.compute_similarity(train_matrix)

		for top_k_neighbors in top_k_values:
			detail_rows = []
			for user_id in selected_user_ids:
				if user_id not in test_matrix.index:
					continue
				row = _evaluate_one_user(
					train_matrix=train_matrix,
					test_row=test_matrix.loc[user_id],
					similarity_bundle=similarity_bundle,
					spec=spec,
					top_k_neighbors=top_k_neighbors,
					top_n=top_n,
					relevance_threshold=relevance_threshold,
				)
				if row:
					detail_rows.append(row)
			if not detail_rows:
				rows.append(_empty_summary(algorithm, top_k_neighbors, top_n).iloc[0].to_dict())
				continue

			detail_df = pd.DataFrame(detail_rows)
			summary = _summarize_detail_results(detail_df, algorithm, top_k_neighbors, top_n)
			rows.append(summary)

	return pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)


def repeated_compare_models(
	matrix: pd.DataFrame,
	algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
	random_states: Sequence[int] = (0, 1, 2, 3, 4),
	test_size: float = 0.2,
	top_k_neighbors: int = 20,
	top_n: int = 10,
	relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> pd.DataFrame:
	"""Run model comparison across multiple random splits."""
	rows = []
	for random_state in random_states:
		result = compare_models(
			matrix=matrix,
			algorithms=algorithms,
			test_size=test_size,
			top_k_neighbors=top_k_neighbors,
			top_n=top_n,
			random_state=random_state,
			relevance_threshold=relevance_threshold,
			user_ids=None,
		)
		if result.empty:
			continue
		result = result.copy()
		result.insert(0, "random_state", random_state)
		rows.extend(result.to_dict("records"))
	columns = ["random_state"] + SUMMARY_COLUMNS
	return pd.DataFrame(rows).reindex(columns=columns)


def summarize_repeated_results(
	repeated_results: pd.DataFrame,
	metric_columns: Sequence[str] = (
		"RMSE",
		"MAE",
		"Weighted_RMSE",
		"Weighted_MAE",
		"Precision@K",
		"Recall@K",
		"HitRate@K",
		"Precision@K_relevant_users",
		"Recall@K_relevant_users",
		"HitRate@K_relevant_users",
	),
) -> pd.DataFrame:
	"""Summarize repeated evaluation as mean/std columns by model setting."""
	if repeated_results.empty:
		return pd.DataFrame()
	group_cols = ["algorithm", "top_k_neighbors", "top_n"]
	available_metrics = [column for column in metric_columns if column in repeated_results.columns]
	aggregated = repeated_results.groupby(group_cols, dropna=False)[available_metrics].agg(["mean", "std"])
	aggregated.columns = [f"{metric}_{stat}" for metric, stat in aggregated.columns]
	aggregated = aggregated.reset_index()

	count_cols = ["n_users", "n_test_ratings", "n_users_with_relevant_items"]
	count_summary = repeated_results.groupby(group_cols, dropna=False)[count_cols].mean().reset_index()
	for column in count_cols:
		count_summary[column] = count_summary[column].round().astype(int)
	return count_summary.merge(aggregated, on=group_cols, how="left")


def load_default_matrix(project_root: Optional[Path] = None) -> pd.DataFrame:
	"""Locate and load the project's default filtered matrix."""
	if project_root is None:
		project_root = Path(__file__).resolve().parents[1]
	return load_filtered_matrix(project_root / "data" / "cleaned_data" / "user_item_matrix_sparse_filtered.csv")


def run_demo() -> Dict[str, pd.DataFrame]:
	"""Print a quick model comparison and top_k sensitivity demo."""
	matrix = load_default_matrix()
	train_matrix, test_matrix = train_test_split_matrix(matrix)
	demo_users = [uid for uid in test_matrix.index if test_matrix.loc[uid].notna().any()][:20]
	comparison = compare_models(matrix=matrix, user_ids=demo_users)
	sensitivity = sensitivity_analysis(matrix=matrix, algorithms=DEFAULT_ALGORITHMS, user_ids=demo_users)

	print("=== Item-CF model comparison ===")
	print(comparison.to_string(index=False))
	print("\n=== top_k sensitivity ===")
	print(sensitivity.to_string(index=False))

	return {
		"comparison": comparison,
		"sensitivity": sensitivity,
	}


def main() -> None:
	run_demo()


if __name__ == "__main__":
	main()
