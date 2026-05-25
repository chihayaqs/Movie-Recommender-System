from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cf_recommender import compute_item_similarity, load_user_item_matrix, predict_ratings_item_based


st.set_page_config(
    page_title="电影推荐 Demo",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main {
        background: #f7f8fb;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.35rem 1.5rem;
        border-radius: 8px;
        color: #f8fafc;
        background: linear-gradient(135deg, #111827 0%, #273244 54%, #9f2936 100%);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        letter-spacing: 0;
    }
    .hero p {
        margin: .45rem 0 0;
        color: #dbe4ef;
        font-size: .98rem;
    }
    .metric-card {
        padding: .9rem 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #ffffff;
    }
    .metric-card span {
        display: block;
        color: #6b7280;
        font-size: .82rem;
        margin-bottom: .25rem;
    }
    .metric-card strong {
        color: #111827;
        font-size: 1.35rem;
    }
    .movie-card {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #ffffff;
        min-height: 150px;
    }
    .movie-card h3 {
        margin: 0 0 .55rem;
        font-size: 1.05rem;
        letter-spacing: 0;
    }
    .tag {
        display: inline-block;
        padding: .15rem .45rem;
        margin: 0 .25rem .25rem 0;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: .76rem;
    }
    .muted {
        color: #6b7280;
        font-size: .9rem;
    }
    div[data-testid="stDataFrame"] {
        background: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="正在读取电影和评分数据...")
def load_movie_data(project_root: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    root = Path(project_root)
    movies = pd.read_csv(root / "data" / "cleaned_data" / "cleaned_movies.csv")
    ratings = pd.read_csv(root / "data" / "cleaned_data" / "cleaned_ratings.csv")

    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce").astype("Int64")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce").astype("Int64")
    ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce").astype("Int64")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

    rating_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )
    movie_stats = movies.merge(rating_stats, on="movieId", how="left")
    movie_stats["avg_rating"] = movie_stats["avg_rating"].fillna(0.0)
    movie_stats["rating_count"] = movie_stats["rating_count"].fillna(0).astype(int)
    movie_stats["genres"] = movie_stats["genres"].fillna("未知")

    genres = sorted(
        {
            genre
            for value in movie_stats["genres"].astype(str)
            for genre in value.split("|")
            if genre and genre != "(no genres listed)"
        }
    )
    return movies, ratings, movie_stats, genres


@st.cache_resource(show_spinner="正在加载协同过滤模型...")
def load_recommender(project_root: str):
    matrix = load_user_item_matrix(Path(project_root))
    sim_matrix = compute_item_similarity(matrix)
    return matrix, sim_matrix


def format_movie_table(df: pd.DataFrame) -> pd.DataFrame:
    view = df[["title", "genres", "avg_rating", "rating_count"]].copy()
    view = view.rename(
        columns={
            "title": "电影名",
            "genres": "类型",
            "avg_rating": "平均评分",
            "rating_count": "评分人数",
        }
    )
    view["平均评分"] = view["平均评分"].round(2)
    return view


def movie_options(df: pd.DataFrame) -> dict[str, int]:
    options = {}
    for row in df.itertuples(index=False):
        options[f"{row.title}  |  {row.genres}"] = int(row.movieId)
    return options


def show_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"<div class='metric-card'><span>{label}</span><strong>{value}</strong></div>",
        unsafe_allow_html=True,
    )


def show_movie_cards(df: pd.DataFrame, score_label: str = "平均评分") -> None:
    cols = st.columns(5)
    for idx, row in enumerate(df.head(10).itertuples(index=False)):
        tags = "".join(f"<span class='tag'>{genre}</span>" for genre in str(row.genres).split("|")[:3])
        with cols[idx % 5]:
            st.markdown(
                f"""
                <div class="movie-card">
                    <h3>{row.title}</h3>
                    <div>{tags}</div>
                    <p class="muted">{score_label}：{row.avg_rating:.2f}<br>评分人数：{int(row.rating_count)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_movie_detail(movie_id: int, movie_stats: pd.DataFrame) -> None:
    movie = movie_stats[movie_stats["movieId"] == movie_id]
    if movie.empty:
        st.info("暂时没有找到这部电影的详情。")
        return

    row = movie.iloc[0]
    st.subheader("电影详情")
    col1, col2, col3, col4 = st.columns([2.2, 1.6, 1, 1])
    col1.markdown(f"**电影名**  \n{row['title']}")
    col2.markdown(f"**电影类型**  \n{row['genres']}")
    col3.markdown(f"**平均评分**  \n{row['avg_rating']:.2f}")
    col4.markdown(f"**评分人数**  \n{int(row['rating_count'])}")


def filter_by_genre(movie_stats: pd.DataFrame, genre: str) -> pd.DataFrame:
    if genre == "全部类型":
        return movie_stats.copy()
    return movie_stats[movie_stats["genres"].str.contains(genre, regex=False, na=False)].copy()


def user_profile(user_id: int, ratings: pd.DataFrame, movie_stats: pd.DataFrame) -> pd.DataFrame:
    history = ratings[ratings["userId"] == user_id].merge(
        movie_stats[["movieId", "title", "genres", "avg_rating", "rating_count"]],
        on="movieId",
        how="left",
    )
    return history.sort_values(["rating", "timestamp"], ascending=[False, False])


def favorite_genres(history: pd.DataFrame) -> str:
    high_rated = history[history["rating"] >= 4.0]
    genre_counts: dict[str, int] = {}
    for genres in high_rated["genres"].dropna().astype(str):
        for genre in genres.split("|"):
            if genre and genre != "(no genres listed)":
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    if not genre_counts:
        return "暂无明显偏好"
    return "、".join([name for name, _ in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]])


def recommend_for_user(
    user_id: int,
    matrix: pd.DataFrame,
    sim_matrix: pd.DataFrame,
    movie_stats: pd.DataFrame,
    top_k: int,
    n: int,
) -> pd.DataFrame:
    preds = predict_ratings_item_based(matrix, sim_matrix, user_id, top_k=top_k)
    recs = (
        preds.dropna()
        .head(n)
        .rename("pred_rating")
        .reset_index()
        .rename(columns={"index": "movieId"})
    )
    recs["movieId"] = pd.to_numeric(recs["movieId"], errors="coerce").astype("Int64")
    recs = recs.merge(movie_stats, on="movieId", how="left")
    return recs.sort_values("pred_rating", ascending=False)


movies, ratings, movie_stats, genres = load_movie_data(str(PROJECT_ROOT))
matrix, sim_matrix = load_recommender(str(PROJECT_ROOT))

st.markdown(
    """
    <div class="hero">
        <h1>电影推荐 Demo</h1>
        <p>一个简化版电影网站原型：浏览热门与高分电影，搜索片名，按类型探索，并获得个性化推荐。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
with metric_cols[0]:
    show_metric_card("电影数量", f"{len(movies):,}")
with metric_cols[1]:
    show_metric_card("评分数量", f"{len(ratings):,}")
with metric_cols[2]:
    show_metric_card("用户数量", f"{ratings['userId'].nunique():,}")
with metric_cols[3]:
    show_metric_card("推荐模型电影数", f"{len(matrix.columns):,}")

page = st.sidebar.radio(
    "功能导航",
    ["首页推荐", "找电影", "我的推荐"],
)
st.sidebar.caption("数据来源：MovieLens ml-latest-small")

if page == "首页推荐":
    st.header("首页推荐")

    hot_tab, high_tab, genre_tab = st.tabs(["热门电影", "高分电影", "类型浏览"])

    with hot_tab:
        st.subheader("评分人数最多的电影")
        hot_movies = movie_stats.sort_values(["rating_count", "avg_rating"], ascending=False).head(10)
        show_movie_cards(hot_movies, score_label="平均评分")
        st.dataframe(format_movie_table(hot_movies), use_container_width=True, hide_index=True)

    with high_tab:
        min_count = st.slider("最低评分人数", 10, 200, 50, step=10)
        high_movies = (
            movie_stats[movie_stats["rating_count"] >= min_count]
            .sort_values(["avg_rating", "rating_count"], ascending=False)
            .head(10)
        )
        st.subheader("评分稳定的高分电影")
        show_movie_cards(high_movies, score_label="平均评分")
        st.dataframe(format_movie_table(high_movies), use_container_width=True, hide_index=True)

    with genre_tab:
        col1, col2 = st.columns([1, 1])
        selected_genre = col1.selectbox("选择电影类型", genres)
        rank_mode = col2.selectbox("排序方式", ["评分人数优先", "平均评分优先"])
        genre_movies = filter_by_genre(movie_stats, selected_genre)
        if rank_mode == "评分人数优先":
            genre_movies = genre_movies.sort_values(["rating_count", "avg_rating"], ascending=False)
        else:
            genre_movies = genre_movies[genre_movies["rating_count"] >= 20].sort_values(
                ["avg_rating", "rating_count"], ascending=False
            )
        st.subheader(f"{selected_genre} 类型电影")
        st.dataframe(format_movie_table(genre_movies.head(20)), use_container_width=True, hide_index=True)

elif page == "找电影":
    st.header("找电影")

    search_col, genre_col = st.columns([2, 1])
    keyword = search_col.text_input("输入电影关键词", placeholder="例如 Toy Story、Matrix、爱情")
    selected_genre = genre_col.selectbox("按类型筛选", ["全部类型"] + genres)

    results = filter_by_genre(movie_stats, selected_genre)
    if keyword.strip():
        pattern = keyword.strip()
        results = results[results["title"].str.contains(pattern, case=False, regex=False, na=False)]

    results = results.sort_values(["rating_count", "avg_rating"], ascending=False)
    st.caption(f"找到 {len(results)} 部电影")
    st.dataframe(format_movie_table(results.head(30)), use_container_width=True, hide_index=True)

    if not results.empty:
        options = movie_options(results.head(100))
        selected_label = st.selectbox("选择一部电影查看详情", list(options.keys()))
        show_movie_detail(options[selected_label], movie_stats)
    else:
        st.info("没有找到匹配的电影，可以换一个关键词或类型试试。")

else:
    st.header("我的推荐")

    user_ids = sorted(int(user_id) for user_id in matrix.index)
    user_mode = st.radio("用户选择方式", ["从已有用户选择", "手动输入 userId"], horizontal=True)
    col1, col2, col3 = st.columns([1.3, 1, 1])
    if user_mode == "从已有用户选择":
        selected_user = col1.selectbox("选择用户 userId", user_ids, index=0)
    else:
        selected_user = col1.number_input("输入 userId", min_value=1, value=user_ids[0], step=1)
    selected_user = int(selected_user)
    rec_count = col2.slider("推荐数量", 5, 20, 10)
    top_k = col3.slider("相似电影数量", 5, 50, 20, step=5)

    history = user_profile(selected_user, ratings, movie_stats)
    if selected_user not in matrix.index or history.empty:
        if history.empty:
            st.warning("这个用户暂时没有历史评分，建议先浏览热门电影作为冷启动推荐。")
        else:
            st.warning("这个用户的历史评分不足以进入协同过滤模型，先展示热门电影作为冷启动推荐。")
        fallback = movie_stats.sort_values(["rating_count", "avg_rating"], ascending=False).head(rec_count)
        st.dataframe(format_movie_table(fallback), use_container_width=True, hide_index=True)
    else:
        profile_cols = st.columns(4)
        with profile_cols[0]:
            show_metric_card("历史评分数", f"{len(history):,}")
        with profile_cols[1]:
            show_metric_card("平均打分", f"{history['rating'].mean():.2f}")
        with profile_cols[2]:
            show_metric_card("最高打分", f"{history['rating'].max():.1f}")
        with profile_cols[3]:
            show_metric_card("偏好类型", favorite_genres(history))

        if len(history) < 5:
            st.warning("该用户历史评分较少，推荐结果可能不够稳定。可以先参考热门电影列表。")

        st.subheader("最近/高分历史评分")
        history_view = history[["title", "genres", "rating", "avg_rating", "rating_count"]].head(15).rename(
            columns={
                "title": "电影名",
                "genres": "类型",
                "rating": "用户评分",
                "avg_rating": "全站平均评分",
                "rating_count": "评分人数",
            }
        )
        history_view["全站平均评分"] = history_view["全站平均评分"].round(2)
        st.dataframe(history_view, use_container_width=True, hide_index=True)

        st.subheader("个性化推荐结果")
        with st.spinner("正在根据该用户的历史评分生成推荐..."):
            recs = recommend_for_user(selected_user, matrix, sim_matrix, movie_stats, top_k=top_k, n=rec_count)

        if recs.empty:
            st.info("暂时没有生成可用推荐，先展示热门电影作为备用推荐。")
            recs = movie_stats.sort_values(["rating_count", "avg_rating"], ascending=False).head(rec_count)
            st.dataframe(format_movie_table(recs), use_container_width=True, hide_index=True)
        else:
            rec_view = recs[["title", "genres", "pred_rating", "avg_rating", "rating_count"]].rename(
                columns={
                    "title": "电影名",
                    "genres": "类型",
                    "pred_rating": "预测评分",
                    "avg_rating": "全站平均评分",
                    "rating_count": "评分人数",
                }
            )
            rec_view["预测评分"] = rec_view["预测评分"].round(2)
            rec_view["全站平均评分"] = rec_view["全站平均评分"].round(2)
            st.dataframe(rec_view, use_container_width=True, hide_index=True)
