# Movie-Recommender-System 电影推荐系统

Python 人工智能程序设计实践项目：基于 MovieLens 数据集的电影推荐系统。项目围绕协同过滤推荐完成了数据清洗、探索性分析、评分矩阵构建、物品协同过滤建模、全量评估、模型对比、交互式页面展示。

## 项目概览

本项目使用 MovieLens `ml-latest-small` 数据集构建电影推荐系统。系统核心是基于物品的协同过滤模型：通过用户-电影评分矩阵计算电影之间的相似度，再根据目标用户的历史评分预测其可能喜欢的电影。

当前版本包含两类模型：

- `baseline_numpy_item_cf`：位于 `src/model.py`，使用 NumPy 手写物品协同过滤，作为基线模型。
- `sklearn_centered_item_cf`：位于 `src/cf_recommender.py`，作为当前主模型，使用 `scikit-learn` 计算余弦相似度，并加入用户均值中心化、相似度 shrinkage、最低共评人数约束和 Top-N 轻量重排序。

评估逻辑统一位于 `src/evaluation.py`。该文件只负责训练/测试划分、指标计算、模型调用与结果汇总，不承载模型本身的优化逻辑。

## 数据来源

数据集来自 [MovieLens Datasets](https://grouplens.org/datasets/movielens/)，具体使用 `ml-latest-small`。

原始数据规模：

- 用户数：610
- 电影数：9724
- 评分数：100836
- 标签数：3683

经过活跃用户和热门电影过滤后，项目用于建模和评估的评分矩阵保留：

- 用户数：475
- 电影数：1617
- 评分数：70454
- 矩阵稀疏率：90.83%

## 项目结构

```text
Movie-Recommender-System/
├─ data/
│  ├─ ml-latest-small/              # MovieLens 原始数据
│  └─ cleaned_data/                 # 清洗后数据与评分矩阵
├─ logs/                            # 数据清洗日志
├─ notebooks/
│  ├─ analysis.ipynb                # 数据分析、矩阵过滤与可视化
│  ├─ model_evaluation_visualization.ipynb
│  │                                  # 主模型快速演示、全量评估与可视化
│  └─ model_comparison.ipynb         # 基线模型与主模型的全量对比实验
├─ src/
│  ├─ data_clean.py                 # 原始 CSV 清洗
│  ├─ data_processing.py            # 构建用户-电影评分矩阵
│  ├─ model.py                      # NumPy 基线物品协同过滤
│  ├─ cf_recommender.py             # 优化后的主模型与推荐函数
│  └─ evaluation.py                 # 评估流程与指标计算
├─ demo.py                          # Streamlit 交互式推荐系统页面
└─ requirements.txt                 # 项目运行依赖
```

## 环境安装

建议使用 Python 3.9 及以上版本。项目当前主要依赖 `numpy`、`pandas`、`scikit-learn`、`matplotlib`、`seaborn`、`streamlit` 和 Jupyter 相关组件。

```bash
pip install -r requirements.txt
```

如果需要在 Jupyter 中选择独立内核，可额外执行：

```bash
python -m ipykernel install --user --name python_course --display-name python_course
```

## 运行流程

### 1. 数据清洗

```bash
python src/data_clean.py
```

输出清洗后的 CSV 文件到 `data/cleaned_data/`，并在 `logs/` 中记录清洗日志。

### 2. 构建评分矩阵

```bash
python src/data_processing.py
```

输出用户-电影评分矩阵，包括完整矩阵和过滤后的稀疏矩阵。

### 3. 运行主模型推荐

```bash
python src/cf_recommender.py
```

该脚本会读取过滤后的评分矩阵，为样例用户生成基于优化后主模型的 Top-N 推荐结果。

### 4. 运行评估 Notebook

推荐按以下顺序运行：

1. `notebooks/analysis.ipynb`：数据探索、稀疏矩阵分析与过滤过程可视化。
2. `notebooks/model_evaluation_visualization.ipynb`：只评估当前主模型，保留 20 用户快速演示，并给出 475 个可评估用户、14093 条测试评分的全量评估和可视化。
3. `notebooks/model_comparison.ipynb`：进行基线模型与优化主模型的全量对比，包括 `top_k` 敏感性分析、多随机种子稳定性评估和指定用户 Top-10 推荐案例。

### 5. 启动 Streamlit 展示页面

```bash
streamlit run demo.py
```

页面提供简化电影网站形式的交互展示，包括首页推荐、电影检索、类型浏览和指定 `userId` 的个性化推荐结果。


## 参考资料

1. [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
2. F. M. Harper and J. A. Konstan. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 2015.
3. [scikit-learn cosine_similarity Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
4. [Recommender-System-Collaborative-Filtering-MovieLens](https://github.com/pratiknabriya/Recommender-System-Collaborative-Filtering-MovieLens)
5. [MovieMatcher-Movie-Recommender-System](https://github.com/abhipatel35/MovieMatcher-Movie-Recommender-System)
