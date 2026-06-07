# Movie-Recommender-System电影推荐系统（协同过滤）
Python人工智能程序设计实践项目A06电影推荐系统（协同过滤）
## 项目简介

本项目旨在构建一个基于协同过滤的电影推荐系统，解决当前流媒体平台和在线影评网站中，如何为用户推荐个性化电影的问题，以此提升用户体验。推荐系统通过分析用户和电影之间的评分数据，通过协同过滤算法为用户推荐电影。我们选择了MovieLens数据集（ml-latest-small）作为数据源，使用协同过滤算法（包括基于用户和基于物品的协同过滤）实现推荐模型，能够输出指定用户的推荐结果，并评估其推荐效果。

## 数据来源

本项目使用的数据集来自 [MovieLens](https://grouplens.org/datasets/)，具体为 `ml-latest-small` 数据集。该数据集描述了MovieLens的五星评分和自由文本标记活动，包含 100836 个评分和 3683 个标签应用，涵盖 9742 部电影，数据由 610 名用户在 1996 年 3 月 29 日至 2018 年 9 月 24 日期间创建。

- 数据集链接：[MovieLens Datasets](https://grouplens.org/datasets/movielens/)
- 数据集文件：[ml-latest-small.zip](https://grouplens.org/datasets/movielens/)

## 参考代码链接

1. [Recommender-System-Collaborative-Filtering-MovieLens](https://github.com/pratiknabriya/Recommender-System-Collaborative-Filtering-MovieLens)
2. [MovieMatcher-Movie-Recommender-System](https://github.com/abhipatel35/MovieMatcher-Movie-Recommender-System)

## 环境依赖

- Python >= 3.7
- pandas
- numpy
- scikit-learn
- matplotlib
- surprise
- seaborn

## 安装与运行

1. 克隆项目仓库：

   ```bash
   git clone https://github.com/chihayaqs/Movie-Recommender-System.git
2. 安装依赖：
    ```bash
   pip install -r requirements.txt
3. 运行推荐系统：
4. 查看推荐结果

## 项目结构
```text
Movie-Recommender-System/
│
├── data/                    # 数据文件夹
│   ├── ml-latest-small/     # MovieLens 数据集
|   ├── cleaned_data/        #清洗后的数据文件
│
├── src/                     # 源代码
│   ├── data_clean.py        # 数据清洗代码
│   ├── data_processing.py   # 数据处理：构建矩阵、稀疏矩阵处理
|   ├── model.py             # 模型训练：协同过滤实现
|   ├── evaluation.py        # 模型评估：评估指标计算
│   └── utils.py             # 功能函数
|
├── logs/                    #日志文件
|
├── notebooks/              
|   ├── analysis.ipynb       # 数据可视化
|   └── model_test.ipynb     # 原型验证、对比实验
|
├── requirements.txt         # 环境依赖
└── README.md                # 项目说明
 
