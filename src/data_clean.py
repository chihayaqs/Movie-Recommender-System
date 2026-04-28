"""
本文件为数据清洗代码
清洗流程：
1. 读取 ratings/tags/movies/links 四个CSV原始数据文件。
2. 做类型转换、异常值处理、缺失值处理、去重。
3. 输出 cleaned_*.csv 到 data/cleaned_data。
4. 输出可追溯日志到 logs,日志说明详情时使用 CSV 文件行号(line)。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List
import pandas as pd

# 本次要处理的四个原始数据文件名称（不含 .csv 后缀）
CSV_FILES = ("ratings", "tags", "movies", "links")

def index_to_csv_line(index: int) -> int:
    """将 DataFrame 的 0-based 索引转换为 CSV 行号(含表头),方便日志记录时使用"""
    return int(index) + 2

@dataclass
class FileCleanLog:
    """单个文件的清洗日志容器"""
    name: str
    original_rows: int = 0
    final_rows: int = 0
    # 记录被删除的行：key=删除原因，value=被删行的原始index列表
    removed_rows: Dict[str, List[int]] = field(default_factory=dict)
    # 记录改动明细（行、列、旧值、新值）
    changed_values: List[str] = field(default_factory=list)
    # 记录改动汇总说明（如“统一转换了多少行”）
    summary_changes: List[str] = field(default_factory=list)
    def add_removed_rows(self, reason: str, indices: List[int]) -> None:
        """追加删除行记录"""
        if not indices:
            return
        if reason not in self.removed_rows:
            self.removed_rows[reason] = []
        self.removed_rows[reason].extend(indices)

    def add_change(self, index: int, column: str, old_value, new_value) -> None:
        """追加一条改动明细"""
        # 旧值和新值都为空时不记录
        if pd.isna(old_value) and pd.isna(new_value):
            return
        self.changed_values.append(
            f"line={index_to_csv_line(index)}, col={column}, old={repr(old_value)}, new={repr(new_value)}"
        )

    def add_summary(self, message: str) -> None:
        """追加改动汇总说明"""
        self.summary_changes.append(message)

@dataclass
class CleanReport:
    """完整的清洗报告"""
    file_logs: Dict[str, FileCleanLog] = field(default_factory=dict)


def read_csv_files(data_dir: Path, file_names=CSV_FILES) -> Dict[str, pd.DataFrame]:
    """
    读取CSV文件,并做列名标准化。
    列名标准化原因：
    部分文件可能有BOM(\ufeff)或隐藏空白字符,会导致 `tmdbId` 被读成 `' tmdbId'`，影响后续类型转换。
    """
    dataframes: Dict[str, pd.DataFrame] = {}
    for name in file_names:
        file_path = data_dir / f"{name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"未找到文件: {file_path}")
        df = pd.read_csv(file_path)
        normalized_cols: List[str] = []
        for col in df.columns:
            col_text = str(col).replace("\ufeff", "")
            col_text = re.sub(r"\s+", "", col_text)
            normalized_cols.append(col_text)
        df.columns = normalized_cols
        dataframes[name] = df
    return dataframes

def convert_columns_to_int(df: pd.DataFrame, columns: List[str], log: FileCleanLog) -> pd.DataFrame:
    """
    将指定列转为整数类型,适用于 userId、movieId、imdbId、tmdbId 等列
    规则：
    1)无法转数值的内容记为NA,并写日志
    2)数值先round再转Int64
    """
    cleaned = df.copy()
    for col in columns:
        if col not in cleaned.columns:
            continue
        raw = cleaned[col].copy()
        numeric = pd.to_numeric(cleaned[col], errors="coerce")
        invalid_mask = raw.notna() & numeric.isna()
        for idx in cleaned.index[invalid_mask]:
            log.add_change(int(idx), col, raw.loc[idx], pd.NA)
        cleaned[col] = numeric.round().astype("Int64")
    return cleaned

def convert_rating_and_fix_anomalies(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    """
    清洗rating列规则:
    1)转为浮点数;无法解析记为NA
    2)小于0或大于5的值记为NA
    3)0~5之间但非0.5步进,修正到最近的0.5档
    """
    cleaned = df.copy()
    if "rating" not in cleaned.columns:
        return cleaned
    raw_rating = cleaned["rating"].copy()
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce").astype("Float64")
    invalid_numeric_mask = raw_rating.notna() & cleaned["rating"].isna()
    for idx in cleaned.index[invalid_numeric_mask]:
        log.add_change(int(idx), "rating", raw_rating.loc[idx], pd.NA)

    out_of_range_mask = cleaned["rating"].notna() & (
        (cleaned["rating"] < 0) | (cleaned["rating"] > 5)
    )
    for idx in cleaned.index[out_of_range_mask]:
        log.add_change(int(idx), "rating", cleaned.loc[idx, "rating"], pd.NA)
    cleaned.loc[out_of_range_mask, "rating"] = pd.NA

    valid_mask = cleaned["rating"].notna()
    normalized = (cleaned.loc[valid_mask, "rating"] * 2).round() / 2
    needs_fix = (cleaned.loc[valid_mask, "rating"] - normalized).abs() > 1e-12

    fix_indices = cleaned.loc[valid_mask].index[needs_fix]
    for idx in fix_indices:
        old_val = cleaned.loc[idx, "rating"]
        new_val = float(normalized.loc[idx])
        log.add_change(int(idx), "rating", old_val, new_val)
        cleaned.loc[idx, "rating"] = new_val

    cleaned["rating"] = cleaned["rating"].astype("Float64")
    return cleaned


def convert_timestamp_column(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    """
    统一timestamp:
    输入语义为“Unix秒(UTC)”
    输出格式:YYYY-mm-dd HH:MM:SS(年月日 时分秒) (UTC)
    非法timestamp记为NA
    """
    cleaned = df.copy()
    if "timestamp" not in cleaned.columns:
        return cleaned

    raw = cleaned["timestamp"].copy()
    numeric_ts = pd.to_numeric(cleaned["timestamp"], errors="coerce")

    invalid_mask = raw.notna() & numeric_ts.isna()
    for idx in cleaned.index[invalid_mask]:
        log.add_change(int(idx), "timestamp", raw.loc[idx], pd.NA)

    dt = pd.to_datetime(numeric_ts, unit="s", utc=True, errors="coerce")
    cleaned["timestamp"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    cleaned.loc[dt.isna(), "timestamp"] = pd.NA

    changed_mask = (raw.astype("string") != cleaned["timestamp"].astype("string")).fillna(False)
    changed_count = int(changed_mask.sum())
    if changed_count > 0:
        sample_indices = cleaned.index[changed_mask][:5].tolist()
        samples = [
            f"line{index_to_csv_line(idx)}:{raw.loc[idx]}->{cleaned.loc[idx, 'timestamp']}"
            for idx in sample_indices
        ]
        log.add_summary(f"timestamp 统一转换 {changed_count} 行，示例: {'; '.join(samples)}")

    return cleaned

def fill_missing_values(df: pd.DataFrame, file_name: str, log: FileCleanLog) -> pd.DataFrame:
    """
    按文件特殊缺失处理：
    tags.csv: tag缺失则填充空字符串""
    movies.csv: genres缺失则填充"Unknown"
    """
    cleaned = df.copy()

    if file_name == "tags" and "tag" in cleaned.columns:
        missing_mask = cleaned["tag"].isna()
        missing_count = int(missing_mask.sum())
        for idx in cleaned.index[missing_mask]:
            log.add_change(int(idx), "tag", cleaned.loc[idx, "tag"], "")
        cleaned["tag"] = cleaned["tag"].fillna("")
        if missing_count > 0:
            log.add_summary(f"tag 缺失填充为空字符串: {missing_count} 行")

    if file_name == "movies" and "genres" in cleaned.columns:
        missing_mask = cleaned["genres"].isna()
        missing_count = int(missing_mask.sum())
        for idx in cleaned.index[missing_mask]:
            log.add_change(int(idx), "genres", cleaned.loc[idx, "genres"], "Unknown")
        cleaned["genres"] = cleaned["genres"].fillna("Unknown")
        if missing_count > 0:
            log.add_summary(f"genres 缺失填充为 Unknown: {missing_count} 行")

    return cleaned

def normalize_text_columns(df: pd.DataFrame, file_name: str, log: FileCleanLog) -> pd.DataFrame:
    """将 tags.csv 的 tag 列统一为小写"""
    cleaned = df.copy()

    if file_name == "tags" and "tag" in cleaned.columns:
        before = cleaned["tag"].astype("string")
        after = before.str.lower()

        changed_mask = (before != after).fillna(False)
        changed_count = int(changed_mask.sum())
        if changed_count > 0:
            sample_indices = cleaned.index[changed_mask][:5].tolist()
            samples = [
                f"line{index_to_csv_line(idx)}:{before.loc[idx]}->{after.loc[idx]}"
                for idx in sample_indices
            ]
            log.add_summary(f"tag 统一转小写 {changed_count} 行，示例: {'; '.join(samples)}")

        cleaned["tag"] = after

    return cleaned


def remove_full_duplicates(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    """删除绝对重复行（整行所有字段一致）"""
    duplicate_mask = df.duplicated(keep="first")
    removed_indices = [int(i) for i in df.index[duplicate_mask].tolist()]
    log.add_removed_rows("绝对重复行", removed_indices)
    return df.loc[~duplicate_mask].copy()


def deduplicate_latest_rating(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    """
    若同一用户对同一电影多次评分，只保留最新时间一条。
    实现说明：
    按 userId、movieId、timestamp 升序排序。
    对同组重复项保留最后一条(即最新 timestamp)。
    """
    if not {"userId", "movieId", "timestamp"}.issubset(df.columns):
        return df
    cleaned = df.copy()
    ts_dt = pd.to_datetime(cleaned["timestamp"], errors="coerce")

    sort_df = cleaned.assign(_ts_order=ts_dt).sort_values(
        by=["userId", "movieId", "_ts_order"],
        kind="mergesort",
        na_position="first",
    )

    dup_mask = sort_df.duplicated(subset=["userId", "movieId"], keep="last")
    removed_indices = [int(i) for i in sort_df.index[dup_mask].tolist()]
    log.add_removed_rows("同一用户同一电影多次评分，保留最新时间", removed_indices)

    result = sort_df.loc[~dup_mask].drop(columns=["_ts_order"]).sort_index()
    return result

def drop_missing_rating_rows(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    """rating 缺失时删除整行"""
    if "rating" not in df.columns:
        return df
    missing_mask = df["rating"].isna()
    removed_indices = [int(i) for i in df.index[missing_mask].tolist()]
    log.add_removed_rows("rating 缺失或非法，删除整行", removed_indices)
    return df.loc[~missing_mask].copy()

def clean_ratings(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    """ratings.csv清洗流程"""
    cleaned = convert_columns_to_int(df, ["userId", "movieId"], log)
    cleaned = convert_rating_and_fix_anomalies(cleaned, log)
    cleaned = convert_timestamp_column(cleaned, log)
    cleaned = remove_full_duplicates(cleaned, log)
    cleaned = drop_missing_rating_rows(cleaned, log)
    cleaned = deduplicate_latest_rating(cleaned, log)
    return cleaned


def clean_tags(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    """tags.csv清洗流程"""
    cleaned = convert_columns_to_int(df, ["userId", "movieId"], log)
    cleaned = fill_missing_values(cleaned, "tags", log)
    cleaned = normalize_text_columns(cleaned, "tags", log)
    cleaned = convert_timestamp_column(cleaned, log)
    cleaned = remove_full_duplicates(cleaned, log)
    return cleaned


def clean_movies(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    """movies.csv清洗流程"""
    cleaned = convert_columns_to_int(df, ["movieId"], log)
    cleaned = fill_missing_values(cleaned, "movies", log)
    cleaned = remove_full_duplicates(cleaned, log)
    return cleaned


def clean_links(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    """links.csv清洗流程"""
    cleaned = convert_columns_to_int(df, ["movieId", "imdbId", "tmdbId"], log)
    cleaned = remove_full_duplicates(cleaned, log)
    return cleaned


def save_cleaned_dataframes(dataframes: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """保存四个清洗后文件到 data/cleaned_data"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in dataframes.items():
        output_path = output_dir / f"cleaned_{name}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"已保存: {output_path}")


def format_log_text(report: CleanReport) -> str:
    """将清洗报告转化为可读日志文本"""
    lines: List[str] = []
    lines.append("电影推荐系统数据清洗日志")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    for file_name, file_log in report.file_logs.items():
        lines.append(f"\n文件: {file_name}.csv")
        lines.append(f"原始数据量: {file_log.original_rows}")
        removed_total = sum(len(v) for v in file_log.removed_rows.values())
        lines.append(f"删除行数: {removed_total}")
        lines.append(f"最终数据量: {file_log.final_rows}")
        lines.append("剔除的行:")
        if not file_log.removed_rows:
            lines.append("- 无")
        else:
            for reason, indices in file_log.removed_rows.items():
                line_text = ",".join(str(index_to_csv_line(i)) for i in indices) if indices else "无"
                lines.append(f"- 原因: {reason}")
                lines.append(f"  line: {line_text}")
        lines.append("改动内容:")
        if file_log.summary_changes:
            lines.append("- 汇总:")
            for item in file_log.summary_changes:
                lines.append(f"  - {item}")
        if not file_log.changed_values:
            lines.append("- 明细: 无")
        else:
            lines.append("- 明细:")
            for item in file_log.changed_values:
                lines.append(f"  - {item}")
        lines.append("-" * 80)
    return "\n".join(lines)

def save_log(report: CleanReport, logs_dir: Path) -> Path:
    """保存日志文件，文件名包含时间戳"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"data_clean_log_{timestamp}.txt"
    log_text = format_log_text(report)
    log_path.write_text(log_text, encoding="utf-8")
    return log_path

def main() -> None:
    """清洗流程:读取、分文件清洗、输出CSV、输出日志"""
    project_root = Path(__file__).resolve().parent.parent
    source_dir = project_root / "data" / "ml-latest-small"
    output_dir = project_root / "data" / "cleaned_data"
    logs_dir = project_root / "logs"
    raw_dataframes = read_csv_files(source_dir)
    report = CleanReport()
    cleaned_dataframes: Dict[str, pd.DataFrame] = {}
    for name, df in raw_dataframes.items():
        file_log = FileCleanLog(name=name, original_rows=len(df))
        if name == "ratings":
            cleaned_df = clean_ratings(df, file_log)
        elif name == "tags":
            cleaned_df = clean_tags(df, file_log)
        elif name == "movies":
            cleaned_df = clean_movies(df, file_log)
        elif name == "links":
            cleaned_df = clean_links(df, file_log)
        else:
            cleaned_df = remove_full_duplicates(df, file_log)
        file_log.final_rows = len(cleaned_df)
        report.file_logs[name] = file_log
        cleaned_dataframes[name] = cleaned_df
    save_cleaned_dataframes(cleaned_dataframes, output_dir)
    log_path = save_log(report, logs_dir)
    print("数据清洗完成")
    print(f"日志已保存: {log_path}")

if __name__ == "__main__":
    main()
