from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List

import pandas as pd


CSV_FILES = ("ratings", "tags", "movies", "links")


@dataclass
class FileCleanLog:
    name: str
    original_rows: int = 0
    final_rows: int = 0
    removed_rows: Dict[str, List[int]] = field(default_factory=dict)
    changed_values: List[str] = field(default_factory=list)
    summary_changes: List[str] = field(default_factory=list)

    def add_removed_rows(self, reason: str, indices: List[int]) -> None:
        if not indices:
            return
        if reason not in self.removed_rows:
            self.removed_rows[reason] = []
        self.removed_rows[reason].extend(indices)

    def add_change(self, index: int, column: str, old_value, new_value) -> None:
        if pd.isna(old_value) and pd.isna(new_value):
            return
        self.changed_values.append(
            f"row={index}, col={column}, old={repr(old_value)}, new={repr(new_value)}"
        )

    def add_summary(self, message: str) -> None:
        self.summary_changes.append(message)


@dataclass
class CleanReport:
    file_logs: Dict[str, FileCleanLog] = field(default_factory=dict)


def read_csv_files(data_dir: Path, file_names=CSV_FILES) -> Dict[str, pd.DataFrame]:
    dataframes: Dict[str, pd.DataFrame] = {}
    for name in file_names:
        file_path = data_dir / f"{name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"未找到文件: {file_path}")
        df = pd.read_csv(file_path)
        normalized_cols = []
        for col in df.columns:
            col_text = str(col).replace("\ufeff", "")
            col_text = re.sub(r"\s+", "", col_text)
            normalized_cols.append(col_text)
        df.columns = normalized_cols
        dataframes[name] = df
    return dataframes


def convert_columns_to_int(df: pd.DataFrame, columns: List[str], log: FileCleanLog) -> pd.DataFrame:
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

    changed_mask = raw.astype("string") != cleaned["timestamp"].astype("string")
    changed_mask = changed_mask.fillna(False)
    changed_count = int(changed_mask.sum())
    if changed_count > 0:
        sample_indices = cleaned.index[changed_mask][:5].tolist()
        samples = [
            f"{idx}:{raw.loc[idx]}->{cleaned.loc[idx, 'timestamp']}" for idx in sample_indices
        ]
        log.add_summary(
            f"timestamp 统一转换 {changed_count} 行，示例: {'; '.join(samples)}"
        )

    return cleaned


def fill_missing_values(df: pd.DataFrame, file_name: str, log: FileCleanLog) -> pd.DataFrame:
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
    cleaned = df.copy()

    if file_name == "tags" and "tag" in cleaned.columns:
        before = cleaned["tag"].astype("string")
        after = before.str.lower()
        changed_mask = (before != after).fillna(False)
        changed_count = int(changed_mask.sum())
        if changed_count > 0:
            sample_indices = cleaned.index[changed_mask][:5].tolist()
            samples = [f"{idx}:{before.loc[idx]}->{after.loc[idx]}" for idx in sample_indices]
            log.add_summary(f"tag 统一转小写 {changed_count} 行，示例: {'; '.join(samples)}")
        cleaned["tag"] = after

    return cleaned


def remove_full_duplicates(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    duplicate_mask = df.duplicated(keep="first")
    removed_indices = [int(i) for i in df.index[duplicate_mask].tolist()]
    log.add_removed_rows("完全重复行", removed_indices)
    return df.loc[~duplicate_mask].copy()


def deduplicate_latest_rating(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
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

    result = sort_df.loc[~dup_mask].drop(columns=["_ts_order"])
    result = result.sort_index()
    return result


def drop_missing_rating_rows(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    if "rating" not in df.columns:
        return df

    missing_mask = df["rating"].isna()
    removed_indices = [int(i) for i in df.index[missing_mask].tolist()]
    log.add_removed_rows("rating 缺失或非法，删除整行", removed_indices)
    return df.loc[~missing_mask].copy()


def clean_ratings(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    cleaned = convert_columns_to_int(df, ["userId", "movieId"], log)
    cleaned = convert_rating_and_fix_anomalies(cleaned, log)
    cleaned = convert_timestamp_column(cleaned, log)
    cleaned = remove_full_duplicates(cleaned, log)
    cleaned = drop_missing_rating_rows(cleaned, log)
    cleaned = deduplicate_latest_rating(cleaned, log)
    return cleaned


def clean_tags(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    cleaned = convert_columns_to_int(df, ["userId", "movieId"], log)
    cleaned = fill_missing_values(cleaned, "tags", log)
    cleaned = normalize_text_columns(cleaned, "tags", log)
    cleaned = convert_timestamp_column(cleaned, log)
    cleaned = remove_full_duplicates(cleaned, log)
    return cleaned


def clean_movies(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    cleaned = convert_columns_to_int(df, ["movieId"], log)
    cleaned = fill_missing_values(cleaned, "movies", log)
    cleaned = remove_full_duplicates(cleaned, log)
    return cleaned


def clean_links(df: pd.DataFrame, log: FileCleanLog) -> pd.DataFrame:
    cleaned = convert_columns_to_int(df, ["movieId", "imdbId", "tmdbId"], log)
    cleaned = remove_full_duplicates(cleaned, log)
    return cleaned


def save_cleaned_dataframes(dataframes: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in dataframes.items():
        output_path = output_dir / f"cleaned_{name}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"已保存: {output_path}")


def format_log_text(report: CleanReport) -> str:
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
                idx_text = ",".join(str(i) for i in indices) if indices else "无"
                lines.append(f"- 原因: {reason}")
                lines.append(f"  行索引: {idx_text}")

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
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"data_clean_log_{timestamp}.txt"
    log_text = format_log_text(report)
    log_path.write_text(log_text, encoding="utf-8")
    return log_path


def main() -> None:
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

    print("数据清洗完成。")
    print(f"日志已保存: {log_path}")


if __name__ == "__main__":
    main()
