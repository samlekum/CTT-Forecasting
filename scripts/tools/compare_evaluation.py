from pathlib import Path
import pandas as pd
import numpy as np
import json


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FILE_DAMPING0 = BASE_DIR / "evaluation" / "test_evaluation_detail_damping0.csv"
FILE_DAMPING05 = BASE_DIR / "evaluation" / "test_evaluation_detail_damping05.csv"

OUTPUT_DIR = BASE_DIR / "evaluation" / "comparison"

OUTPUT_JSON = OUTPUT_DIR / "comparison_summary.json"
OUTPUT_CSV = OUTPUT_DIR / "comparison_numeric.csv"


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    print(f"Loading: {path}")
    
    df = pd.read_csv(path, low_memory=False)

    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns):,}")

    return df


def basic_summary(df):
    numeric = df.select_dtypes(include=np.number)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "numeric_columns": list(numeric.columns),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    return summary


def numeric_summary(df):
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return pd.DataFrame()

    result = pd.DataFrame({
        "count": numeric.count(),
        "missing": numeric.isna().sum(),
        "mean": numeric.mean(),
        "median": numeric.median(),
        "std": numeric.std(),
        "min": numeric.min(),
        "q25": numeric.quantile(0.25),
        "q75": numeric.quantile(0.75),
        "max": numeric.max(),
    })

    return result


def detect_group_columns(df):
    """
    Cari kolom kategorikal yang kemungkinan berguna
    untuk membandingkan hasil per model / ticker / horizon.
    """

    preferred = [
        "model",
        "ticker",
        "symbol",
        "horizon",
        "target",
        "feature_set",
        "experiment",
        "experiment_id",
        "regime",
    ]

    found = []

    for col in preferred:
        if col in df.columns:
            found.append(col)

    return found


def grouped_summary(df, group_columns):
    if not group_columns:
        return {}

    result = {}

    numeric_columns = list(
        df.select_dtypes(include=np.number).columns
    )

    for col in group_columns:

        # Hindari groupby pada kolom dengan terlalu banyak unique values
        nunique = df[col].nunique(dropna=False)

        if nunique > 100:
            continue

        if not numeric_columns:
            continue

        grouped = (
            df.groupby(col, dropna=False)[numeric_columns]
            .agg(["mean", "median", "std"])
        )

        # Convert MultiIndex columns menjadi string
        grouped.columns = [
            f"{a}_{b}" for a, b in grouped.columns
        ]

        result[col] = grouped.reset_index().to_dict(orient="records")

    return result


def compare_numeric(df0, df05):
    """
    Membandingkan statistik numerik kedua dataset.
    """

    numeric0 = df0.select_dtypes(include=np.number)
    numeric05 = df05.select_dtypes(include=np.number)

    common_columns = sorted(
        set(numeric0.columns) & set(numeric05.columns)
    )

    rows = []

    for col in common_columns:

        a = numeric0[col]
        b = numeric05[col]

        mean0 = a.mean()
        mean05 = b.mean()

        median0 = a.median()
        median05 = b.median()

        std0 = a.std()
        std05 = b.std()

        row = {
            "column": col,

            "damping0_mean": mean0,
            "damping05_mean": mean05,
            "mean_difference": mean05 - mean0,

            "damping0_median": median0,
            "damping05_median": median05,
            "median_difference": median05 - median0,

            "damping0_std": std0,
            "damping05_std": std05,
            "std_difference": std05 - std0,

            "damping0_missing": int(a.isna().sum()),
            "damping05_missing": int(b.isna().sum()),
        }

        # Persentase perubahan mean
        if pd.notna(mean0) and mean0 != 0:
            row["mean_pct_change"] = (
                (mean05 - mean0) / abs(mean0) * 100
            )
        else:
            row["mean_pct_change"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def find_interesting_metrics(comparison_df, top_n=20):
    """
    Cari perubahan absolut terbesar.
    """

    if comparison_df.empty:
        return {}

    result = {}

    # Absolute mean difference
    tmp = comparison_df.dropna(
        subset=["mean_difference"]
    ).copy()

    if not tmp.empty:
        tmp["abs_difference"] = tmp["mean_difference"].abs()

        result["largest_mean_difference"] = (
            tmp.sort_values(
                "abs_difference",
                ascending=False
            )
            .head(top_n)
            .drop(columns=["abs_difference"])
            .to_dict(orient="records")
        )

    # Percentage change
    tmp = comparison_df.dropna(
        subset=["mean_pct_change"]
    ).copy()

    if not tmp.empty:
        tmp["abs_pct_change"] = tmp["mean_pct_change"].abs()

        result["largest_percentage_change"] = (
            tmp.sort_values(
                "abs_pct_change",
                ascending=False
            )
            .head(top_n)
            .drop(columns=["abs_pct_change"])
            .to_dict(orient="records")
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df0 = load_csv(FILE_DAMPING0)
    df05 = load_csv(FILE_DAMPING05)

    # --------------------------------------------------------
    # Basic summaries
    # --------------------------------------------------------

    summary0 = basic_summary(df0)
    summary05 = basic_summary(df05)

    # --------------------------------------------------------
    # Column comparison
    # --------------------------------------------------------

    columns0 = set(df0.columns)
    columns05 = set(df05.columns)

    column_comparison = {
        "only_in_damping0": sorted(columns0 - columns05),
        "only_in_damping05": sorted(columns05 - columns0),
        "common_columns": sorted(columns0 & columns05),
    }

    # --------------------------------------------------------
    # Numeric summaries
    # --------------------------------------------------------

    numeric0 = numeric_summary(df0)
    numeric05 = numeric_summary(df05)

    # --------------------------------------------------------
    # Numeric comparison
    # --------------------------------------------------------

    comparison = compare_numeric(df0, df05)

    if not comparison.empty:
        comparison.to_csv(
            OUTPUT_CSV,
            index=False
        )

    # --------------------------------------------------------
    # Grouped summaries
    # --------------------------------------------------------

    group_columns = sorted(
        set(
            detect_group_columns(df0)
            + detect_group_columns(df05)
        )
    )

    grouped0 = grouped_summary(df0, group_columns)
    grouped05 = grouped_summary(df05, group_columns)

    # --------------------------------------------------------
    # Interesting differences
    # --------------------------------------------------------

    interesting = find_interesting_metrics(comparison)

    # --------------------------------------------------------
    # Build JSON
    # --------------------------------------------------------

    result = {
        "files": {
            "damping0": str(FILE_DAMPING0),
            "damping05": str(FILE_DAMPING05),
        },

        "dataset_summary": {
            "damping0": summary0,
            "damping05": summary05,
        },

        "column_comparison": column_comparison,

        "numeric_summary": {
            "damping0": (
                numeric0.reset_index()
                .rename(columns={"index": "column"})
                .to_dict(orient="records")
            ),

            "damping05": (
                numeric05.reset_index()
                .rename(columns={"index": "column"})
                .to_dict(orient="records")
            ),
        },

        "numeric_comparison": (
            comparison.to_dict(orient="records")
            if not comparison.empty
            else []
        ),

        "group_columns_detected": group_columns,

        "grouped_summary": {
            "damping0": grouped0,
            "damping05": grouped05,
        },

        "interesting_differences": interesting,
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    print()
    print("Dataset:")
    print(f"  damping0  : {len(df0):,} rows")
    print(f"  damping05 : {len(df05):,} rows")

    print()
    print("Missing values:")
    print(
        f"  damping0  : "
        f"{df0.isna().sum().sum():,}"
    )
    print(
        f"  damping05 : "
        f"{df05.isna().sum().sum():,}"
    )

    print()
    print("Common numeric columns:")
    print(
        f"  {len(comparison):,}"
    )

    print()
    print("Group columns detected:")
    print(
        f"  {group_columns}"
    )

    print()
    print("Output:")
    print(f"  JSON : {OUTPUT_JSON}")
    print(f"  CSV  : {OUTPUT_CSV}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()