# ============================================================
# Imports (mantidos o mais próximo possível do original)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import zscore

from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType, LongType, FloatType, DoubleType, DecimalType, ShortType
)

# ============================================================
# Helpers
# ============================================================
_NUMERIC_TYPES = (
    IntegerType, LongType, FloatType, DoubleType, DecimalType, ShortType
)

def _spark_numeric_columns(df):
    return [f.name for f in df.schema.fields if isinstance(f.dataType, _NUMERIC_TYPES)]


def _spark_to_pandas_safe(df, max_rows=200_000):
    """
    Converte DF já agregado (ou pequeno) para pandas de forma segura.
    """
    n = df.count()
    if n > max_rows:
        frac = max_rows / n
        df = df.sample(False, frac, seed=42)
    return df.toPandas()


# ============================================================
# 1. plot_missing_data  (VISUAL IDÊNTICO AO ORIGINAL)
# ============================================================
def plot_missing_data(df, figsize=(14, 6)):
    """
    Plot missing values analysis (count and percentage) with improved styling
    — versão PySpark, mantendo EXACTAMENTE o visual original
    """

    total_rows = df.count()

    # ==== Spark aggregation (substitui df.isnull().sum()) ====
    missing_counts = {
        c: df.select(F.sum(F.col(c).isNull().cast("int")).alias("cnt")).collect()[0]["cnt"]
        for c in df.columns
    }

    # ==== Volta para pandas (dados JÁ agregados) ====
    missing_count = pd.Series(missing_counts)
    missing_percent = (missing_count / total_rows) * 100

    # Filter only columns with missing values
    missing_count = missing_count[missing_count > 0].sort_values(ascending=True)
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=True)

    if len(missing_count) == 0:
        print("No missing values found in the DataFrame!")
        return

    # ==== DAQUI PARA BAIXO: CÓDIGO DE PLOT ORIGINAL (INALTERADO) ====
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    bars1 = ax1.barh(
        missing_count.index, missing_count.values,
        color='#2E86AB', alpha=0.8, height=0.6
    )
    ax1.set_title('Missing Values Count', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Number of Missing Values', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_facecolor('white')

    for i, (idx, v) in enumerate(missing_count.items()):
        ax1.text(v + 0.01*max(missing_count), i, f'{v:,}',
                 va='center', ha='left', fontsize=11, fontweight='bold')

    bars2 = ax2.barh(
        missing_percent.index, missing_percent.values,
        color='#A23B72', alpha=0.8, height=0.6
    )
    ax2.set_title('Missing Values Percentage', fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_facecolor('white')

    for i, (idx, v) in enumerate(missing_percent.items()):
        ax2.text(v + 0.01*max(missing_percent), i, f'{v:.1f}%',
                 va='center', ha='left', fontsize=11, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout()
    plt.show()


# ============================================================
# 2. plot_numerical_distributions (BINS, GRID, KDE IDÊNTICOS)
# ============================================================
def plot_numerical_distributions(df, features=None, figsize=(15, 6), cols_per_row=3):

    if features is None:
        features = _spark_numeric_columns(df)

    if len(features) == 0:
        print("No numerical features found.")
        return

    # Spark → pandas (somente colunas numéricas)
    pdf = _spark_to_pandas_safe(df.select(*features))

    n_features = len(features)
    n_rows = (n_features + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(
        n_rows, cols_per_row,
        figsize=(figsize[0], figsize[1] * n_rows)
    )
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        data = pdf[col].dropna()

        if data.empty:
            ax.set_title(f"{col} (no data)")
            ax.axis("off")
            continue

        sns.histplot(
            data,
            bins=30,          # 👈 IGUAL AO ORIGINAL
            kde=True,
            ax=ax,
            color="steelblue",
            edgecolor="white"
        )
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# 3. plot_numerical_boxplots (ESTÉTICA ORIGINAL)
# ============================================================
def plot_numerical_boxplots(df, features=None, figsize=(14, 6), cols_per_row=3):

    if features is None:
        features = _spark_numeric_columns(df)

    if not features:
        print("No numerical features found in the dataframe.")
        return

    pdf = _spark_to_pandas_safe(df.select(*features))

    n_features = len(features)
    n_rows = -(-n_features // cols_per_row)

    plt.style.use('default')
    fig, axes = plt.subplots(
        n_rows, cols_per_row,
        figsize=(figsize[0], figsize[1] * n_rows)
    )
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        sns.boxplot(
            x=pdf[col],
            ax=ax,
            color="#2E86AB",
            fliersize=3,
            linewidth=1.2
        )
        ax.set_title(col, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.set_facecolor("white")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Numerical Feature Distributions (Boxplots)",
        fontsize=18, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. plot_correlation_heatmap (PALETA E LIMITES IGUAIS)
# ============================================================
def plot_correlation_heatmap(df, features=None, figsize=(12, 8), annot=True):

    if features is None:
        features = _spark_numeric_columns(df)

    if len(features) < 2:
        print("Not enough numerical features.")
        return

    pdf = _spark_to_pandas_safe(df.select(*features))

    corr = pdf[features].corr()

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')

    cmap = LinearSegmentedColormap.from_list(
        "blue_red", ["#4A90E2", "#E74C3C"]
    )

    sns.heatmap(
        corr,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0.8,
        linecolor='lightgray',
        cbar_kws={"shrink": 0.8}
    )

    ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    ax.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()

################################################
def plot_categorical_distribution_same_style(
    df,
    features,
    figsize=(14, 6),
):
    """
    Plot categorical feature distributions (count + percentage)
    """

    plt.style.use('default')
    n_features = len(features)

    fig, axes = plt.subplots(
        nrows=n_features,
        ncols=2,
        figsize=(figsize[0], figsize[1] * n_features)
    )
    fig.patch.set_facecolor('white')

    if n_features == 1:
        axes = [axes]

    for i, feature in enumerate(features):

        # ==== Spark aggregation ====
        agg = (
            df.groupBy(feature)
            .count()
            .orderBy("count", ascending=True)
            .toPandas()
        )

        counts = pd.Series(
            agg["count"].values,
            index=agg[feature].astype(str)
        )

        percents = (counts / counts.sum()) * 100

        ax1, ax2 = axes[i]

        # ==== COUNT PLOT (COPIADO DO plot_missing_data) ====
        ax1.barh(
            counts.index,
            counts.values,
            color='#2E86AB',
            alpha=0.8,
            height=0.6
        )

        ax1.set_title(
            f'{feature} — Count',
            fontsize=18,
            fontweight='bold',
            pad=20
        )

        ax1.set_xlabel(
            'Number of Records',
            fontsize=14,
            fontweight='bold'
        )

        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_facecolor('white')

        for y, v in enumerate(counts.values):
            ax1.text(
                v + 0.01 * max(counts),
                y,
                f'{v:,}',
                va='center',
                ha='left',
                fontsize=11,
                fontweight='bold'
            )

        # ==== PERCENT PLOT (COPIADO DO plot_missing_data) ====
        ax2.barh(
            percents.index,
            percents.values,
            color='#A23B72',
            alpha=0.8,
            height=0.6
        )

        ax2.set_title(
            f'{feature} — Percentage',
            fontsize=18,
            fontweight='bold',
            pad=20
        )

        ax2.set_xlabel(
            'Percentage (%)',
            fontsize=14,
            fontweight='bold'
        )

        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.set_facecolor('white')

        for y, v in enumerate(percents.values):
            ax2.text(
                v + 0.01 * max(percents),
                y,
                f'{v:.1f}%',
                va='center',
                ha='left',
                fontsize=11,
                fontweight='bold'
            )

        # ==== Spines (INALTERADAS) ====
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout()
    plt.show()
######################################
def spark_shape(df):
    return df.count(), len(df.columns)
######################################
def check_unique_ids(df, id_col):
    """
    Check number of unique IDs and whether they match row count.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame.
    id_col : str
        Column name containing the ID.

    Returns
    -------
    dict
        {
            'row_count': int,
            'unique_id_count': int,
            'is_unique': bool
        }
    """

    row_count = df.count()

    unique_id_count = (
        df.agg(F.countDistinct(id_col).alias("unique_cnt"))
          .collect()[0]["unique_cnt"]
    )

    return {
        "row_count": row_count,
        "unique_id_count": unique_id_count,
        "is_unique": unique_id_count == row_count
    }
