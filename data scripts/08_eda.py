"""Step 8 - EDA: charts for the report, made only from the real data tables.

Reads : data/processed/papers_flagged.csv, papers_clean.csv, cleaning_summary.csv,
        missing_values_report.csv
Saves : eda/*.png   (one chart per question we want to answer)
        logs/eda_log.txt   (the key numbers behind each chart, for the written interpretation)
"""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import PROCESSED_DIR, ROOT, log

EDA_DIR = ROOT / "eda"
LOG = "eda_log.txt"
TYPE_ORDER = ["quiz", "mid", "final", "exam_unspecified"]
TYPE_COLORS = {"quiz": "#4C72B0", "mid": "#DD8452", "final": "#55A868", "exam_unspecified": "#8C8C8C"}
sns.set_theme(style="whitegrid", font_scale=0.95)


def save(figure, name):
    figure.tight_layout()
    figure.savefig(EDA_DIR / name, dpi=150)
    plt.close(figure)
    log(f"Saved eda/{name}", LOG)


def chart_papers_by_type(flagged, clean):
    """Class distribution of assessment types, before and after cleaning."""
    before = flagged["assessment_type"].value_counts().reindex(TYPE_ORDER).fillna(0)
    after = clean["assessment_type"].value_counts().reindex(TYPE_ORDER).fillna(0)
    table = pd.DataFrame({"Before cleaning": before, "After cleaning": after})
    ax = table.plot(kind="bar", figsize=(7, 4.5), color=["#B0B0B0", "#4C72B0"], rot=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=8)
    ax.set_title("Exam papers by assessment type")
    ax.set_xlabel("Assessment type")
    ax.set_ylabel("Number of papers")
    save(ax.figure, "01_papers_by_type.png")
    log(f"Papers by type before: {before.astype(int).to_dict()}  after: {after.astype(int).to_dict()}", LOG)


def chart_papers_by_subject(clean):
    """How many papers each subject has (top 20) - shows how uneven the data is."""
    counts = clean["subject"].fillna("Unknown").value_counts()
    top = counts.head(20).iloc[::-1]
    figure, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_title(f"Papers per subject (top 20 of {len(counts)} subjects)")
    ax.set_xlabel("Number of papers (after cleaning)")
    ax.set_ylabel("Subject")
    save(figure, "02_papers_by_subject.png")
    log(f"Subjects: {len(counts)}; median papers per subject: {counts.median():.0f}; "
        f"subjects with fewer than 5 papers: {int((counts < 5).sum())}; "
        f"top 5: {counts.head(5).to_dict()}", LOG)


def chart_papers_by_year(clean):
    """Papers per year, stacked by assessment type (only papers whose year is known)."""
    known = clean[clean["year"].notna()].copy()
    known["year"] = known["year"].astype(int)
    table = (known.groupby(["year", "assessment_type"]).size().unstack(fill_value=0)
             .reindex(columns=[t for t in TYPE_ORDER if t in known["assessment_type"].unique()]))
    table = table.reindex(range(known["year"].min(), known["year"].max() + 1), fill_value=0)  # show empty years
    ax = table.plot(kind="bar", stacked=True, figsize=(9, 4.5), rot=45,
                    color=[TYPE_COLORS[c] for c in table.columns])
    ax.set_title(f"Papers per year (year known for {len(known)} of {len(clean)} papers)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.legend(title="Assessment type")
    save(ax.figure, "03_papers_by_year.png")
    log(f"Year known: {len(known)}/{len(clean)}; year range {known['year'].min()}-{known['year'].max()}; "
        f"busiest years: {known['year'].value_counts().head(3).to_dict()}", LOG)
    terms = clean["term"].fillna("Unknown").value_counts().to_dict()
    log(f"Papers by term: {terms}", LOG)


def chart_subject_year_heatmap(clean):
    """Which subject has papers in which year (top 15 subjects)."""
    known = clean[clean["year"].notna()].copy()
    known["year"] = known["year"].astype(int)
    top = known["subject"].value_counts().head(15).index
    table = (known[known["subject"].isin(top)].pivot_table(index="subject", columns="year",
                                                            values="file_id", aggfunc="count", fill_value=0)
             .loc[top])
    table = table.reindex(columns=range(known["year"].min(), known["year"].max() + 1), fill_value=0)
    figure, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(table, cmap="Blues", linewidths=0.4, linecolor="white", cbar_kws={"label": "Papers"}, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Coverage: papers per subject and year (top 15 subjects)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Subject")
    save(figure, "04_subject_year_heatmap.png")
    log(f"Heatmap: share of empty subject-year cells among top 15 subjects: "
        f"{(table == 0).to_numpy().mean():.0%}", LOG)


def chart_missing_values():
    """Percent missing per field (from the missing-values report)."""
    report = pd.read_csv(PROCESSED_DIR / "missing_values_report.csv").iloc[::-1]
    figure, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(report["field"], report["percent"], color="#C44E52")
    for y, (pct, miss, total) in enumerate(zip(report["percent"], report["missing"], report["out_of"])):
        ax.text(pct + 0.8, y, f"{pct:.1f}%  ({miss}/{total})", va="center", fontsize=8)
    ax.set_xlim(0, 135)
    ax.set_title("Missing values per field (before cleaning, all exam papers)")
    ax.set_xlabel("Percent missing (denominator differs per field, shown in brackets)")
    ax.set_ylabel("Field")
    save(figure, "05_missing_values.png")
    log("Missing values: " + "; ".join(f"{r.field} {r.percent}%" for r in report.iloc[::-1].itertuples()), LOG)


def chart_text_coverage(clean):
    """Which papers have readable text (the rest are scans/photos that need OCR)."""
    table = (clean.groupby(["assessment_type", "has_text"]).size().unstack(fill_value=0)
             .reindex([t for t in TYPE_ORDER if t in clean["assessment_type"].unique()]))
    table = table.reindex(columns=[True, False], fill_value=0)
    table.columns = ["Readable text", "No text (needs OCR)"]
    ax = table.plot(kind="bar", stacked=True, figsize=(7, 4.5), rot=0, color=["#55A868", "#C44E52"])
    ax.set_title("Text availability by assessment type (after cleaning)")
    ax.set_xlabel("Assessment type")
    ax.set_ylabel("Number of papers")
    save(ax.figure, "06_text_coverage.png")
    share = (table.div(table.sum(axis=1), axis=0) * 100).round(1)
    log(f"Share of papers WITHOUT text by type (%): {share['No text (needs OCR)'].to_dict()}", LOG)
    by_subject = clean.groupby("subject")["has_text"].sum()
    log(f"Subjects with zero readable papers: {int((by_subject == 0).sum())} of {len(by_subject)}", LOG)


def chart_cleaning_funnel():
    """How many papers remain after each cleaning step."""
    summary = pd.read_csv(PROCESSED_DIR / "cleaning_summary.csv")
    figure, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(summary["step"], summary["rows_remaining"], color="#4C72B0")
    for bar, flagged in zip(bars, summary["rows_flagged"]):
        label = f"{int(bar.get_height())}" + (f"\n(-{int(flagged)})" if flagged else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15, label, ha="center", fontsize=8)
    ax.set_ylim(0, summary["rows_remaining"].max() * 1.15)
    ax.set_title("Cleaning: papers remaining after each step")
    ax.set_xlabel("Cleaning step")
    ax.set_ylabel("Papers remaining")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(figure, "07_cleaning_funnel.png")
    log("Cleaning: " + "; ".join(f"{r.step} -{r.rows_flagged} -> {r.rows_remaining}"
                                  for r in summary.itertuples()), LOG)


def chart_numeric_boxplots(clean):
    """Distribution of the numeric features per assessment type (outliers shown as dots)."""
    data = clean.copy()
    data["word_count"] = data["word_count"].where(data["has_text"])
    columns = [("page_count", "Pages"), ("word_count", "Words in text"),
               ("total_marks", "Total marks"), ("question_count_est", "Questions (estimate)")]
    order = [t for t in TYPE_ORDER if t in data["assessment_type"].unique() and t != "exam_unspecified"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (column, label) in zip(axes.ravel(), columns):
        sns.boxplot(data=data, x="assessment_type", y=column, order=order, palette=TYPE_COLORS,
                    hue="assessment_type", legend=False, fliersize=2, ax=ax)
        ax.set_title(f"{label} by assessment type (n={int(data[column].notna().sum())})")
        ax.set_xlabel("Assessment type")
        ax.set_ylabel(label)
        if column in ("page_count", "word_count"):
            ax.set_yscale("log")
    figure.suptitle("Numeric features per assessment type (dots = IQR outliers)", y=1.0)
    save(figure, "08_numeric_boxplots.png")
    for column, _ in columns:
        medians = data.groupby("assessment_type")[column].median().round(1).to_dict()
        log(f"Median {column} by type: {medians}", LOG)


def chart_correlation(clean):
    """Spearman correlation between the numeric features (uses pairs with both values)."""
    data = clean[["year", "page_count", "word_count", "total_marks", "question_count_est"]].copy()
    data["word_count"] = data["word_count"].where(clean["has_text"])
    corr = data.corr(method="spearman")
    figure, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Spearman correlation"})
    ax.set_title("Correlation between numeric features")
    save(figure, "09_correlation.png")
    pairs = {f"{a} vs {b}": round(float(corr.loc[a, b]), 2) for i, a in enumerate(corr.index)
             for b in corr.columns[i + 1:]}
    log(f"Spearman correlations: {pairs}", LOG)


def main():
    EDA_DIR.mkdir(exist_ok=True)
    flagged = pd.read_csv(PROCESSED_DIR / "papers_flagged.csv")
    clean = pd.read_csv(PROCESSED_DIR / "papers_clean.csv")
    log(f"EDA on {len(clean)} clean papers ({len(flagged)} before cleaning)", LOG)

    chart_papers_by_type(flagged, clean)
    chart_papers_by_subject(clean)
    chart_papers_by_year(clean)
    chart_subject_year_heatmap(clean)
    chart_missing_values()
    chart_text_coverage(clean)
    chart_cleaning_funnel()
    chart_numeric_boxplots(clean)
    chart_correlation(clean)
    log("EDA finished", LOG)


if __name__ == "__main__":
    main()