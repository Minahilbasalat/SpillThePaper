"""Contribution table: map Git commit authors to the modules they changed.

Run AFTER all three branches are merged to main:
    git checkout main
    git pull origin main
    python -m src.evaluation.contribution_table

Test on the current branch history (writes to a throw-away folder):
    python -m src.evaluation.contribution_table --ref HEAD --out-dir results/dummy

A commit counts once for every module it touches. Merge commits are excluded.
If a Git author name does not match a team member, add one line to ALIASES
below (git name or email in lower case -> real name). Never rewrite history.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.evaluation.predictions_io import REPO_ROOT

TEAM = ["Minahil Basalat", "Khadija Rao", "Iman Abid"]

# Fix for mismatching Git names, for example:
#   "minahilbasalat": "Minahil Basalat",
#   "someone@example.com": "Khadija Rao",
ALIASES: dict[str, str] = {
    "minahilbasalat": "Minahil Basalat",
    "minahilbasalat@gmail.com": "Minahil Basalat",
    "imannn52": "Iman Abid",
    "abidiman906@gmail.com": "Iman Abid",
    "khadija": "Khadija Rao",
    "khadijahrao135@gmail.com": "Khadija Rao",
}

MODULE_PREFIXES = [
    ("src/data/", "src/data"),
    ("src/models/", "src/models"),
    ("src/evaluation/", "src/evaluation"),
    ("tests/", "tests"),
    ("docs/", "docs"),
    ("report/", "report"),
    ("data scripts/", "data scripts"),
    ("data/", "data files"),
    ("config/", "config"),
    ("results/", "results"),
    ("eda/", "eda"),
    ("logs/", "logs"),
]
OTHER = "other"
MODULES = [m for _, m in MODULE_PREFIXES] + [OTHER]


def classify(path: str) -> str:
    for prefix, module in MODULE_PREFIXES:
        if path.startswith(prefix):
            return module
    return OTHER


def resolve_author(name: str, email: str) -> tuple[str, bool]:
    """Real member name for a Git author, and whether it was recognised."""
    for key in (email.strip().lower(), name.strip().lower()):
        if key in ALIASES:
            return ALIASES[key], True
    for member in TEAM:
        if name.strip().lower() == member.lower():
            return member, True
    return name.strip(), False


def read_commits(ref: str) -> list[dict]:
    fmt = "@@%H%x1f%an%x1f%ae%x1f%ad%x1f%s"
    result = subprocess.run(
        ["git", "log", ref, "--no-merges", "--no-renames", "--numstat",
         "--date=short", f"--format={fmt}"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise SystemExit(f"git log failed for '{ref}':\n{result.stderr}")
    commits, cur = [], None
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            sha, name, email, date, subject = line[2:].split("\x1f", 4)
            cur = {"sha": sha[:7], "name": name, "email": email, "date": date,
                   "subject": subject, "files": []}
            commits.append(cur)
        elif line.strip() and cur is not None:
            added, deleted, path = line.split("\t", 2)
            counted = path.endswith(".py")  # data and CSV files would inflate line counts
            cur["files"].append(
                (int(added) if added.isdigit() and counted else 0,
                 int(deleted) if deleted.isdigit() and counted else 0,
                 path)
            )
    return commits


def summarise(commits: list[dict]):
    by_module = defaultdict(lambda: {"commits": 0, "files": 0, "added": 0, "deleted": 0})
    totals = defaultdict(lambda: {"commits": 0, "added": 0, "deleted": 0})
    unmatched = {}
    commit_rows = []
    for c in commits:
        author, ok = resolve_author(c["name"], c["email"])
        if not ok:
            unmatched[(c["name"], c["email"])] = unmatched.get((c["name"], c["email"]), 0) + 1
        touched = defaultdict(lambda: [0, 0, 0])  # files, added, deleted
        for added, deleted, path in c["files"]:
            m = touched[classify(path)]
            m[0] += 1
            m[1] += added
            m[2] += deleted
        totals[author]["commits"] += 1
        for module, (files, added, deleted) in touched.items():
            row = by_module[(author, module)]
            row["commits"] += 1
            row["files"] += files
            row["added"] += added
            row["deleted"] += deleted
            totals[author]["added"] += added
            totals[author]["deleted"] += deleted
        commit_rows.append(
            {"commit": c["sha"], "date": c["date"], "author": author,
             "git_name": c["name"], "modules": ", ".join(sorted(touched)),
             "subject": c["subject"]}
        )
    rows = [
        {"author": a, "module": m, "commits": v["commits"], "files_changed": v["files"],
         "lines_added": v["added"], "lines_deleted": v["deleted"],
         "author_commits_total": totals[a]["commits"]}
        for (a, m), v in by_module.items()
    ]
    table = pd.DataFrame(rows).sort_values(
        ["author_commits_total", "author", "commits"], ascending=[False, True, False]
    )
    return table, totals, unmatched, pd.DataFrame(commit_rows)


def latex_escape(text: str) -> str:
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(ch, ch) for ch in text)


def latex_fragment(table: pd.DataFrame, totals: dict, ref: str) -> str:
    authors = list(totals) + [m for m in TEAM if m not in totals]
    authors.sort(key=lambda a: -totals[a]["commits"] if a in totals else 1)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Git contribution per author on \texttt{%s}: commits that changed each "
        r"module (a commit touching several modules counts for each; merge commits excluded).}"
        % latex_escape(ref),
        r"\label{tab:p4-contribution}",
        r"\begin{tabular}{|l|r|r|p{7.2cm}|}",
        r"\hline",
        r"Member & Commits & Python lines added & Modules changed (commits) \\",
        r"\hline",
    ]
    for author in authors:
        sub = table[table["author"] == author].sort_values("commits", ascending=False)
        if sub.empty:
            mods = "no commits found"
            commits, added = 0, 0
        else:
            mods = ", ".join(
                r"\texttt{%s} (%d)" % (latex_escape(r.module), r.commits)
                for r in sub.itertuples()
            )
            commits, added = totals[author]["commits"], totals[author]["added"]
        lines.append(f"{latex_escape(author)} & {commits} & {added} & {mods} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="main", help="branch to analyse (default: main)")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    head = subprocess.run(["git", "rev-parse", "--short", args.ref], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()
    commits = read_commits(args.ref)
    print(f"ref: {args.ref} ({head}) | {len(commits)} non-merge commits")

    table, totals, unmatched, commit_log = summarise(commits)
    wide = table.pivot_table(index="author", columns="module", values="commits",
                             fill_value=0, aggfunc="sum")
    wide = wide.reindex(columns=MODULES, fill_value=0)
    wide.insert(0, "commits_total", [totals[a]["commits"] for a in wide.index])
    print("\ncommits per author and module (a commit counts once per module it touches):")
    print(wide.sort_values("commits_total", ascending=False).to_string())

    missing = [m for m in TEAM if m not in totals]
    if unmatched:
        print("\nWARNING: Git authors NOT matched to a team member:")
        for (name, email), n in unmatched.items():
            print(f"  name='{name}' email='{email}' ({n} commits)")
        print("  -> add a line to ALIASES in this script, e.g. "
              '"git name or email in lower case": "Real Name". Do NOT rewrite history.')
    if missing:
        print(f"\nWARNING: no commits found for: {missing} "
              "(not merged yet, or the Git name does not match; see ALIASES).")

    table.to_csv(out_dir / "contribution_table.csv", index=False)
    commit_log.to_csv(out_dir / "contribution_commits.csv", index=False)
    tex = latex_fragment(table, totals, args.ref)
    (out_dir / "contribution_table_fragment.tex").write_text(tex + "\n", encoding="utf-8")
    for name in ["contribution_table.csv", "contribution_commits.csv",
                 "contribution_table_fragment.tex"]:
        print(f"wrote {(out_dir / name).relative_to(REPO_ROOT)}")
    print("\nLaTeX fragment (paste into the report section; never commit a .tex file):\n")
    print(tex)


if __name__ == "__main__":
    main()