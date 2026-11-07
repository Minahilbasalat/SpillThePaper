# Data-side settings (Phase 3, data and labels)

| Setting | Values tried | Final | Why |
|---|---|---|---|
| Min train papers per subject | 10, 6 | 6 | 10 left 26 eligible subjects; 6 gave 28 while keeping topics meaningful |
| Min train questions per subject | 40, 30 | 30 | needed for stable clustering |
| Min train years per subject | 3 | 3 | recency features need at least three years of history |
| Topics per subject (K) | 4 to 10, chosen per subject by silhouette | 4-10 (median 9) | no single K suits all subjects |
| TF-IDF | unigrams + bigrams, 2000 features, min_df 2 | same | short questions need bigrams (for example "circuit switching") |
| Stop words | English + exam words, then + header words | both lists | header words (course, section, fast, nu) formed junk clusters |
| Min prior papers / years for a target exam | 3 / 2 | 3 / 2 | otherwise history features are empty |
| Recent window | 2 years | 2 | syllabi change; older appearances matter less |
| Split | train 2014-2022, val 2023, test 2024-2026 | same | time-based; no future papers in training |

Notes:
- The eligible-subject count (26 to 28) and topic count (212 to 220) changed between runs because
  the header-chunk fix and the threshold change were applied together.
- Silhouette scores are low (median 0.097, max 0.188). Short exam questions overlap heavily in
  vocabulary, so topics are approximate and some clusters are instruction text.
- Topics are fitted on train-period questions only; later questions are assigned to the nearest topic.