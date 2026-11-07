# Dataset schema (Phase 3 and 4)

Task: given (subject, topic, target_year, target_type), predict whether the topic
appears in that exam. Features use ONLY papers with year < target_year.

| Column | Type | Meaning |
|---|---|---|
| subject | str | cleaned subject name |
| topic_id | int | topic cluster id within the subject |
| topic_label | str | top keywords of the topic |
| target_year | int | year of the exam being predicted |
| target_type | str | quiz / mid / final |
| prior_papers | int | papers of this subject before target_year |
| prior_years | int | distinct years of this subject before target_year |
| topic_count_prior | int | times the topic appeared before |
| topic_rate_prior | float | topic_count_prior / prior_papers |
| years_since_last | float | years since topic last appeared (NaN if never) |
| appeared_last_year | int | 1 if topic appeared in target_year - 1 |
| recent_rate | float | appearance rate over the last 2 years |
| type_rate_prior | float | appearance rate within the same assessment type |
| label | int | 1 if topic appears in the target exam, else 0 |

Files (in data/model/): train.csv, val.csv, test.csv, split by target_year
(train <= 2022, val = 2023, test >= 2024; may be adjusted after checking counts).