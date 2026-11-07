"""Shared constants for the supervised dataset (see docs/dataset_schema.md).

Every module (data, models, evaluation) imports column names from here so that
the three parts of the project stay consistent.
"""

ID_COLUMNS = ["subject", "topic_id", "topic_label", "target_year", "target_type"]

NUMERIC_FEATURES = [
    "prior_papers",
    "prior_years",
    "topic_count_prior",
    "topic_rate_prior",
    "years_since_last",   # NaN when the topic was never seen: impute on TRAIN only
    "appeared_last_year",
    "recent_rate",
    "type_rate_prior",
]
CATEGORICAL_FEATURES = ["target_type"]  # quiz / mid / final
TARGET = "label"

ALL_COLUMNS = ID_COLUMNS + NUMERIC_FEATURES + [TARGET]

# Time-based split (adjusted in build step if counts are too small)
TRAIN_MAX_YEAR = 2022
VAL_YEARS = [2023]
TEST_MIN_YEAR = 2024

MODEL_DATA_DIR = "data/processed/model"
DUMMY_DATA_DIR = "data/processed/model_dummy"