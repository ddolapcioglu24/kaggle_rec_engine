from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "personalized_recommendation_dataset.csv"
)


# Recommendation settings
TOP_K = 10

# Number of similar items considered for each seen item
KNN_NEIGHBORS = 20

# Interactions before this date are used for training
SPLIT_DATE = "2024-10-01"

# Ensures reproducible random operations
RANDOM_SEED = 42

# Negative sampling ratios tested for LightGBM
NEGATIVE_SAMPLING_RATIOS = [1, 3, 5, 10]

# Negative sampling ratio selected for the final LightGBM model
FINAL_NEGATIVE_SAMPLING_RATIO = 10

# LightGBM model settings
LIGHTGBM_N_ESTIMATORS = 100
LIGHTGBM_LEARNING_RATE = 0.05
LIGHTGBM_NUM_LEAVES = 31

