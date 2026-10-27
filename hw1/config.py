"""Predeclared experiment choices; test scores never select these settings."""

SEED = 42
K = 10
PRIMARY_METRIC = "NDCG@10"
RELEVANT_RATING = 4
TRAIN_FRACTION = 0.80
VALIDATION_END_FRACTION = 0.90
MAX_ITEMS = 4000
MAX_USERS = 30000
MIN_TRAIN_INTERACTIONS = 5
MAX_EVALUATION_USERS = 1500
SCORE_BATCH_SIZE = 64

# Exactly three validation configurations for each of the five model families.
SEARCH_SPACES = {
    "ItemKNN": [
        {"neighbors": 30, "shrinkage": 10.0},
        {"neighbors": 80, "shrinkage": 10.0},
        {"neighbors": 80, "shrinkage": 50.0},
    ],
    "iALS": [
        {"factors": 16, "regularization": 1.0, "alpha": 10.0, "iterations": 8},
        {"factors": 24, "regularization": 5.0, "alpha": 10.0, "iterations": 8},
        {"factors": 24, "regularization": 5.0, "alpha": 30.0, "iterations": 8},
    ],
    "EASE": [
        {"regularization": 50.0},
        {"regularization": 200.0},
        {"regularization": 800.0},
    ],
    "FM": [
        {"factors": 8, "learning_rate": 0.03, "regularization": 0.0001, "epochs": 6},
        {"factors": 16, "learning_rate": 0.03, "regularization": 0.0001, "epochs": 6},
        {"factors": 16, "learning_rate": 0.01, "regularization": 0.001, "epochs": 6},
    ],
    "CustomALS": [
        {"factors": 16, "regularization": 1.0, "alpha": 10.0, "iterations": 8, "half_life_days": 365.0},
        {"factors": 24, "regularization": 5.0, "alpha": 10.0, "iterations": 8, "half_life_days": 365.0},
        {"factors": 24, "regularization": 5.0, "alpha": 30.0, "iterations": 8, "half_life_days": 365.0},
    ],
}
