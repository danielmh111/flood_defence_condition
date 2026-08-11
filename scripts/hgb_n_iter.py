import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from experiments.oof_grid.runner import PARQUET, load_set

_, X, y, cat_idx, *_ = load_set(PARQUET)

model = HistGradientBoostingClassifier(
    categorical_features=list(cat_idx),
    learning_rate=0.05,
    max_iter=2000,
    early_stopping=True,
    random_state=123,
    class_weight=None,
).fit(X, y)

print(model.n_iter_)
