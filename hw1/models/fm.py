"""Second-order factorization machine, implemented with NumPy and SciPy.

The binary training label is ``rating >= 4``.  Besides user/item IDs, the
metadata model uses historical user/item activity, mean rating, positive-rating
fraction, recency, tenure, and the calendar month of the recommendation query.
These are attributes of users, objects, and timestamped interactions, rather
than properties of the interaction being predicted.  Every training feature is
computed from events with timestamp STRICTLY smaller than that training row's
timestamp; simultaneous reviews cannot reveal one another's ratings.  At
inference, all features use the fit-history snapshot and the supplied fit
cutoff, never the timestamp or rating of a held-out event.

This deliberately avoids Amazon's crawl-time product metadata: historical
availability of the product-page rating aggregates, title, category, or price
is not established.  Missing-history values have their own ordinary categorical
feature IDs; feature ID zero is valid, not a padding index.

Uniformly sampled unknown items are weak negatives, not observed dislikes.
Sampling uses only items introduced before the query timestamp and excludes
the user's already observed prefix, regardless of rating, and simultaneous
observed pairs.  A later positive TRAIN interaction may therefore have been an
unknown negative earlier: avoiding it would require looking into the future.
No validation/test pairs participate in sampling.

The optimizer is mini-batch AdaGrad written here.  ``regularization`` penalizes
the mean squared active parameters per example, allowing exact sparse updates.
``score`` returns logits for every item and leaves seen-item masking to the
common evaluator.  A decomposition into user/context and item feature blocks
makes full-catalog inference a small matrix multiplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit


_DAY_MS = 86_400_000.0
_COUNT_EDGES = np.array([1, 2, 5, 10, 20, 50, 100, 250, 500])
_RATING_EDGES = np.array([1.5, 2.5, 3.5, 4.5])
_RATE_EDGES = np.array([0.2, 0.4, 0.6, 0.8])
_DAY_EDGES = np.array([1, 7, 30, 90, 365, 1095], dtype=np.float64)
_KEY_DTYPE = np.dtype([("entity", np.int64), ("timestamp", np.int64)])


class _HistoryIndex:
    """Compact entity timelines supporting vectorized strict-prefix queries."""

    def __init__(self, frame: pd.DataFrame, entity_column: str):
        data = frame[[entity_column, "timestamp", "rating"]].copy()
        data["positive"] = (data["rating"] >= 4).astype(np.int64)
        groups = data.groupby([entity_column, "timestamp"], sort=True).agg(
            count=("rating", "size"),
            rating_sum=("rating", "sum"),
            positive_sum=("positive", "sum"),
        ).reset_index()
        cumulative = groups.groupby(entity_column, sort=False)[
            ["count", "rating_sum", "positive_sum"]
        ].cumsum()
        self.entity = groups[entity_column].to_numpy(dtype=np.int64)
        self.timestamp = groups["timestamp"].to_numpy(dtype=np.int64)
        self.count = cumulative["count"].to_numpy(dtype=np.int64)
        self.rating_sum = cumulative["rating_sum"].to_numpy(dtype=np.float64)
        self.positive_sum = cumulative["positive_sum"].to_numpy(dtype=np.int64)
        self.first = groups.groupby(entity_column, sort=False)["timestamp"].transform(
            "first"
        ).to_numpy(dtype=np.int64)
        self.keys = np.empty(len(groups), dtype=_KEY_DTYPE)
        self.keys["entity"] = self.entity
        self.keys["timestamp"] = self.timestamp

    def query(self, entities: np.ndarray, timestamps: np.ndarray) -> dict:
        """Return history strictly before each query, including tied timestamps."""
        entities = np.asarray(entities, dtype=np.int64)
        timestamps = np.broadcast_to(np.asarray(timestamps, dtype=np.int64), entities.shape)
        keys = np.empty(len(entities), dtype=_KEY_DTYPE)
        keys["entity"], keys["timestamp"] = entities, timestamps
        positions = np.searchsorted(self.keys, keys, side="left") - 1
        safe_positions = np.maximum(positions, 0)
        known = (positions >= 0) & (self.entity[safe_positions] == entities)
        count = np.where(known, self.count[safe_positions], 0)
        denominator = np.maximum(count, 1)
        return {
            "count": count,
            "mean_rating": np.where(known, self.rating_sum[safe_positions] / denominator, 0),
            "positive_fraction": np.where(
                known, self.positive_sum[safe_positions] / denominator, 0
            ),
            "recency_days": np.where(
                known, (timestamps - self.timestamp[safe_positions]) / _DAY_MS, 0
            ),
            "tenure_days": np.where(
                known, (timestamps - self.first[safe_positions]) / _DAY_MS, 0
            ),
        }


class _CausalFeatureEncoder:
    """One-hot fields with fixed, documented bins; no fitted future vocabulary."""

    def __init__(self, frame: pd.DataFrame, n_users: int, n_items: int, metadata: bool):
        self.n_users, self.n_items = n_users, n_items
        self.metadata = metadata
        self.user_history = _HistoryIndex(frame, "user_idx") if metadata else None
        self.item_history = _HistoryIndex(frame, "item_idx") if metadata else None
        self.schema = {"user_id": (0, n_users), "item_id": (n_users, n_items)}
        cursor = n_users + n_items
        if metadata:
            for prefix in ("user", "item"):
                for name, size in (
                    ("activity", 10), ("mean_rating", 6), ("positive_fraction", 6),
                    ("recency_days", 8), ("tenure_days", 8),
                ):
                    self.schema[f"{prefix}_{name}"] = (cursor, size)
                    cursor += size
            self.schema["query_month"] = (cursor, 12)
            cursor += 12
        self.n_features = cursor
        self.user_fields = ["user_id"]
        self.item_fields = ["item_id"]
        if metadata:
            self.user_fields += [name for name in self.schema if name.startswith("user_") and name != "user_id"]
            self.user_fields += ["query_month"]
            self.item_fields += [name for name in self.schema if name.startswith("item_") and name != "item_id"]

    def _entity_block(self, entities, timestamps, prefix):
        entities = np.asarray(entities, dtype=np.int64)
        timestamps = np.broadcast_to(np.asarray(timestamps, dtype=np.int64), entities.shape)
        id_offset = 0 if prefix == "user" else self.n_users
        columns = [entities + id_offset]
        if self.metadata:
            history = self.user_history if prefix == "user" else self.item_history
            stats = history.query(entities, timestamps)
            known = stats["count"] > 0
            bins = {
                "activity": np.searchsorted(_COUNT_EDGES, stats["count"], side="right"),
                "mean_rating": np.where(known, 1 + np.searchsorted(_RATING_EDGES, stats["mean_rating"], side="right"), 0),
                "positive_fraction": np.where(known, 1 + np.searchsorted(_RATE_EDGES, stats["positive_fraction"], side="right"), 0),
                "recency_days": np.where(known, 1 + np.searchsorted(_DAY_EDGES, stats["recency_days"], side="right"), 0),
                "tenure_days": np.where(known, 1 + np.searchsorted(_DAY_EDGES, stats["tenure_days"], side="right"), 0),
            }
            for name, values in bins.items():
                columns.append(self.schema[f"{prefix}_{name}"][0] + values)
            if prefix == "user":
                month = timestamps.astype("datetime64[ms]").astype("datetime64[M]").astype(np.int64) % 12
                columns.append(self.schema["query_month"][0] + month)
        return np.column_stack(columns).astype(np.int32, copy=False)

    def user_block(self, users, timestamps):
        return self._entity_block(users, timestamps, "user")

    def item_block(self, items, timestamps):
        return self._entity_block(items, timestamps, "item")

    def encode_pairs(self, users, items, timestamps):
        return np.concatenate(
            [self.user_block(users, timestamps), self.item_block(items, timestamps)], axis=1
        )


def _batch_loss_and_gradient(feature_ids, labels, bias, weights, factors, regularization=0.0):
    """FM logistic loss and sparse gradients for one mini-batch.

    Returns loss, scalar bias gradient, active feature IDs, and the corresponding
    linear/factor gradients.  Each field contributes a unit-valued one-hot feature.
    """
    batch_size, n_fields = feature_ids.shape
    active, inverse = np.unique(feature_ids, return_inverse=True)
    inverse = inverse.reshape(feature_ids.shape)
    embeddings = factors[feature_ids]
    summed = embeddings.sum(axis=1)
    logits = bias + weights[feature_ids].sum(axis=1) + 0.5 * (
        np.square(summed).sum(axis=1) - np.square(embeddings).sum(axis=(1, 2))
    )
    delta = (expit(logits) - labels) / batch_size
    local_features = sparse.csr_matrix(
        (np.ones(feature_ids.size, dtype=factors.dtype), inverse.ravel(),
         np.arange(0, feature_ids.size + 1, n_fields)),
        shape=(batch_size, len(active)),
    )
    linear_gradient = np.asarray(local_features.T @ delta).ravel()
    factor_gradient = np.asarray(local_features.T @ (delta[:, None] * summed))
    factor_gradient -= linear_gradient[:, None] * factors[active]
    frequency = np.bincount(inverse.ravel(), minlength=len(active)).astype(factors.dtype) / batch_size
    penalty = 0.5 * regularization * np.sum(
        frequency * (np.square(weights[active]) + np.square(factors[active]).sum(axis=1))
    )
    linear_gradient += regularization * frequency * weights[active]
    factor_gradient += regularization * frequency[:, None] * factors[active]
    loss = np.mean(np.logaddexp(0, logits) - labels * logits) + penalty
    return float(loss), float(delta.sum()), active, linear_gradient, factor_gradient


class FactorizationMachine:
    """Order-2 FM for personalized binary-relevance ranking.

    ``use_metadata=False`` keeps the
    identical learner and causal negative samples but removes every field except
    user/item IDs, providing an IDs-only ablation of historical/time metadata.
    The returned logits rank items; they are not calibrated purchase probabilities.
    """

    def __init__(self, factors=12, learning_rate=0.03, regularization=1e-4,
                 epochs=6, batch_size=2048, negative_ratio=2, seed=42,
                 use_metadata=True):
        if factors < 1 or epochs < 1 or batch_size < 1:
            raise ValueError("factors, epochs, and batch_size must be positive")
        if negative_ratio < 0 or int(negative_ratio) != negative_ratio:
            raise ValueError("negative_ratio must be a nonnegative integer")
        if learning_rate <= 0 or regularization < 0:
            raise ValueError("learning_rate must be positive and regularization nonnegative")
        self.factors = int(factors)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.negative_ratio = int(negative_ratio)
        self.seed = int(seed)
        self.use_metadata = bool(use_metadata)

    def _sample_training_pairs(self, frame, rng):
        """Observed labels plus prefix-causal uniform unknown negatives."""
        users = frame["user_idx"].to_numpy(dtype=np.int64)
        items = frame["item_idx"].to_numpy(dtype=np.int64)
        timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
        labels = (frame["rating"].to_numpy() >= 4).astype(np.float32)
        capacity = int(labels.sum()) * self.negative_ratio
        negative_users = np.empty(capacity, dtype=np.int64)
        negative_items = np.empty(capacity, dtype=np.int64)
        negative_times = np.empty(capacity, dtype=np.int64)
        seen = [set() for _ in range(self.n_users_)]
        active_items = []
        introduced = np.zeros(self.n_items_, dtype=bool)
        count = 0
        boundaries = np.r_[0, np.flatnonzero(np.diff(timestamps)) + 1, len(frame)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            simultaneous = {}
            for position in range(start, end):
                simultaneous.setdefault(int(users[position]), set()).add(int(items[position]))
            if active_items and self.negative_ratio:
                for position in range(start, end):
                    if not labels[position]:
                        continue
                    user = int(users[position])
                    blocked = seen[user] | simultaneous[user]
                    selected = set()
                    attempts = 0
                    # Rejection sampling is cheap for the sparse catalog.  A
                    # bounded exact fallback also handles tiny/dense fixtures.
                    while len(selected) < self.negative_ratio and attempts < 20 * self.negative_ratio + 20:
                        candidate = active_items[int(rng.integers(len(active_items)))]
                        if candidate not in blocked:
                            selected.add(candidate)
                        attempts += 1
                    if len(selected) < self.negative_ratio:
                        eligible = [item for item in active_items if item not in blocked and item not in selected]
                        need = min(self.negative_ratio - len(selected), len(eligible))
                        if need:
                            selected.update(rng.choice(eligible, size=need, replace=False).tolist())
                    # Sorting avoids Python set iteration affecting reproducibility.
                    for candidate in sorted(selected):
                        negative_users[count], negative_items[count] = user, candidate
                        negative_times[count] = timestamps[position]
                        count += 1
            for position in range(start, end):
                user, item = int(users[position]), int(items[position])
                seen[user].add(item)
                if not introduced[item]:
                    active_items.append(item)
                    introduced[item] = True
        return (
            np.r_[users, negative_users[:count]],
            np.r_[items, negative_items[:count]],
            np.r_[timestamps, negative_times[:count]],
            np.r_[labels, np.zeros(count, dtype=np.float32)],
            count,
        )

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int, cutoff_ms: int):
        """Fit using only rows strictly before ``cutoff_ms`` (UTC milliseconds)."""
        required = ["user_idx", "item_idx", "rating", "timestamp"]
        if not set(required).issubset(train.columns) or train.empty:
            raise ValueError("train must be nonempty and contain " + ", ".join(required))
        frame = train[required].copy()
        if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
            raise ValueError("training values must be finite and nonmissing")
        self.n_users_, self.n_items_ = int(n_users), int(n_items)
        self.cutoff_ms_ = int(cutoff_ms)
        if self.n_users_ < 1 or self.n_items_ < 1:
            raise ValueError("n_users and n_items must be positive")
        for column, upper in (("user_idx", self.n_users_), ("item_idx", self.n_items_)):
            values = frame[column].to_numpy()
            if (values < 0).any() or (values >= upper).any() or (values != values.astype(np.int64)).any():
                raise ValueError(f"{column} must contain integer indices in [0, {upper})")
            frame[column] = values.astype(np.int64)
        timestamps = frame["timestamp"].to_numpy()
        if (timestamps != timestamps.astype(np.int64)).any() or (timestamps >= self.cutoff_ms_).any():
            raise ValueError("timestamps must be integer milliseconds strictly before cutoff_ms")
        frame["timestamp"] = timestamps.astype(np.int64)
        frame = frame.sort_values(["timestamp", "user_idx", "item_idx"], kind="mergesort").reset_index(drop=True)
        # Separate streams keep negative samples and minibatch ordering identical
        # for the metadata model and its IDs-only ablation.
        sampling_rng, initialization_rng, shuffle_rng = [
            np.random.default_rng(child) for child in np.random.SeedSequence(self.seed).spawn(3)
        ]
        self.feature_encoder_ = _CausalFeatureEncoder(frame, self.n_users_, self.n_items_, self.use_metadata)
        self.feature_schema_ = self.feature_encoder_.schema.copy()
        users, items, times, labels, n_negative = self._sample_training_pairs(frame, sampling_rng)
        feature_ids = self.feature_encoder_.encode_pairs(users, items, times)
        n_features = self.feature_encoder_.n_features
        self.w_ = np.zeros(n_features, dtype=np.float32)
        self.V_ = initialization_rng.normal(0, 0.01, size=(n_features, self.factors)).astype(np.float32)
        fraction = np.clip(labels.mean(), 1e-5, 1 - 1e-5)
        self.bias_ = float(np.log(fraction / (1 - fraction)))
        accumulated_w = np.zeros_like(self.w_)
        accumulated_v = np.zeros_like(self.V_)
        accumulated_bias = 0.0
        self.loss_history_ = []
        for _ in range(self.epochs):
            order = shuffle_rng.permutation(len(labels))
            total_loss = 0.0
            for start in range(0, len(labels), self.batch_size):
                indices = order[start:start + self.batch_size]
                loss, db, active, dw, dv = _batch_loss_and_gradient(
                    feature_ids[indices], labels[indices], self.bias_, self.w_, self.V_, self.regularization
                )
                accumulated_w[active] += np.square(dw)
                accumulated_v[active] += np.square(dv)
                accumulated_bias += db * db
                self.w_[active] -= self.learning_rate * dw / (np.sqrt(accumulated_w[active]) + 1e-8)
                self.V_[active] -= self.learning_rate * dv / (np.sqrt(accumulated_v[active]) + 1e-8)
                self.bias_ -= self.learning_rate * db / (np.sqrt(accumulated_bias) + 1e-8)
                total_loss += loss * len(indices)
            self.loss_history_.append(total_loss / len(labels))
        self.training_summary_ = {
            "observed_rows": len(frame), "observed_positive_rows": int((frame["rating"] >= 4).sum()),
            "sampled_negative_rows": n_negative, "training_rows": len(labels),
            "n_features": n_features, "active_fields_per_row": feature_ids.shape[1],
            "negative_policy": "uniform introduced items; exclude observed prefix and timestamp ties",
        }
        user_fields = self.feature_encoder_.user_block(np.arange(self.n_users_), self.cutoff_ms_)
        item_fields = self.feature_encoder_.item_block(np.arange(self.n_items_), self.cutoff_ms_)
        self._user_sum_, self._user_offset_ = self._project_block(user_fields)
        self._item_sum_, self._item_offset_ = self._project_block(item_fields)
        return self

    def _project_block(self, feature_ids):
        embeddings = self.V_[feature_ids]
        summed = embeddings.sum(axis=1)
        offset = self.w_[feature_ids].sum(axis=1) + 0.5 * (
            np.square(summed).sum(axis=1) - np.square(embeddings).sum(axis=(1, 2))
        )
        return summed, offset

    def score(self, users: np.ndarray) -> np.ndarray:
        """Return ``(len(users), n_items)`` raw FM logits without seen masking.

        Only the returned float32 matrix scales with users times items.  For
        1,500 users and 4,000 candidates it occupies approximately 24 MB.
        """
        if not hasattr(self, "V_"):
            raise RuntimeError("fit must be called before score")
        users = np.atleast_1d(np.asarray(users))
        if users.ndim != 1 or (users != users.astype(np.int64)).any():
            raise ValueError("users must be a one-dimensional array of integer indices")
        users = users.astype(np.int64)
        if (users < 0).any() or (users >= self.n_users_).any():
            raise ValueError("user index outside fitted user universe")
        scores = self._user_sum_[users] @ self._item_sum_.T
        scores += self._user_offset_[users, None]
        scores += self._item_offset_[None, :]
        scores += self.bias_
        return scores
