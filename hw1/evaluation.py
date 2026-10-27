"""Shared, deterministic full-catalog top-K evaluation written from scratch."""

import numpy as np
from scipy import sparse

from config import K, RELEVANT_RATING, SCORE_BATCH_SIZE


def history_matrix(frame, n_users, n_items):
    return sparse.csr_matrix((np.ones(len(frame)), (frame.user_idx, frame.item_idx)), shape=(n_users, n_items))


def top_k(scores, k=K):
    """Exact deterministic ranking: smaller item ID wins a score tie."""
    scores = np.asarray(scores)
    if np.isnan(scores).any():
        raise ValueError("NaN in recommendation scores")
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def evaluate(model, history, heldout, users, n_users, n_items, k=K):
    seen = history_matrix(history, n_users, n_items)
    truth = heldout[heldout.rating >= RELEVANT_RATING].groupby("user_idx").item_idx.apply(set).to_dict()
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ndcgs, recalls, recommendations = [], [], []
    for start in range(0, len(users), SCORE_BATCH_SIZE):
        batch = np.asarray(users[start:start + SCORE_BATCH_SIZE])
        scores = np.asarray(model.score(batch), dtype=np.float64)
        if scores.shape != (len(batch), n_items) or not np.isfinite(scores).all():
            raise ValueError("Invalid model.score output")
        for row, user in enumerate(batch):
            known = seen.indices[seen.indptr[user]:seen.indptr[user + 1]]
            scores[row, known] = -np.inf
            if n_items - len(known) < k:
                raise ValueError("Not enough unseen candidates for top-K")
        ranked = top_k(scores, k)
        for user, items in zip(batch, ranked):
            relevant = truth[int(user)]
            known = set(seen.indices[seen.indptr[user]:seen.indptr[user + 1]])
            assert not (set(items) & known)
            assert not (relevant & known), "Protocol must remove repeated user-item events"
            hits = np.array([item in relevant for item in items])
            recalls.append(float(hits.sum() / len(relevant)))
            ndcgs.append(float((hits * discounts).sum() / discounts[:min(k, len(relevant))].sum()))
            recommendations.append([int(x) for x in items])
    if not ndcgs:
        raise ValueError("No eligible evaluation users")
    return {"NDCG@10": float(np.mean(ndcgs)), "Recall@10": float(np.mean(recalls)),
            "users": len(users), "per_user_ndcg": ndcgs, "per_user_recall": recalls,
            "recommendations": recommendations}
