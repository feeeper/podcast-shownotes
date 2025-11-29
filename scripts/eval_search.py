import json
from pathlib import Path
import numpy as np
from sklearn.metrics import ndcg_score


def precision_at_k(retrieved: list[str], relevant: set[tuple[int, int]], k: int) -> float:
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[tuple[int, int]], k: int) -> float:
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & relevant) / len(relevant) if relevant else 0


def average_precision(retrieved: list[str], relevant: set[tuple[int, int]]) -> float:
    score = 0.0
    num_hits = 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            num_hits += 1
            score += num_hits / i
    return score / len(relevant) if relevant else 0


def dcg(retrieved: list[str], relevant: set[tuple[int, int]], k: int) -> float:
    dcg_val = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            dcg_val += 1 / np.log2(i + 1)  # log base 2
    return dcg_val


def ndcg_at_k(retrieved: list[str], relevant: set[tuple[int, int]], k: int) -> float:
    # FIXED: Create ideal ranking by putting all relevant docs first
    ideal_ranking = list(relevant) + [doc for doc in retrieved if doc not in relevant]
    ideal = dcg(ideal_ranking, relevant, k)
    return dcg(retrieved, relevant, k) / ideal if ideal > 0 else 0


def main() -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser("Eval search results")
    parser.add_argument("--expected", type=Path, required=True, help="Path to the ground truth")
    parser.add_argument("--actual", type=Path, required=True, help="Path to the actual results")
    parser.add_argument("--k", type=int, help="K for NDCG@K, Precision@K, and Recall@K", default=5)

    args = parser.parse_args()
    
    ground_truth = json.loads(args.expected.read_text())
    actual = json.loads(args.actual.read_text())
    
    K = args.k
    precisions, recalls, maps, ndcgs = [], [], [], []

    # Fix: Extract documents correctly from both ground truth and actual results
    all_docs = []
    for docs in actual.values():
        for doc in docs:
            all_docs.append((doc["episode"], doc["segment"]))
    
    for docs in ground_truth.values():
        for doc in docs:
            all_docs.append((doc["episode"], doc["segment"]))
    
    all_docs = set(all_docs)
    doc_index = {doc: i for i, doc in enumerate(all_docs)}
    y_true, y_score = [], []

    for q in ground_truth:
        # Fix: Extract relevant documents correctly from ground truth
        relevant = {(doc["episode"], doc["segment"]) for doc in ground_truth[q]}
        retrieved_docs = [(doc["episode"], doc["segment"]) for doc in actual[q]]
        
        precisions.append(precision_at_k(retrieved_docs, relevant, K))
        recalls.append(recall_at_k(retrieved_docs, relevant, K))
        maps.append(average_precision(retrieved_docs, relevant))
        ndcgs.append(ndcg_at_k(retrieved_docs, relevant, K))

        # True relevance (binary: 1 for relevant doc)
        true_vec = np.zeros(len(all_docs))
        for r in relevant:
            true_vec[doc_index[r]] = 1
        y_true.append(true_vec)
        
        # Scores from search engine
        score_vec = np.zeros(len(all_docs))
        for doc in actual[q]:
            ep = doc["episode"]
            seg = doc["segment"]
            score = doc["score"]
            score_vec[doc_index[(ep, seg)]] = score
        y_score.append(score_vec)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    print("NDCG@3:", ndcg_score(y_true, y_score, k=3))
    print("NDCG@5:", ndcg_score(y_true, y_score, k=5))

    print(f"Precision@{K}: {np.mean(precisions):.5f}")
    print(f"Recall@{K}: {np.mean(recalls):.5f}")
    print(f"MAP: {np.mean(maps):.5f}")
    print(f"NDCG@{K}: {np.mean(ndcgs):.5f}")
    # ndcg_score()


if __name__ == "__main__":
    main()


# # Example ground truth (which segments are relevant per query)
# ground_truth = {
#     "q1": {(1, 2), (1, 3)},        # Query 1: d1 and d3 are relevant
#     "q2": {(1, 2)},              # Query 2: d2 is relevant
#     "q3": {(1, 3), (1, 4), (1, 5)},  # Query 3: d3, d4, d5 are relevant
# }

# # Example retrieved ranked lists (your search engine output)
# retrieved = {
#     "q1": [(1, 2), (1, 2), (1, 3), (1, 4)],
#     "q2": [(1, 3), (1, 2), (1, 5)],
#     "q3": [(1, 5), (1, 2), (1, 4), (1, 3)],
# }


# for q in ground_truth:
#     relevant = ground_truth[q]
#     retrieved_docs = retrieved[q]
    
#     precisions.append(precision_at_k(retrieved_docs, relevant, K))
#     recalls.append(recall_at_k(retrieved_docs, relevant, K))
#     maps.append(average_precision(retrieved_docs, relevant))
#     ndcgs.append(ndcg_at_k(retrieved_docs, relevant, K))

# print(f"Precision@{K}: {np.mean(precisions):.3f}")
# print(f"Recall@{K}: {np.mean(recalls):.3f}")
# print(f"MAP: {np.mean(maps):.3f}")
# print(f"NDCG@{K}: {np.mean(ndcgs):.3f}")
