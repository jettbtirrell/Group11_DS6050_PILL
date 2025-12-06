import numpy as np
import torch
from collections import Counter
import time

from train_knn_retrieval import train_knn_retrieval


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_knn(knn, feature_model, test_loader, label_encoder):
    # We wanted to look at the accuracy metric as a way to initially identify
    # how well this model performed.
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feats = feature_model(imgs).cpu().numpy()
            preds = knn.predict(feats)
            true_names = label_encoder.inverse_transform(labels.numpy())
            correct += np.sum(preds == true_names)
            total += len(true_names)

    return correct / total


def evaluate_knn_metrics(knn, feature_model, test_loader, label_encoder, pill_names, topk_list=[1, 5, 10]):
    # Very similar to the notebook code — calculating recall@k, MRR@k, and time
    y = np.array(pill_names)

    recall_totals = {k: 0 for k in topk_list}
    mrr_totals = {k: 0 for k in topk_list}
    total_images = 0
    times = []

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.numpy()
        true_names = label_encoder.inverse_transform(labels)

        with torch.no_grad():
            start_time = time.time()
            feats = feature_model(imgs).cpu().numpy()
            end_time = time.time()

            times.append((end_time - start_time) / len(imgs))

            for k in topk_list:
                distances, indices = knn.kneighbors(feats, n_neighbors=k)
                topk_preds = y[indices]

                for i in range(len(true_names)):
                    total_images += 1
                    # Recall
                    if true_names[i] in topk_preds[i]:
                        recall_totals[k] += 1
                        # MRR
                        rank = np.where(topk_preds[i] == true_names[i])[0][0] + 1
                        mrr_totals[k] += 1 / rank
                    else:
                        mrr_totals[k] += 0

    avg_recall = {k: recall_totals[k] / total_images for k in topk_list}
    avg_mrr = {k: mrr_totals[k] / total_images for k in topk_list}
    avg_time = np.mean(times)

    return avg_recall, avg_mrr, avg_time


def get_knn_predictions(knn, feature_model, test_loader, label_encoder, pill_names, k=5):
    # Part of the project was to perform an error analysis
    # Similar to notebook: show true label, predicted, top-5 for wrong examples
    y = np.array(pill_names)
    results = []

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            feats = feature_model(imgs).cpu().numpy()
            top1_preds = knn.predict(feats)
            distances, indices = knn.kneighbors(feats, n_neighbors=k)
            topk_preds = y[indices]

            true_names = label_encoder.inverse_transform(labels.numpy())

            for i in range(len(true_names)):
                results.append({
                    "true": true_names[i],
                    "top1": top1_preds[i],
                    "topk": topk_preds[i].tolist(),
                })

    return results


def run_full_retrieval_evaluation():
    """
    Runs:
      1. KNN training (imported)
      2. Recall@k, MRR@k, inference speed
      3. Error analysis with top-5 neighbors

    Returns all objects so the notebook or scripts can print tables / plots.
    """

    # Load trained KNN + feature extractor + label encoder
    knn, feature_model, le, pill_names, test_loader = train_knn_retrieval()

    # 1. Test accuracy
    accuracy = evaluate_knn(knn, feature_model, test_loader, le)

    # 2. Recall@k, MRR@k, timing
    recalls, mrrs, avg_time = evaluate_knn_metrics(
        knn, feature_model, test_loader, le,
        pill_names,
        topk_list=[1, 5, 10]
    )

    # 3. Error analysis
    results = get_knn_predictions(knn, feature_model, test_loader, le, pill_names, k=5)
    errors = [r for r in results if r["true"] != r["top1"]]

    return {
        "accuracy": accuracy,
        "recall": recalls,
        "mrr": mrrs,
        "avg_time": avg_time,
        "results": results,
        "errors": errors,
    }


if __name__ == "__main__":
    out = run_full_retrieval_evaluation()

    print("\n=== KNN Retrieval Evaluation ===")
    print("Accuracy:", out["accuracy"])
    print("Recall:", out["recall"])
    print("MRR:", out["mrr"])
    print("Average time:", out["avg_time"])
    print("Total results:", len(out["results"]))
    print("Errors:", len(out["errors"]))
