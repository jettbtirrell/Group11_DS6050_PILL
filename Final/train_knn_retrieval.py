# train_knn_retrieval.py

import numpy as np
import torch
import time
from sklearn.neighbors import KNeighborsClassifier

from train_resnet18_classifier import train_pretrained_resnet


def train_knn_retrieval():
    (
        model,
        feature_model,
        features_dict,
        df_nlm,
        le,
        num_classes,
        test_loader
    ) = train_pretrained_resnet()

    pill_names = list(features_dict.keys())
    X = []
    y = []

    for pill_name in pill_names:
        embed_tensor = features_dict[pill_name]
        flat_vec = embed_tensor.flatten().cpu().numpy()
        X.append(flat_vec)
        y.append(pill_name)

    X = np.array(X)
    y = np.array(y)

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine"
    )
    knn.fit(X, y)

    return knn, feature_model, le, y, test_loader


def evaluate_knn(knn, feature_model, test_loader, label_encoder):
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to("cuda" if torch.cuda.is_available() else "cpu")
            feats = feature_model(imgs).cpu().numpy()
            preds = knn.predict(feats)
            true_names = label_encoder.inverse_transform(labels.numpy())
            correct += np.sum(preds == true_names)
            total += len(true_names)

    return correct / total


def evaluate_knn_metrics(knn, feature_model, test_loader, label_encoder, pill_names, topk_list=[1,5,10]):
    y = np.array(pill_names)

    recall_totals = {k: 0 for k in topk_list}
    mrr_totals = {k: 0 for k in topk_list}
    total_images = 0
    times = []

    for imgs, labels in test_loader:
        imgs = imgs.to("cuda" if torch.cuda.is_available() else "cpu")
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
                    if true_names[i] in topk_preds[i]:
                        recall_totals[k] += 1
                        rank = np.where(topk_preds[i] == true_names[i])[0][0] + 1
                        mrr_totals[k] += 1 / rank
                    else:
                        mrr_totals[k] += 0

    avg_recall = {k: recall_totals[k] / total_images for k in topk_list}
    avg_mrr    = {k: mrr_totals[k] / total_images for k in topk_list}
    avg_time   = np.mean(times)

    return avg_recall, avg_mrr, avg_time
