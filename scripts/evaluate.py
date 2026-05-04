import numpy as np
import json
import os
from .config import class_names, N_CLASSES


def hit_at_k(y_true, y_pred_proba, k=5):
    """
    Proportion of samples where the true label is in the top-K predicted classes.
    
    Args:
        y_true: array of true label indices (shape: n_samples,)
        y_pred_proba: predicted probability matrix (shape: n_samples x n_classes)
        k: number of top predictions to consider
    
    Returns:
        float: hit rate at k
    """
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    hits = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return hits.mean()


def ndcg_at_k(y_true, y_pred_proba, k=5):
    """
    Normalized Discounted Cumulative Gain at K.
    For single-label classification: DCG = 1/log2(rank+1) if true label is in top-K, else 0.
    IDCG = 1 (best case: true label at rank 1).
    
    Args:
        y_true: array of true label indices (shape: n_samples,)
        y_pred_proba: predicted probability matrix (shape: n_samples x n_classes)
        k: number of top predictions to consider
    
    Returns:
        float: mean NDCG@K across all samples
    """
    n_samples = len(y_true)
    ndcg_scores = np.zeros(n_samples)
    
    # Get descending rank order for each sample
    sorted_indices = np.argsort(y_pred_proba, axis=1)[:, ::-1]
    
    for i in range(n_samples):
        # Find rank of true label (0-indexed)
        rank_positions = np.where(sorted_indices[i, :k] == y_true[i])[0]
        if len(rank_positions) > 0:
            rank = rank_positions[0]  # 0-indexed position
            ndcg_scores[i] = 1.0 / np.log2(rank + 2)  # +2 because rank is 0-indexed
    
    return ndcg_scores.mean()


def revenue_at_k(y_true, y_pred_proba, k=5):
    """
    Average revenue (pack denomination) of correct predictions within top-K.
    Measures whether the model correctly identifies high-value packs.
    
    Args:
        y_true: array of true label indices (shape: n_samples,)
        y_pred_proba: predicted probability matrix (shape: n_samples x n_classes)
        k: number of top predictions to consider
    
    Returns:
        dict with 'revenue_when_hit' (avg denomination when correct) and 
        'potential_revenue' (avg denomination of true labels)
    """
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    hits = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    
    # Map label indices back to denominations
    deno_values = np.array(class_names)
    true_denominations = deno_values[y_true]
    
    revenue_when_hit = true_denominations[hits].mean() if hits.sum() > 0 else 0.0
    potential_revenue = true_denominations.mean()
    
    return {
        "revenue_when_hit": float(revenue_when_hit),
        "potential_revenue": float(potential_revenue),
        "revenue_capture_rate": float(revenue_when_hit / potential_revenue) if potential_revenue > 0 else 0.0
    }


def class_distribution_report(y_true, y_pred_proba):
    """
    Analyze prediction distribution to detect class collapse.
    Warns if any class captures >20% of predictions (majority class dominance).
    
    Args:
        y_true: array of true label indices
        y_pred_proba: predicted probability matrix
    
    Returns:
        dict with distribution analysis
    """
    predicted_classes = np.argmax(y_pred_proba, axis=1)
    n_samples = len(predicted_classes)
    
    # True label distribution
    true_counts = np.bincount(y_true, minlength=N_CLASSES)
    true_pcts = true_counts / n_samples * 100
    
    # Predicted label distribution
    pred_counts = np.bincount(predicted_classes, minlength=N_CLASSES)
    pred_pcts = pred_counts / n_samples * 100
    
    # Detect dominant classes (>20% of predictions)
    dominant_classes = np.where(pred_pcts > 20)[0]
    
    # Per-class accuracy
    per_class_acc = {}
    for cls_idx in range(N_CLASSES):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            cls_correct = (predicted_classes[mask] == cls_idx).sum()
            per_class_acc[int(class_names[cls_idx])] = {
                "accuracy": float(cls_correct / mask.sum()),
                "support": int(mask.sum()),
                "pred_share_pct": float(pred_pcts[cls_idx])
            }
    
    warnings = []
    if len(dominant_classes) > 0:
        for dc in dominant_classes:
            warnings.append(
                f"CLASS COLLAPSE WARNING: Pack {class_names[dc]} captures "
                f"{pred_pcts[dc]:.1f}% of all predictions (true share: {true_pcts[dc]:.1f}%)"
            )
    
    # Classes that never get predicted
    zero_pred_classes = np.where(pred_counts == 0)[0]
    if len(zero_pred_classes) > 0:
        never_predicted = [class_names[i] for i in zero_pred_classes]
        warnings.append(f"DEAD CLASSES: These packs are never recommended: {never_predicted}")
    
    return {
        "warnings": warnings,
        "dominant_classes": [int(class_names[dc]) for dc in dominant_classes],
        "never_predicted": [int(class_names[i]) for i in zero_pred_classes],
        "per_class": per_class_acc,
        "imbalance_ratio": float(true_counts.max() / max(true_counts[true_counts > 0].min(), 1))
    }


def evaluate_model(y_true, y_pred_proba, model_name="model"):
    """
    Run full evaluation suite and print results.
    
    Args:
        y_true: array of true label indices
        y_pred_proba: predicted probability matrix
        model_name: identifier for this model (e.g. 'taker', 'non_taker')
    
    Returns:
        dict: all metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*60}")
    print(f"Test samples: {len(y_true)}")
    
    # Core ranking metrics
    h1 = hit_at_k(y_true, y_pred_proba, k=1)
    h3 = hit_at_k(y_true, y_pred_proba, k=3)
    h5 = hit_at_k(y_true, y_pred_proba, k=5)
    
    n3 = ndcg_at_k(y_true, y_pred_proba, k=3)
    n5 = ndcg_at_k(y_true, y_pred_proba, k=5)
    
    rev = revenue_at_k(y_true, y_pred_proba, k=5)
    
    print(f"\n  Hit@1:  {h1:.4f}  ({h1*100:.1f}%)")
    print(f"  Hit@3:  {h3:.4f}  ({h3*100:.1f}%)")
    print(f"  Hit@5:  {h5:.4f}  ({h5*100:.1f}%)")
    print(f"  NDCG@3: {n3:.4f}")
    print(f"  NDCG@5: {n5:.4f}")
    print(f"\n  Revenue@5 (avg deno when hit): {rev['revenue_when_hit']:.1f} BDT")
    print(f"  Potential revenue (avg true deno): {rev['potential_revenue']:.1f} BDT")
    print(f"  Revenue capture rate: {rev['revenue_capture_rate']:.2%}")
    
    # Class distribution
    dist = class_distribution_report(y_true, y_pred_proba)
    
    if dist["warnings"]:
        print(f"\n  WARNINGS:")
        for w in dist["warnings"]:
            print(f"    - {w}")
    
    print(f"\n  Imbalance ratio (max/min class): {dist['imbalance_ratio']:.1f}x")
    print(f"{'='*60}\n")
    
    metrics = {
        "model_name": model_name,
        "n_test_samples": int(len(y_true)),
        "hit_at_1": float(h1),
        "hit_at_3": float(h3),
        "hit_at_5": float(h5),
        "ndcg_at_3": float(n3),
        "ndcg_at_5": float(n5),
        "revenue_at_5": rev,
        "class_distribution": {
            "imbalance_ratio": dist["imbalance_ratio"],
            "dominant_classes": dist["dominant_classes"],
            "never_predicted": dist["never_predicted"],
            "warnings": dist["warnings"]
        }
    }
    
    return metrics


def save_evaluation(metrics_list, output_path="artifacts/eval_metrics.json"):
    """Save evaluation metrics to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Evaluation metrics saved to {output_path}")
