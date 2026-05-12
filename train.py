"""开放集HSI训练脚本（仅保留 DSD-MoE）"""
import copy
import json
import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from config import get_args
from utils import init_dataset, get_dataloader
from model import DSDMoEHSI, DSDMoELoss


SCORE_COMPONENT_NAMES = ['avg_dist', 'routing_entropy', 'discrepancy', 'confidence_gap']
LEGACY_FULL_PRESET = {
    'routing_weight': 0.05,
    'inconsistency_weight': 0.2,
    'confidence_weight': 0.2,
    'score_norm': 'zscore',
    'threshold_quantile': 0.95,
    'calibration_mode': 'global',
}
MAINLINE_SIMPLE_PRESET = {
    'routing_weight': 0.0,
    'inconsistency_weight': 0.0,
    'confidence_weight': 0.0,
    'score_norm': 'zscore',
    'threshold_quantile': 0.95,
    'calibration_mode': 'global',
}
ABLATION_VARIANTS = {
    'mainline_simple': {**MAINLINE_SIMPLE_PRESET},
    'full': {**LEGACY_FULL_PRESET},
    # --- 旧消融（基于 LEGACY_FULL，保留兼容） ---
    'routing_only': {**LEGACY_FULL_PRESET, 'feature_space_mode': 'routing_only'},
    'image_only': {**LEGACY_FULL_PRESET, 'feature_space_mode': 'image_only'},
    'single_proto': {**LEGACY_FULL_PRESET, 'num_prototypes': 1},
    'no_proto_repel': {**LEGACY_FULL_PRESET, 'lambda_proto_repel': 0.0},
    'no_moe': {**LEGACY_FULL_PRESET, 'use_moe': False},
    'no_balance': {**LEGACY_FULL_PRESET, 'lambda_balance': 0.0, 'lambda_usage_balance': 0.0},
    'distance_only': {**MAINLINE_SIMPLE_PRESET},
    'no_discrepancy': {**LEGACY_FULL_PRESET, 'inconsistency_weight': 0.0},
    'no_calibration': {**LEGACY_FULL_PRESET, 'threshold_quantile': 0.0, 'score_norm': 'none'},
    'no_align': {**LEGACY_FULL_PRESET, 'lambda_align': 0.0},
    # --- 新消融（基于 MAINLINE_SIMPLE，逐一移除单个组件） ---
    'w_routing_only': {**MAINLINE_SIMPLE_PRESET, 'feature_space_mode': 'routing_only'},
    'w_image_only': {**MAINLINE_SIMPLE_PRESET, 'feature_space_mode': 'image_only'},
    'w_no_moe': {**MAINLINE_SIMPLE_PRESET, 'use_moe': False},
    'w_single_proto': {**MAINLINE_SIMPLE_PRESET, 'num_prototypes': 1},
    'w_no_proto_repel': {**MAINLINE_SIMPLE_PRESET, 'lambda_proto_repel': 0.0},
    'w_no_balance': {**MAINLINE_SIMPLE_PRESET, 'lambda_balance': 0.0, 'lambda_usage_balance': 0.0},
    'w_no_align': {**MAINLINE_SIMPLE_PRESET, 'lambda_align': 0.0},
    'w_no_calibration': {**MAINLINE_SIMPLE_PRESET, 'threshold_quantile': 0.0, 'score_norm': 'none'},
}


def get_explicit_cli_overrides(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit = set()
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if not token.startswith('--'):
            idx += 1
            continue
        if token == '--':
            break
        if '=' in token:
            key = token[2:].split('=', 1)[0]
            explicit.add(key)
            idx += 1
            continue
        key = token[2:]
        explicit.add(key)
        idx += 1
    return explicit


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_ablation_preset(args):
    variant = getattr(args, 'ablation_variant', 'mainline_simple')
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f'Unsupported ablation variant: {variant}')
    explicit_overrides = get_explicit_cli_overrides()
    for key, value in ABLATION_VARIANTS[variant].items():
        if key in explicit_overrides:
            continue
        setattr(args, key, value)
    return args


def export_results_json(args, final_metrics, best_threshold, best_selection):
    if not getattr(args, 'results_json', ''):
        return
    output_path = args.results_json
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        'model_name': 'DSC-MOE',
        'paper_method_name': 'DSC-MOE',
        'display_name': 'DSC-MOE' if getattr(args, 'ablation_variant', 'mainline_simple') == 'mainline_simple' else getattr(args, 'ablation_variant', 'mainline_simple'),
        'ablation_variant': args.ablation_variant,
        'dataset': args.dataset,
        'seed': args.seed,
        'epochs': args.epochs,
        'best_metric': args.best_metric,
        'best_threshold': float(best_threshold),
        'best_selection': float(best_selection),
        'val_ratio': float(args.val_ratio),
        'threshold_quantile': float(args.threshold_quantile),
        'num_prototypes': int(args.num_prototypes),
        'num_experts': int(args.num_experts),
        'top_k': int(args.top_k),
        'score_norm': args.score_norm,
        'calibration_mode': args.calibration_mode,
        'distance_weight': float(args.distance_weight),
        'routing_weight': float(args.routing_weight),
        'inconsistency_weight': float(args.inconsistency_weight),
        'confidence_weight': float(args.confidence_weight),
        'lambda_align': float(args.lambda_align),
        'lambda_balance': float(args.lambda_balance),
        'lambda_usage_balance': float(args.lambda_usage_balance),
        'lambda_proto_repel': float(args.lambda_proto_repel),
        'router_noise_std': float(args.router_noise_std),
        'router_temperature': float(args.router_temperature),
    }
    payload.update({
        key: float(value) if isinstance(value, (int, float, np.floating)) else value
        for key, value in final_metrics.items()
    })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_colors():
    return np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
        [255, 0, 255], [176, 48, 96], [46, 139, 87], [160, 32, 240],
        [255, 127, 80], [127, 255, 212], [218, 112, 214], [160, 82, 45],
        [127, 255, 0], [216, 191, 216], [128, 0, 0], [0, 128, 0], [0, 0, 128]
    ])


def draw_map(predictions, test_idx, dataset_info, save_path, known_classes):
    colors = get_colors()
    gt_path = os.path.join('./datasets', dataset_info['path'], dataset_info['gt_file_name'])
    gt = loadmat(gt_path)[dataset_info['gt_mat_name']].astype(np.int64)
    height, width = gt.shape
    pred_map = np.full((height, width), -2, dtype=np.int64)
    for i, idx in enumerate(test_idx):
        x = idx // width
        y = idx % width
        pred_map[x, y] = predictions[i]

    image = np.zeros((height, width, 3), dtype=np.uint8)
    unknown_mask = pred_map == -1
    background_mask = (gt == 0) | (pred_map == -2)

    # Use original class number for color mapping: predicted class c -> known_classes[c] -> colors[known_classes[c]-1]
    for c in range(len(known_classes)):
        mask = pred_map == c
        if mask.sum() > 0:
            original_class = known_classes[c]
            image[mask] = colors[original_class - 1]

    image[unknown_mask] = [255, 255, 255]
    image[background_mask] = [0, 0, 0]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()


def draw_gt_map(dataset_info, save_path):
    colors = get_colors()
    gt_path = os.path.join('./datasets', dataset_info['path'], dataset_info['gt_file_name'])
    gt = loadmat(gt_path)[dataset_info['gt_mat_name']].astype(np.int64)
    image = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

    known_classes = dataset_info['known_classes']
    known_set = set(known_classes)

    # Color known classes using original class number: colors[original_class - 1]
    for c in known_classes:
        mask = gt == c
        if mask.sum() > 0:
            image[mask] = colors[c - 1]

    # Color ALL unknown classes (any class not in known_classes and not 0) as white
    for c in np.unique(gt):
        if c != 0 and c not in known_set:
            mask = gt == c
            if mask.sum() > 0:
                image[mask] = [255, 255, 255]

    image[gt == 0] = [0, 0, 0]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()


def build_model_and_criterion(args, num_bands, num_classes, device):
    model = DSDMoEHSI(
        num_bands=num_bands,
        num_classes=num_classes,
        embed_dim=args.num_features,
        num_experts=args.num_experts,
        top_k=args.top_k,
        dropout=0.3,
        num_prototypes=args.num_prototypes,
        routing_weight=args.routing_weight,
        inconsistency_weight=args.inconsistency_weight,
        distance_weight=args.distance_weight,
        confidence_weight=args.confidence_weight,
        router_noise_std=args.router_noise_std,
        router_temperature=args.router_temperature,
        min_proto_cluster_samples=args.min_proto_cluster_samples,
        feature_space_mode='full' if args.ablation_variant == 'full' else getattr(args, 'feature_space_mode', 'full'),
        use_moe=getattr(args, 'use_moe', True),
    ).to(device)
    criterion = DSDMoELoss(
        num_classes,
        temperature=args.temperature,
        lambda_con=args.lambda_con,
        lambda_align=args.lambda_align,
        lambda_balance=args.lambda_balance,
        lambda_usage_balance=args.lambda_usage_balance,
        lambda_proto_repel=args.lambda_proto_repel,
        consistency_temperature=args.consistency_temperature,
        proto_repulsion_margin=args.proto_repulsion_margin,
    )
    return model, criterion


def train_epoch(model, dataloader, criterion, optimizer, device, args, momentum=0.99, epoch=0):
    model.train()
    total_loss = 0.0
    details_sum = {}

    for data, label, _ in tqdm(dataloader, desc='Training', leave=False):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()

        logits, routing_features, image_features, routing_weights = model(data)
        model.update_prototypes(routing_features.detach(), image_features.detach(), label, momentum)
        loss, details = criterion(
            logits,
            routing_features,
            image_features,
            model.routing_prototypes,
            model.image_prototypes,
            label,
            routing_weights,
            top_k=args.top_k,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in details.items():
            details_sum[k] = details_sum.get(k, 0.0) + v

    n_batches = max(len(dataloader), 1)
    return total_loss / n_batches, {k: v / n_batches for k, v in details_sum.items()}


@torch.no_grad()
def collect_outputs(model, dataloader, device):
    model.eval()

    all_labels = []
    all_logits = []
    all_indices = []
    all_routing_features = []
    all_image_features = []
    all_routing_weights = []

    for data, label, indices in dataloader:
        data, label = data.to(device), label.to(device)
        logits, routing_features, image_features, routing_weights = model(data)
        all_labels.append(label.cpu())
        all_logits.append(logits.cpu())
        all_indices.append(indices)
        all_routing_features.append(routing_features.cpu())
        all_image_features.append(image_features.cpu())
        all_routing_weights.append(routing_weights.cpu())

    if not all_labels:
        empty_long = torch.empty(0, dtype=torch.long)
        empty_float = torch.empty(0, dtype=torch.float32)
        empty_logits = torch.empty(0, model.num_classes, dtype=torch.float32)
        empty_feats = torch.empty(0, model.embed_dim, dtype=torch.float32)
        empty_weights = torch.empty(0, model.num_experts, dtype=torch.float32)
        return {
            'labels': empty_long,
            'logits': empty_logits,
            'indices': empty_long,
            'routing_features': empty_feats,
            'image_features': empty_feats.clone(),
            'routing_weights': empty_weights,
        }

    return {
        'labels': torch.cat(all_labels),
        'logits': torch.cat(all_logits),
        'indices': torch.cat(all_indices),
        'routing_features': torch.cat(all_routing_features),
        'image_features': torch.cat(all_image_features),
        'routing_weights': torch.cat(all_routing_weights),
    }


def summarize_stats(values):
    values = values.float()
    return {
        'mean': values.mean(),
        'std': values.std(unbiased=False).clamp_min(1e-6),
        'min': values.min(),
        'max': values.max(),
    }


def stats_to_python(stats):
    return {
        key: float(value.item() if torch.is_tensor(value) else value)
        for key, value in stats.items()
    }


def summarize_calibration_usage(used_per_class_threshold, known_mask, predicted_classes, per_class_thresholds):
    known_used = used_per_class_threshold[known_mask]
    known_predictions = predicted_classes[known_mask]
    total_known = int(known_mask.sum().item())
    used_count = int(known_used.sum().item())
    fallback_count = total_known - used_count

    fallback_classes = []
    if total_known > 0:
        fallback_mask = ~known_used
        if fallback_mask.any():
            fallback_classes = sorted(torch.unique(known_predictions[fallback_mask]).cpu().tolist())

    threshold_values = []
    if per_class_thresholds:
        threshold_values = [float(v) for v in per_class_thresholds.values()]

    summary = {
        'known_total': total_known,
        'used_count': used_count,
        'used_ratio': used_count / total_known if total_known > 0 else 0.0,
        'fallback_count': fallback_count,
        'fallback_classes': fallback_classes,
        'fallback_class_count': len(fallback_classes),
    }
    if threshold_values:
        summary.update({
            'per_class_threshold_min': min(threshold_values),
            'per_class_threshold_max': max(threshold_values),
            'per_class_threshold_mean': float(np.mean(threshold_values)),
        })
    else:
        summary.update({
            'per_class_threshold_min': None,
            'per_class_threshold_max': None,
            'per_class_threshold_mean': None,
        })
    return summary


@torch.no_grad()
def calibrate_threshold(model, val_outputs, device, num_classes, args):
    labels = val_outputs['labels']
    calibration = {
        'global_threshold': args.detection_threshold,
        'global_score_stats': None,
        'per_class_thresholds': {},
        'per_class_score_stats': {},
        'per_class_counts': {},
        'mode': args.calibration_mode,
    }
    if labels.numel() == 0:
        return calibration

    known_mask = labels < num_classes
    if known_mask.sum() == 0:
        return calibration

    logits = val_outputs['logits'][known_mask].to(device)
    routing_features = val_outputs['routing_features'][known_mask].to(device)
    image_features = val_outputs['image_features'][known_mask].to(device)
    routing_weights = val_outputs['routing_weights'][known_mask].to(device)

    components = model.compute_unknown_components(
        routing_features,
        image_features,
        routing_weights=routing_weights,
        logits=logits,
    )

    global_score_stats = {
        name: summarize_stats(components[name].detach().cpu())
        for name in SCORE_COMPONENT_NAMES
    }
    calibration['global_score_stats'] = global_score_stats

    predicted_classes = logits.argmax(dim=-1)
    global_unknown_score = model.compose_unknown_score(
        components,
        score_stats=global_score_stats,
        score_norm=args.score_norm,
        distance_weight=args.distance_weight,
        routing_weight=args.routing_weight,
        inconsistency_weight=args.inconsistency_weight,
        confidence_weight=args.confidence_weight,
    ).detach().cpu()

    if args.threshold_quantile > 0:
        calibration['global_threshold'] = torch.quantile(global_unknown_score, args.threshold_quantile).item()

    if args.calibration_mode != 'pred_class':
        return calibration

    predicted_classes_cpu = predicted_classes.detach().cpu()
    for class_id in range(num_classes):
        class_mask = predicted_classes_cpu == class_id
        class_count = int(class_mask.sum().item())
        calibration['per_class_counts'][class_id] = class_count
        if class_count < args.min_calibration_samples_per_class:
            continue

        class_component_stats = {
            name: stats_to_python(summarize_stats(components[name][class_mask].detach().cpu()))
            for name in SCORE_COMPONENT_NAMES
        }
        class_unknown_score = model.compose_unknown_score(
            components,
            score_stats=global_score_stats,
            score_norm=args.score_norm,
            distance_weight=args.distance_weight,
            routing_weight=args.routing_weight,
            inconsistency_weight=args.inconsistency_weight,
            confidence_weight=args.confidence_weight,
            predicted_classes=predicted_classes,
            per_class_score_stats={class_id: class_component_stats},
        ).detach().cpu()[class_mask]

        calibration['per_class_score_stats'][class_id] = class_component_stats
        if args.threshold_quantile > 0:
            calibration['per_class_thresholds'][class_id] = torch.quantile(class_unknown_score, args.threshold_quantile).item()

    return calibration


def compute_selection_score(metrics, args):
    if args.best_metric == 'HOS':
        return metrics['HOS']
    if args.best_metric == 'AUROC':
        return metrics['AUROC']
    if args.best_metric == 'Unknown_Acc':
        return metrics['Unknown_Acc']
    return args.best_metric_alpha * metrics['HOS'] + (1.0 - args.best_metric_alpha) * metrics['AUROC']


def format_stat_pair(name, tensor, known_mask, unknown_mask):
    known_mean = tensor[known_mask].mean().item() if known_mask.sum() > 0 else 0.0
    unknown_mean = tensor[unknown_mask].mean().item() if unknown_mask.sum() > 0 else 0.0
    gap = unknown_mean - known_mean
    return name, known_mean, unknown_mean, gap


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes, threshold, args, score_stats=None,
             per_class_score_stats=None, per_class_thresholds=None, print_details=True):
    outputs = collect_outputs(model, dataloader, device)
    labels = outputs['labels']
    logits = outputs['logits']
    routing_features = outputs['routing_features']
    image_features = outputs['image_features']
    routing_weights = outputs['routing_weights']
    test_indices = outputs['indices'].numpy()

    known_mask = labels < num_classes
    unknown_mask = ~known_mask
    predicted_classes = logits.argmax(dim=-1)

    predictions, is_inconsistent, components, unknown_score, applied_thresholds, used_per_class_threshold, predicted_classes = model.predict_unknown(
        routing_features.to(device),
        image_features.to(device),
        threshold=threshold,
        routing_weights=routing_weights.to(device),
        logits=logits.to(device),
        score_stats=score_stats,
        score_norm=args.score_norm,
        distance_weight=args.distance_weight,
        routing_weight=args.routing_weight,
        inconsistency_weight=args.inconsistency_weight,
        confidence_weight=args.confidence_weight,
        predicted_classes=predicted_classes.to(device),
        per_class_score_stats=per_class_score_stats,
        per_class_thresholds=per_class_thresholds,
    )

    predictions = predictions.cpu()
    unknown_score = unknown_score.cpu()
    is_inconsistent = is_inconsistent.cpu()
    applied_thresholds = applied_thresholds.cpu()
    used_per_class_threshold = used_per_class_threshold.cpu()
    predicted_classes = predicted_classes.cpu()
    components_cpu = {
        key: (value.cpu() if torch.is_tensor(value) else value)
        for key, value in components.items()
    }

    class_accs = []
    per_class_acc = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            acc = (predictions[mask] == c).float().mean().item()
            class_accs.append(acc)
            original_class_id = args.known_classes[c] if hasattr(args, 'known_classes') and c < len(args.known_classes) else (c + 1)
            per_class_acc[f'Class_{original_class_id}_Acc'] = acc
    aa = float(np.mean(class_accs)) if class_accs else 0.0

    oa = (predictions[known_mask] == labels[known_mask]).float().mean().item() if known_mask.sum() > 0 else 0.0
    unknown_acc = (predictions[unknown_mask] == -1).float().mean().item() if unknown_mask.sum() > 0 else 0.0
    hos = 2 * oa * unknown_acc / (oa + unknown_acc) if (oa + unknown_acc) > 0 else 0.0

    try:
        auroc = roc_auc_score(unknown_mask.numpy(), unknown_score.numpy())
    except Exception:
        auroc = 0.5

    confusion = np.zeros((num_classes, num_classes))
    for t in range(num_classes):
        for p in range(num_classes):
            confusion[t, p] = ((labels == t) & (predictions == p)).sum().item()
    n_samples = confusion.sum()
    if n_samples > 0:
        po = np.diag(confusion).sum() / n_samples
        row_sum = confusion.sum(axis=1)
        col_sum = confusion.sum(axis=0)
        pe = np.sum(row_sum * col_sum) / (n_samples ** 2)
        kappa = (po - pe) / (1 - pe + 1e-8)
    else:
        kappa = 0.0

    topk_usage_mean = None
    if components_cpu['topk_usage'] is not None:
        topk_usage_mean = components_cpu['topk_usage'].float().mean(dim=0).numpy() / max(args.top_k, 1)

    known_reject_rate = (predictions[known_mask] == -1).float().mean().item() if known_mask.sum() > 0 else 0.0
    calibration_summary = summarize_calibration_usage(
        used_per_class_threshold,
        known_mask,
        predicted_classes,
        per_class_thresholds,
    )

    metrics = {
        'OA': oa,
        'AA': aa,
        'Kappa': kappa,
        'AUROC': auroc,
        'Unknown_Acc': unknown_acc,
        'HOS': hos,
        'Threshold': float(threshold),
        'AppliedThresholdMean_Known': applied_thresholds[known_mask].mean().item() if known_mask.sum() > 0 else float(threshold),
        'AppliedThresholdMean_Unknown': applied_thresholds[unknown_mask].mean().item() if unknown_mask.sum() > 0 else float(threshold),
        'ScoreMean_Known': unknown_score[known_mask].mean().item() if known_mask.sum() > 0 else 0.0,
        'ScoreMean_Unknown': unknown_score[unknown_mask].mean().item() if unknown_mask.sum() > 0 else 0.0,
        'Entropy_Known': components_cpu['routing_entropy'][known_mask].mean().item() if known_mask.sum() > 0 else 0.0,
        'Entropy_Unknown': components_cpu['routing_entropy'][unknown_mask].mean().item() if unknown_mask.sum() > 0 else 0.0,
        'Discrepancy_Known': components_cpu['discrepancy'][known_mask].mean().item() if known_mask.sum() > 0 else 0.0,
        'Discrepancy_Unknown': components_cpu['discrepancy'][unknown_mask].mean().item() if unknown_mask.sum() > 0 else 0.0,
        'ConfidenceGap_Known': components_cpu['confidence_gap'][known_mask].mean().item() if known_mask.sum() > 0 else 0.0,
        'ConfidenceGap_Unknown': components_cpu['confidence_gap'][unknown_mask].mean().item() if unknown_mask.sum() > 0 else 0.0,
        'AvgDist_Known': components_cpu['avg_dist'][known_mask].mean().item() if known_mask.sum() > 0 else 0.0,
        'AvgDist_Unknown': components_cpu['avg_dist'][unknown_mask].mean().item() if unknown_mask.sum() > 0 else 0.0,
        'Inconsistent_Known': is_inconsistent[known_mask].float().mean().item() if known_mask.sum() > 0 else 0.0,
        'Inconsistent_Unknown': is_inconsistent[unknown_mask].float().mean().item() if unknown_mask.sum() > 0 else 0.0,
        'KnownRejectRate': known_reject_rate,
        'PerClassThresholdUsageRatio': calibration_summary['used_ratio'],
        'FallbackSampleCount': calibration_summary['fallback_count'],
        'FallbackClassCount': calibration_summary['fallback_class_count'],
    }
    metrics.update(per_class_acc)

    if print_details:
        print(f"\n  校准模式: {args.calibration_mode}")
        print(f"  全局校准阈值: {threshold:.4f}")
        print(f"  已知类平均生效阈值: {metrics['AppliedThresholdMean_Known']:.4f}")
        if args.calibration_mode == 'pred_class':
            print(f"  按预测类阈值使用占比: {calibration_summary['used_count']} / {calibration_summary['known_total']} ({calibration_summary['used_ratio']:.2%})")
            print(f"  回退到全局阈值的样本数: {calibration_summary['fallback_count']}")
            print(f"  回退涉及预测类数: {calibration_summary['fallback_class_count']} {calibration_summary['fallback_classes']}")
            if calibration_summary['per_class_threshold_min'] is not None:
                print(
                    f"  Per-class threshold 范围: min={calibration_summary['per_class_threshold_min']:.4f}, "
                    f"max={calibration_summary['per_class_threshold_max']:.4f}, "
                    f"mean={calibration_summary['per_class_threshold_mean']:.4f}"
                )
        print(f"  Known误拒率: {known_reject_rate:.4f}")
        if per_class_acc:
            print("  各已知类准确率:")
            for metric_name, metric_value in per_class_acc.items():
                print(f"    {metric_name}: {metric_value:.4f}")
        print(f"  不一致样本数: {is_inconsistent.sum().item()}")
        print(f"    已知类中不一致: {is_inconsistent[known_mask].sum().item()} / {known_mask.sum().item()}")
        print(f"    未知类中不一致: {is_inconsistent[unknown_mask].sum().item()} / {unknown_mask.sum().item()}")
        for display_name, comp_name in [
            ('平均距离', 'avg_dist'),
            ('路由熵', 'routing_entropy'),
            ('分歧分数', 'discrepancy'),
            ('低置信度', 'confidence_gap'),
        ]:
            name, known_mean, unknown_mean, gap = format_stat_pair(
                display_name,
                components_cpu[comp_name],
                known_mask,
                unknown_mask,
            )
            print(f"  {name} - 已知类: {known_mean:.4f}, 未知类: {unknown_mean:.4f}, 差值: {gap:.4f}")
        print(f"  未知分数 - 已知类: {metrics['ScoreMean_Known']:.4f}, 未知类: {metrics['ScoreMean_Unknown']:.4f}, 差值: {metrics['ScoreMean_Unknown'] - metrics['ScoreMean_Known']:.4f}")
        print(f"  Soft专家使用: {np.round(routing_weights.mean(dim=0).numpy(), 3)}")
        if topk_usage_mean is not None:
            print(f"  TopK专家使用: {np.round(topk_usage_mean, 3)}")

    return metrics, predictions.numpy(), test_indices, components_cpu, unknown_score.numpy(), labels.numpy()


def _save_score_distribution(args, unknown_scores, labels, threshold):
    """保存校准后未知分数分布数据为 npz，供后续画图使用"""
    import numpy as np
    output_dir = 'outputs/score_distribution'
    os.makedirs(output_dir, exist_ok=True)

    known_classes = np.array(getattr(args, 'known_classes', []), dtype=np.int64)
    unknown_classes = np.array(getattr(args, 'unknown_classes', []), dtype=np.int64)

    dataset_name = args.dataset
    if dataset_name == 'PaviaU':
        dist_key = 'UP'
    elif dataset_name == 'LongKou':
        dist_key = 'LK'
    elif dataset_name == 'HT':
        dist_key = 'HT'
    elif dataset_name == 'IP':
        dist_key = 'IP'
    else:
        dist_key = dataset_name

    npz_path = os.path.join(output_dir, f'score_dist_{dist_key}.npz')
    np.savez(npz_path,
             scores=unknown_scores.astype(np.float32),
             labels=labels.astype(np.int64),
             threshold=np.float32(threshold),
             known_classes=known_classes,
             unknown_classes=unknown_classes,
             dataset=dist_key)
    print(f'Score distribution saved to: {npz_path}')
    print(f'  N_test={len(unknown_scores)}  threshold={threshold:.4f}')
    print(f'  known_classes={known_classes.tolist()}  unknown_classes={unknown_classes.tolist()}')
@torch.no_grad()
def analyze_dual_space(model, dataloader, device, num_classes):
    model.eval()

    all_labels = []
    all_routing_features = []
    all_image_features = []

    for data, label, _ in dataloader:
        data, label = data.to(device), label.to(device)
        _, routing_features, image_features, _ = model(data)
        all_labels.append(label.cpu())
        all_routing_features.append(routing_features.detach().cpu())
        all_image_features.append(image_features.detach().cpu())

    labels = torch.cat(all_labels)
    routing_features = torch.cat(all_routing_features)
    image_features = torch.cat(all_image_features)
    known_mask = labels < num_classes
    unknown_mask = ~known_mask

    route_pred, img_pred, route_min_dist, img_min_dist = model.get_dual_predictions(
        routing_features.to(device), image_features.to(device)
    )
    route_min = route_min_dist.cpu()
    img_min = img_min_dist.cpu()
    route_pred = route_pred.cpu()
    img_pred = img_pred.cpu()

    print("\n双空间距离分析:")
    print(f"  Routing - Known: {route_min[known_mask].mean():.4f}, Unknown: {route_min[unknown_mask].mean():.4f}")
    print(f"  Image   - Known: {img_min[known_mask].mean():.4f}, Unknown: {img_min[unknown_mask].mean():.4f}")
    consist_known = (route_pred[known_mask] == img_pred[known_mask]).float().mean()
    consist_unknown = (route_pred[unknown_mask] == img_pred[unknown_mask]).float().mean()
    print(f"  一致性 - Known: {consist_known:.4f}, Unknown: {consist_unknown:.4f}")
    print(f"  差异: {consist_known - consist_unknown:.4f} (正=已知类更一致)")


def main():
    args = get_args()
    args = apply_ablation_preset(args)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'\nLoading dataset: {args.dataset}')
    print(f'Ablation Variant: {args.ablation_variant}')
    dataset_info = init_dataset(args)
    train_loader = get_dataloader(dataset_info, args.batch_size, 'train')
    val_loader = get_dataloader(dataset_info, args.batch_size, 'val')
    test_loader = get_dataloader(dataset_info, args.batch_size, 'test')

    num_classes = dataset_info['num_classes']
    num_bands = dataset_info['num_bands']
    print(f'Known classes: {num_classes}, Bands: {num_bands}, Patch: {dataset_info["patch_size"]}')

    model, criterion = build_model_and_criterion(args, num_bands, num_classes, device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    print('\n' + '=' * 70)
    print('Model: DSD-MoE')
    print('Routing Space + Image Space -> Distance-first Unknown Detection')
    print(f'Ablation Variant: {args.ablation_variant}')
    print(f'Unknown score weights: dist={args.distance_weight}, entropy={args.routing_weight}, discrepancy={args.inconsistency_weight}, confidence={args.confidence_weight}')
    print(f'Score normalization: {args.score_norm} | Threshold quantile: {args.threshold_quantile}')
    print('=' * 70)

    best_selection = -float('inf')
    best_metrics = None
    best_state_dict = None
    best_threshold = args.detection_threshold
    best_score_stats = None
    best_per_class_score_stats = None
    best_per_class_thresholds = None

    for epoch in range(1, args.epochs + 1):
        loss, details = train_epoch(model, train_loader, criterion, optimizer, device, args,
                                    momentum=args.momentum, epoch=epoch)
        scheduler.step()

        should_eval = (epoch % args.eval_interval == 0) or (epoch == args.epochs)
        if should_eval:
            calibration = calibrate_threshold(model, collect_outputs(model, val_loader, device), device, num_classes, args)
            threshold = calibration['global_threshold']
            score_stats = calibration['global_score_stats']
            per_class_score_stats = calibration['per_class_score_stats']
            per_class_thresholds = calibration['per_class_thresholds']
            print(f"\nEpoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Cls: {details['cls']:.4f} | Con: {details['con']:.4f} | "
                  f"Align: {details['align']:.4f} | Bal: {details['balance']:.4f} | "
                  f"UsageBal: {details.get('usage_balance', 0.0):.4f} | Repel: {details.get('proto_repel', 0.0):.4f}")
            metrics, _, _, _, _, _ = evaluate(
                model,
                test_loader,
                device,
                num_classes,
                threshold=threshold,
                args=args,
                score_stats=score_stats,
                per_class_score_stats=per_class_score_stats,
                per_class_thresholds=per_class_thresholds,
                print_details=True,
            )

            print(f"  OA: {metrics['OA']:.4f} | AA: {metrics['AA']:.4f} | HOS: {metrics['HOS']:.4f} | "
                  f"AUROC: {metrics['AUROC']:.4f} | Unknown_Acc: {metrics['Unknown_Acc']:.4f}")

            selection_score = compute_selection_score(metrics, args)
            if best_metrics is None or selection_score > best_selection:
                best_selection = selection_score
                best_metrics = metrics
                best_threshold = threshold
                best_score_stats = score_stats
                best_per_class_score_stats = per_class_score_stats
                best_per_class_thresholds = per_class_thresholds
                best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    analyze_dual_space(model, test_loader, device, num_classes)

    print('\n' + '=' * 70)
    print(f'最终采用阈值: {best_threshold:.4f}')
    print(f'Calibration Mode: {args.calibration_mode}')
    if args.calibration_mode == 'pred_class' and best_per_class_thresholds:
        threshold_values = list(best_per_class_thresholds.values())
        print(f'Per-class threshold count: {len(best_per_class_thresholds)}')
        print(f'Per-class threshold range: {min(threshold_values):.4f} ~ {max(threshold_values):.4f}')
    print('=' * 70)
    print('(阈值来自已知类验证集校准；若未启用校准则使用固定阈值)')

    final_metrics, final_preds, final_test_idx, _, final_unknown_scores, final_labels = evaluate(
        model,
        test_loader,
        device,
        num_classes,
        threshold=best_threshold,
        args=args,
        score_stats=best_score_stats,
        per_class_score_stats=best_per_class_score_stats,
        per_class_thresholds=best_per_class_thresholds,
        print_details=True,
    )

    # Save calibrated score distribution data
    _save_score_distribution(args, final_unknown_scores, final_labels, best_threshold)

    print('\n' + '=' * 70)
    print('Final Results')
    print('=' * 70)
    print('Model: DSD-MoE')
    print(f'Threshold: {best_threshold:.4f}')
    print(f'Best Metric: {args.best_metric}')
    print(f'OA: {final_metrics["OA"]:.4f}')
    print(f'AA: {final_metrics["AA"]:.4f}')
    print(f'HOS: {final_metrics["HOS"]:.4f}')
    print(f'AUROC: {final_metrics["AUROC"]:.4f}')
    print(f'Best Selected Score: {best_selection:.4f}')
    print(f'Unknown Acc: {final_metrics["Unknown_Acc"]:.4f}')
    for metric_name, metric_value in final_metrics.items():
        if metric_name.startswith('Class_') and metric_name.endswith('_Acc'):
            print(f'{metric_name}: {metric_value:.4f}')

    export_results_json(args, final_metrics, best_threshold, best_selection)

    if not args.disable_map_saving:
        os.makedirs(args.save_dir, exist_ok=True)
        draw_map(final_preds, final_test_idx, dataset_info,
                 os.path.join(args.save_dir, f'{args.dataset}_dsd_moe_prediction.png'),
                 args.known_classes)
        draw_gt_map(dataset_info, os.path.join(args.save_dir, f'{args.dataset}_ground_truth.png'))


if __name__ == '__main__':
    main()
