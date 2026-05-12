"""DSD-MoE-HSI: Dual-Space Detection with Mixture-of-Experts for Hyperspectral Images

保留的单一路径:
- Routing Feature Space: 中层路由/光谱判别特征
- Image Feature Space: 高层语义/空间特征
- Unknown Detection: calibrated multi-signal anomaly score
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class UnifiedEncoder(nn.Module):
    """统一的特征提取器"""

    def __init__(self, num_bands, embed_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(num_bands, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.mid_layer = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.high_layer = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.routing_proj = nn.Linear(128, embed_dim)
        self.image_proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        shared = self.shared(x)
        mid = self.mid_layer(shared)
        routing_features = self.routing_proj(mid.mean(dim=[2, 3]))
        high = self.high_layer(mid)
        image_features = self.image_proj(high.squeeze(-1).squeeze(-1))
        return routing_features, image_features


class SimpleRouter(nn.Module):
    """简化的路由器"""

    def __init__(self, embed_dim, num_experts, noise_std=0.3, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.noise_std = noise_std
        self.temperature = temperature
        self.routing_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, num_experts),
        )

    def forward(self, routing_features, training=True):
        logits = self.routing_net(routing_features)
        if training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        return F.softmax(logits / max(self.temperature, 1e-6), dim=-1)


class ExpertLayer(nn.Module):
    """MoE专家层"""

    def __init__(self, embed_dim, num_experts=6, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            ) for _ in range(num_experts)
        ])

    def forward(self, x, routing_weights):
        batch_size = x.shape[0]
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)
        for i in range(self.top_k):
            for b in range(batch_size):
                expert_idx = topk_indices[b, i]
                weight = topk_weights[b, i]
                output[b] = output[b] + weight * self.experts[expert_idx](x[b:b + 1]).squeeze(0)
        return output


class DSDMoEHSI(nn.Module):
    """双空间检测模型（单一保留实现）"""

    def __init__(self, num_bands, num_classes, embed_dim=128,
                 num_experts=6, top_k=2, dropout=0.3,
                 num_prototypes=3, routing_weight=0.2,
                 inconsistency_weight=1.0,
                 distance_weight=1.0,
                 confidence_weight=0.2,
                 router_noise_std=0.3,
                 router_temperature=1.0,
                 min_proto_cluster_samples=9,
                 feature_space_mode='full',
                 use_moe=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.num_prototypes = num_prototypes
        self.routing_weight = routing_weight
        self.inconsistency_weight = inconsistency_weight
        self.distance_weight = distance_weight
        self.confidence_weight = confidence_weight
        self.min_proto_cluster_samples = max(num_prototypes * 2, min_proto_cluster_samples)
        self.feature_space_mode = feature_space_mode
        self.use_moe = use_moe

        self.encoder = UnifiedEncoder(num_bands, embed_dim)
        self.router = SimpleRouter(embed_dim, num_experts, noise_std=router_noise_std, temperature=router_temperature)
        self.moe = ExpertLayer(embed_dim, num_experts, top_k)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self.routing_bn = nn.BatchNorm1d(embed_dim)
        self.image_bn = nn.BatchNorm1d(embed_dim)

        self.register_buffer('routing_prototypes', torch.zeros(num_classes, num_prototypes, embed_dim))
        self.register_buffer('image_prototypes', torch.zeros(num_classes, num_prototypes, embed_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))

    def forward(self, x):
        routing_features, image_features = self.encoder(x)
        if self.feature_space_mode == 'routing_only':
            image_features = routing_features
        elif self.feature_space_mode == 'image_only':
            routing_features = image_features
        elif self.feature_space_mode != 'full':
            raise ValueError(f'Unsupported feature space mode: {self.feature_space_mode}')

        routing_features = self.routing_bn(routing_features)
        image_features = self.image_bn(image_features)
        routing_weights = self.router(routing_features, self.training)
        moe_output = self.moe(image_features, routing_weights) if self.use_moe else image_features
        logits = self.classifier(moe_output)
        return logits, routing_features, image_features, routing_weights

    def compute_routing_entropy(self, routing_weights):
        entropy = -(routing_weights.clamp_min(1e-8) * routing_weights.clamp_min(1e-8).log()).sum(dim=-1)
        norm = np.log(self.num_experts) if self.num_experts > 1 else 1.0
        return entropy / (norm + 1e-8)

    def compute_space_class_logits(self, routing_features, image_features):
        route_dist_to_protos = 1 - F.cosine_similarity(
            routing_features.unsqueeze(1).unsqueeze(2),
            self.routing_prototypes.unsqueeze(0),
            dim=-1,
        )
        img_dist_to_protos = 1 - F.cosine_similarity(
            image_features.unsqueeze(1).unsqueeze(2),
            self.image_prototypes.unsqueeze(0),
            dim=-1,
        )
        route_logits = -route_dist_to_protos.min(dim=-1)[0]
        img_logits = -img_dist_to_protos.min(dim=-1)[0]
        return route_logits, img_logits

    def compute_space_class_probs(self, routing_features, image_features, temperature=0.2):
        route_logits, img_logits = self.compute_space_class_logits(routing_features, image_features)
        route_prob = F.softmax(route_logits / temperature, dim=-1)
        img_prob = F.softmax(img_logits / temperature, dim=-1)
        return route_prob, img_prob

    def compute_dual_discrepancy(self, routing_features, image_features, temperature=0.2):
        route_prob, img_prob = self.compute_space_class_probs(
            routing_features, image_features, temperature=temperature
        )
        mean_prob = 0.5 * (route_prob + img_prob)
        js_route = F.kl_div(route_prob.clamp_min(1e-8).log(), mean_prob, reduction='none').sum(dim=-1)
        js_img = F.kl_div(img_prob.clamp_min(1e-8).log(), mean_prob, reduction='none').sum(dim=-1)
        return 0.5 * (js_route + js_img) / np.log(2.0)

    @torch.no_grad()
    def update_prototypes(self, routing_features, image_features, labels, momentum=0.99):
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue

            route_feats_c = routing_features[mask]
            img_feats_c = image_features[mask]
            n_samples = mask.sum().item()

            if n_samples < self.num_prototypes:
                route_mean = route_feats_c.mean(dim=0)
                img_mean = img_feats_c.mean(dim=0)
                if self.prototype_counts[c] == 0:
                    self.routing_prototypes[c, 0] = route_mean
                    self.image_prototypes[c, 0] = img_mean
                else:
                    self.routing_prototypes[c, 0] = momentum * self.routing_prototypes[c, 0] + (1 - momentum) * route_mean
                    self.image_prototypes[c, 0] = momentum * self.image_prototypes[c, 0] + (1 - momentum) * img_mean
            elif n_samples < self.min_proto_cluster_samples:
                if self.prototype_counts[c] == 0:
                    route_mean = route_feats_c.mean(dim=0)
                    img_mean = img_feats_c.mean(dim=0)
                    self.routing_prototypes[c, 0] = route_mean
                    self.image_prototypes[c, 0] = img_mean
                else:
                    route_assign = torch.cdist(route_feats_c, self.routing_prototypes[c], p=2).argmin(dim=1)
                    img_assign = torch.cdist(img_feats_c, self.image_prototypes[c], p=2).argmin(dim=1)
                    for p_idx in range(self.num_prototypes):
                        route_mask = route_assign == p_idx
                        img_mask = img_assign == p_idx
                        if route_mask.any():
                            route_proto = route_feats_c[route_mask].mean(dim=0)
                            self.routing_prototypes[c, p_idx] = momentum * self.routing_prototypes[c, p_idx] + (1 - momentum) * route_proto
                        if img_mask.any():
                            img_proto = img_feats_c[img_mask].mean(dim=0)
                            self.image_prototypes[c, p_idx] = momentum * self.image_prototypes[c, p_idx] + (1 - momentum) * img_proto
            else:
                route_feats_np = route_feats_c.cpu().numpy()
                kmeans = KMeans(n_clusters=self.num_prototypes, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(route_feats_np)
                for p_idx in range(self.num_prototypes):
                    cluster_mask = cluster_labels == p_idx
                    if cluster_mask.sum() == 0:
                        continue
                    route_proto = route_feats_c[cluster_mask].mean(dim=0)
                    img_proto = img_feats_c[cluster_mask].mean(dim=0)
                    if self.prototype_counts[c] == 0:
                        self.routing_prototypes[c, p_idx] = route_proto
                        self.image_prototypes[c, p_idx] = img_proto
                    else:
                        self.routing_prototypes[c, p_idx] = momentum * self.routing_prototypes[c, p_idx] + (1 - momentum) * route_proto
                        self.image_prototypes[c, p_idx] = momentum * self.image_prototypes[c, p_idx] + (1 - momentum) * img_proto

            self.prototype_counts[c] += 1

    def get_dual_predictions(self, routing_features, image_features):
        route_dist_to_protos = 1 - F.cosine_similarity(
            routing_features.unsqueeze(1).unsqueeze(2),
            self.routing_prototypes.unsqueeze(0),
            dim=-1,
        )
        img_dist_to_protos = 1 - F.cosine_similarity(
            image_features.unsqueeze(1).unsqueeze(2),
            self.image_prototypes.unsqueeze(0),
            dim=-1,
        )
        route_dist = route_dist_to_protos.min(dim=-1)[0]
        img_dist = img_dist_to_protos.min(dim=-1)[0]
        route_pred = route_dist.argmin(dim=-1)
        img_pred = img_dist.argmin(dim=-1)
        route_min_dist = route_dist.min(dim=-1)[0]
        img_min_dist = img_dist.min(dim=-1)[0]
        return route_pred, img_pred, route_min_dist, img_min_dist

    def compute_unknown_components(self, routing_features, image_features, routing_weights=None, logits=None):
        route_pred, img_pred, route_dist, img_dist = self.get_dual_predictions(routing_features, image_features)
        avg_dist = (route_dist + img_dist) / 2
        discrepancy_score = self.compute_dual_discrepancy(routing_features, image_features, temperature=0.2)
        if routing_weights is None:
            routing_entropy = torch.zeros_like(route_dist)
            topk_usage = None
        else:
            routing_entropy = self.compute_routing_entropy(routing_weights)
            topk_indices = torch.topk(routing_weights, self.moe.top_k, dim=-1).indices
            topk_usage = F.one_hot(topk_indices, num_classes=self.num_experts).float().sum(dim=1)
        confidence_gap = torch.zeros_like(route_dist)
        if logits is not None:
            confidence_gap = 1.0 - F.softmax(logits, dim=-1).max(dim=-1)[0]

        return {
            'route_pred': route_pred,
            'img_pred': img_pred,
            'route_dist': route_dist,
            'img_dist': img_dist,
            'avg_dist': avg_dist,
            'discrepancy': discrepancy_score,
            'routing_entropy': routing_entropy,
            'confidence_gap': confidence_gap,
            'topk_usage': topk_usage,
        }

    @staticmethod
    def build_classwise_stat_tensor(component, predicted_classes, per_class_stats, stat_name, default_value):
        stat_tensor = torch.full_like(component, float(default_value))
        if predicted_classes is None or not per_class_stats:
            return stat_tensor
        for class_id, class_stats in per_class_stats.items():
            if class_stats is None or stat_name not in class_stats:
                continue
            class_mask = predicted_classes == int(class_id)
            if class_mask.any():
                stat_value = torch.as_tensor(
                    class_stats[stat_name],
                    device=component.device,
                    dtype=component.dtype,
                )
                stat_tensor[class_mask] = stat_value
        return stat_tensor

    @staticmethod
    def normalize_score_component(component, stats=None, mode='none', predicted_classes=None, per_class_stats=None):
        if mode == 'none':
            return component
        if predicted_classes is not None and per_class_stats:
            if mode == 'zscore':
                default_mean = 0.0 if stats is None else stats['mean']
                default_std = 1.0 if stats is None else stats['std']
                mean = DSDMoEHSI.build_classwise_stat_tensor(
                    component, predicted_classes, per_class_stats, 'mean', default_mean
                )
                std = DSDMoEHSI.build_classwise_stat_tensor(
                    component, predicted_classes, per_class_stats, 'std', default_std
                )
                return (component - mean) / std.clamp_min(1e-6)
            if mode == 'minmax':
                default_low = 0.0 if stats is None else stats['min']
                default_high = 1.0 if stats is None else stats['max']
                low = DSDMoEHSI.build_classwise_stat_tensor(
                    component, predicted_classes, per_class_stats, 'min', default_low
                )
                high = DSDMoEHSI.build_classwise_stat_tensor(
                    component, predicted_classes, per_class_stats, 'max', default_high
                )
                return (component - low) / (high - low).clamp_min(1e-6)
            raise ValueError(f'Unsupported score normalization mode: {mode}')
        if stats is None:
            return component
        if mode == 'zscore':
            mean = stats['mean'].to(component.device)
            std = stats['std'].to(component.device)
            return (component - mean) / std.clamp_min(1e-6)
        if mode == 'minmax':
            low = stats['min'].to(component.device)
            high = stats['max'].to(component.device)
            return (component - low) / (high - low).clamp_min(1e-6)
        raise ValueError(f'Unsupported score normalization mode: {mode}')

    def compose_unknown_score(self, components, score_stats=None, score_norm='none',
                              distance_weight=None, routing_weight=None,
                              inconsistency_weight=None, confidence_weight=None,
                              predicted_classes=None, per_class_score_stats=None):
        dw = self.distance_weight if distance_weight is None else distance_weight
        rw = self.routing_weight if routing_weight is None else routing_weight
        iw = self.inconsistency_weight if inconsistency_weight is None else inconsistency_weight
        cw = self.confidence_weight if confidence_weight is None else confidence_weight

        def get_component_class_stats(component_name):
            if not per_class_score_stats:
                return None
            return {
                int(class_id): class_stats[component_name]
                for class_id, class_stats in per_class_score_stats.items()
                if class_stats is not None and component_name in class_stats
            }

        avg_dist = self.normalize_score_component(
            components['avg_dist'],
            None if score_stats is None else score_stats.get('avg_dist'),
            score_norm,
            predicted_classes=predicted_classes,
            per_class_stats=get_component_class_stats('avg_dist'),
        )
        entropy = self.normalize_score_component(
            components['routing_entropy'],
            None if score_stats is None else score_stats.get('routing_entropy'),
            score_norm,
            predicted_classes=predicted_classes,
            per_class_stats=get_component_class_stats('routing_entropy'),
        )
        discrepancy = self.normalize_score_component(
            components['discrepancy'],
            None if score_stats is None else score_stats.get('discrepancy'),
            score_norm,
            predicted_classes=predicted_classes,
            per_class_stats=get_component_class_stats('discrepancy'),
        )
        confidence_gap = self.normalize_score_component(
            components['confidence_gap'],
            None if score_stats is None else score_stats.get('confidence_gap'),
            score_norm,
            predicted_classes=predicted_classes,
            per_class_stats=get_component_class_stats('confidence_gap'),
        )

        unknown_score = dw * avg_dist + rw * entropy + iw * discrepancy + cw * confidence_gap
        return unknown_score

    def predict_unknown(self, routing_features, image_features, threshold=None,
                        routing_weights=None, logits=None, score_stats=None,
                        score_norm='none', distance_weight=None,
                        routing_weight=None, inconsistency_weight=None,
                        confidence_weight=None, predicted_classes=None,
                        per_class_score_stats=None, per_class_thresholds=None):
        components = self.compute_unknown_components(
            routing_features, image_features, routing_weights=routing_weights, logits=logits
        )
        known_predictions = predicted_classes
        if known_predictions is None:
            known_predictions = logits.argmax(dim=-1) if logits is not None else components['img_pred']
        unknown_score = self.compose_unknown_score(
            components,
            score_stats=score_stats,
            score_norm=score_norm,
            distance_weight=distance_weight,
            routing_weight=routing_weight,
            inconsistency_weight=inconsistency_weight,
            confidence_weight=confidence_weight,
            predicted_classes=known_predictions,
            per_class_score_stats=per_class_score_stats,
        )
        default_threshold = threshold if threshold is not None else 0.4
        applied_threshold = torch.full_like(unknown_score, float(default_threshold))
        used_per_class_threshold = torch.zeros_like(known_predictions, dtype=torch.bool)
        if per_class_thresholds and known_predictions is not None:
            for class_id, class_threshold in per_class_thresholds.items():
                class_mask = known_predictions == int(class_id)
                if class_mask.any():
                    applied_threshold[class_mask] = float(class_threshold)
                    used_per_class_threshold[class_mask] = True
        is_unknown = unknown_score > applied_threshold
        is_inconsistent = components['route_pred'] != components['img_pred']
        predictions = torch.where(is_unknown, torch.full_like(known_predictions, -1), known_predictions)
        return predictions, is_inconsistent, components, unknown_score, applied_threshold, used_per_class_threshold, known_predictions



class DSDMoELoss(nn.Module):
    """DSD-MoE损失函数（仅保留 semantic 对齐）"""

    def __init__(self, num_classes, temperature=0.1,
                 lambda_con=0.1, lambda_align=0.5, lambda_balance=0.05,
                 lambda_usage_balance=0.05,
                 lambda_proto_repel=0.01, consistency_temperature=1.0,
                 proto_repulsion_margin=0.6):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.lambda_con = lambda_con
        self.lambda_align = lambda_align
        self.lambda_balance = lambda_balance
        self.lambda_usage_balance = lambda_usage_balance
        self.lambda_proto_repel = lambda_proto_repel
        self.consistency_temperature = consistency_temperature
        self.proto_repulsion_margin = proto_repulsion_margin

    def contrastive_loss(self, features, prototypes, labels):
        if prototypes.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        features = F.normalize(features, dim=-1)
        num_classes, num_prototypes, embed_dim = prototypes.shape
        prototypes_flat = F.normalize(prototypes.view(-1, embed_dim), dim=-1)
        sim = torch.matmul(features, prototypes_flat.T) / self.temperature

        batch_size = features.shape[0]
        mask = torch.zeros(batch_size, num_classes * num_prototypes, device=features.device)
        for i in range(batch_size):
            c = labels[i].item()
            start_idx = c * num_prototypes
            end_idx = (c + 1) * num_prototypes
            mask[i, start_idx:end_idx] = 1.0 / num_prototypes

        log_prob = F.log_softmax(sim, dim=-1)
        return -(mask * log_prob).sum(dim=-1).mean()

    def alignment_loss(self, routing_features, image_features, labels,
                       routing_prototypes=None, image_prototypes=None):
        if routing_prototypes is None or image_prototypes is None:
            return torch.tensor(0.0, device=routing_features.device)
        if routing_prototypes.sum() == 0 or image_prototypes.sum() == 0:
            return torch.tensor(0.0, device=routing_features.device)

        known_mask = (labels >= 0) & (labels < self.num_classes)
        if known_mask.sum() == 0:
            return torch.tensor(0.0, device=routing_features.device)

        route_feats = F.normalize(routing_features[known_mask], dim=-1)
        img_feats = F.normalize(image_features[known_mask], dim=-1)
        route_protos = F.normalize(routing_prototypes.mean(dim=1), dim=-1)
        img_protos = F.normalize(image_prototypes.mean(dim=1), dim=-1)

        route_logits = torch.matmul(route_feats, route_protos.T) / self.consistency_temperature
        img_logits = torch.matmul(img_feats, img_protos.T) / self.consistency_temperature

        route_log_prob = F.log_softmax(route_logits, dim=-1)
        img_log_prob = F.log_softmax(img_logits, dim=-1)
        route_prob = route_log_prob.exp()
        img_prob = img_log_prob.exp()

        loss_ri = F.kl_div(route_log_prob, img_prob.detach(), reduction='batchmean')
        loss_ir = F.kl_div(img_log_prob, route_prob.detach(), reduction='batchmean')
        return 0.5 * (loss_ri + loss_ir)

    def prototype_repulsion_loss(self, prototypes):
        if prototypes.ndim != 3 or prototypes.shape[1] <= 1:
            return torch.tensor(0.0, device=prototypes.device)

        protos = F.normalize(prototypes, dim=-1)
        sim = torch.matmul(protos, protos.transpose(-1, -2))
        eye = torch.eye(sim.shape[-1], device=sim.device).unsqueeze(0)
        penalties = F.relu(sim - self.proto_repulsion_margin) * (1 - eye)
        denom = max(prototypes.shape[0] * prototypes.shape[1] * (prototypes.shape[1] - 1), 1)
        return penalties.sum() / denom

    def balance_loss(self, routing_weights):
        if routing_weights is None:
            return torch.tensor(0.0, device=self._dummy_device())
        expert_mean = routing_weights.mean(dim=0)
        target = torch.full_like(expert_mean, 1.0 / expert_mean.numel())
        return F.mse_loss(expert_mean, target)

    def usage_balance_loss(self, routing_weights, top_k):
        if routing_weights is None:
            return torch.tensor(0.0, device=self._dummy_device())
        topk_indices = torch.topk(routing_weights, top_k, dim=-1).indices
        usage = F.one_hot(topk_indices, num_classes=routing_weights.shape[-1]).float().sum(dim=1)
        usage_mean = usage.mean(dim=0)
        usage_mean = usage_mean / usage_mean.sum().clamp_min(1e-6)
        target = torch.full_like(usage_mean, 1.0 / usage_mean.numel())
        return F.mse_loss(usage_mean, target)

    def _dummy_device(self):
        return next(self.parameters()).device

    def forward(self, logits, routing_features, image_features,
                routing_prototypes, image_prototypes, labels, routing_weights=None, top_k=2):
        cls_loss = F.cross_entropy(logits, labels)
        route_con = self.contrastive_loss(routing_features, routing_prototypes, labels)
        img_con = self.contrastive_loss(image_features, image_prototypes, labels)
        con_loss = (route_con + img_con) / 2

        align_loss = self.alignment_loss(
            routing_features,
            image_features,
            labels,
            routing_prototypes=routing_prototypes,
            image_prototypes=image_prototypes,
        )
        bal_loss = self.balance_loss(routing_weights)
        usage_bal_loss = self.usage_balance_loss(routing_weights, top_k=top_k)
        proto_repel = 0.5 * (
            self.prototype_repulsion_loss(routing_prototypes) +
            self.prototype_repulsion_loss(image_prototypes)
        )

        total = cls_loss + self.lambda_con * con_loss + self.lambda_align * align_loss + \
            self.lambda_balance * bal_loss + self.lambda_usage_balance * usage_bal_loss + \
            self.lambda_proto_repel * proto_repel

        return total, {
            'cls': cls_loss.item(),
            'con': con_loss.item() if torch.is_tensor(con_loss) else con_loss,
            'align': align_loss.item() if torch.is_tensor(align_loss) else align_loss,
            'balance': bal_loss.item() if torch.is_tensor(bal_loss) else bal_loss,
            'usage_balance': usage_bal_loss.item() if torch.is_tensor(usage_bal_loss) else usage_bal_loss,
            'proto_repel': proto_repel.item() if torch.is_tensor(proto_repel) else proto_repel,
        }
