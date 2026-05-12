"""配置文件"""
import argparse


def get_args():
    parser = argparse.ArgumentParser('DSD-MoE open-set HSI recognition')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='PaviaU', choices=['PaviaU', 'Pavia', 'Salinas', 'IP', 'HT', 'LongKou'])
    parser.add_argument('--train_num', type=int, default=20, help='每类训练样本数（已知类）')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='从已知类训练样本中划出用于阈值校准的验证比例')

    # 开放集设置
    parser.add_argument('--known_classes_str', type=str, default='', help='已知类（逗号分隔，如 "1,2,3,4,5,6,7,8"）')
    parser.add_argument('--unknown_classes_str', type=str, default='', help='未知类（逗号分隔，如 "9"）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patch', type=int, default=9, help='空间patch大小')
    parser.add_argument('--momentum', type=float, default=0.99, help='原型更新动量')
    parser.add_argument('--eval_interval', type=int, default=10, help='评估间隔')

    # DSD-MoE 参数
    parser.add_argument('--num_features', type=int, default=128, help='特征维度')
    parser.add_argument('--num_prototypes', type=int, default=3, help='每个类别的原型数量')
    parser.add_argument('--num_experts', type=int, default=6, help='MoE专家数量')
    parser.add_argument('--top_k', type=int, default=2, help='MoE Top-K选择')
    parser.add_argument('--temperature', type=float, default=0.07, help='对比学习温度')
    parser.add_argument('--lambda_con', type=float, default=0.1, help='双空间对比损失权重')
    parser.add_argument('--lambda_align', type=float, default=0.3, help='双空间语义一致性损失权重')
    parser.add_argument('--lambda_balance', type=float, default=0.05, help='soft routing importance balance 权重')
    parser.add_argument('--lambda_usage_balance', type=float, default=0.05, help='top-k expert usage balance 权重')
    parser.add_argument('--lambda_proto_repel', type=float, default=0.01, help='同类多原型分散约束权重')
    parser.add_argument('--consistency_temperature', type=float, default=1.0, help='语义一致性分布温度')
    parser.add_argument('--distance_weight', type=float, default=1.0, help='平均原型距离在未知分数中的权重')
    parser.add_argument('--routing_weight', type=float, default=0.0, help='路由熵在未知分数中的权重')
    parser.add_argument('--inconsistency_weight', type=float, default=0.0, help='双空间分歧在未知分数中的权重（正向异常项）')
    parser.add_argument('--confidence_weight', type=float, default=0.0, help='分类低置信度在未知分数中的权重')
    parser.add_argument('--score_norm', type=str, default='zscore', choices=['none', 'zscore', 'minmax'], help='分数分量归一化方式')
    parser.add_argument('--calibration_mode', type=str, default='global', choices=['global', 'pred_class'], help='校准模式：当前默认简化主线依赖 global calibration；pred_class 保留为实验能力')
    parser.add_argument('--min_calibration_samples_per_class', type=int, default=5, help='启用按预测类校准所需的最少验证样本数')
    parser.add_argument('--threshold_quantile', type=float, default=0.95, help='已知类验证分数分位数阈值；<=0 时退回固定阈值')
    parser.add_argument('--proto_repulsion_margin', type=float, default=0.6, help='多原型分散约束的相似度边界')
    parser.add_argument('--min_proto_cluster_samples', type=int, default=9, help='启用类内KMeans更新原型所需的最少样本数')
    parser.add_argument('--detection_threshold', type=float, default=0.4, help='固定未知类检测阈值（无校准时使用）')
    parser.add_argument('--router_noise_std', type=float, default=0.3, help='训练时路由噪声强度')
    parser.add_argument('--router_temperature', type=float, default=1.0, help='路由softmax温度，>1 更平缓')

    # 模型选择/日志
    parser.add_argument('--best_metric', type=str, default='HOS', choices=['HOS', 'AUROC', 'Unknown_Acc', 'Composite'], help='训练中选择最佳模型的指标')
    parser.add_argument('--best_metric_alpha', type=float, default=0.5, help='Composite = alpha * HOS + (1-alpha) * AUROC')

    # 消融实验
    parser.add_argument(
        '--ablation_variant',
        type=str,
        default='mainline_simple',
        choices=['mainline_simple', 'full', 'routing_only', 'image_only', 'single_proto', 'no_proto_repel', 'no_moe', 'no_balance', 'distance_only', 'no_discrepancy', 'no_calibration', 'no_align',
                 'w_routing_only', 'w_image_only', 'w_no_moe', 'w_single_proto', 'w_no_proto_repel', 'w_no_balance', 'w_no_align', 'w_no_calibration'],
        help='消融实验变体预设；默认 `mainline_simple` 为带 calibration 的候选1简化主线，`distance_only` 保留为兼容别名'
    )
    parser.add_argument('--results_json', type=str, default='', help='若提供则将最终结果导出为 JSON')
    parser.add_argument('--disable_map_saving', action='store_true', help='禁用预测图和 GT 图保存（适合批量消融）')

    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./map')

    args = parser.parse_args()

    # 处理命令行传入的类别设置
    if args.known_classes_str:
        args.known_classes = [int(x.strip()) for x in args.known_classes_str.split(',')]
    if args.unknown_classes_str:
        args.unknown_classes = [int(x.strip()) for x in args.unknown_classes_str.split(',')]

    # 默认类别设置（如果未通过命令行指定）
    if not hasattr(args, 'known_classes') or not args.known_classes:
        if args.dataset == 'PaviaU':
            args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8]
            args.unknown_classes = [9]
        elif args.dataset == 'Pavia':
            args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8]
            args.unknown_classes = [9]
        elif args.dataset == 'Salinas':
            args.known_classes = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            args.unknown_classes = [6, 13, 14, 15, 16]
        elif args.dataset == 'IP':
            args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8]
            args.unknown_classes = [9, 10, 11, 12, 13, 14, 15, 16]
        elif args.dataset == 'HT':
            args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            args.unknown_classes = [10, 11, 12, 13, 14, 15]
        elif args.dataset == 'LongKou':
            args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8]
            args.unknown_classes = [9]

    # 如果只设置了known_classes但未设置unknown_classes，需要设置unknown_classes
    if not hasattr(args, 'unknown_classes') or not args.unknown_classes:
        if args.dataset == 'PaviaU':
            args.unknown_classes = [9]
        elif args.dataset == 'Pavia':
            args.unknown_classes = [9]
        elif args.dataset == 'Salinas':
            args.unknown_classes = [6, 13, 14, 15, 16]
        elif args.dataset == 'IP':
            args.unknown_classes = [9, 10, 11, 12, 13, 14, 15, 16]
        elif args.dataset == 'HT':
            args.unknown_classes = [10, 11, 12, 13, 14, 15]
        elif args.dataset == 'LongKou':
            args.unknown_classes = [9]

    return args
