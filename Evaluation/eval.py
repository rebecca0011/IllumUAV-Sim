import open3d as o3d
import numpy as np
import os

def compute_geometric_accuracy(reconstruction_pcd, ground_truth_pcd, thresholds=[0.70, 0.80]):
    """
    计算几何精度：对于每个重建点，找到最近的地面真实点，并计算距离。
    然后返回70%和80%的距离阈值。

    :param reconstruction_pcd: 重建点云
    :param ground_truth_pcd: 地面真实点云
    :param thresholds: 需要计算的距离阈值百分比（默认为70%和80%）
    :return: 返回各个阈值对应的距离
    """
    # 构建地面真实点云的KD树
    gt_tree = o3d.geometry.KDTreeFlann(ground_truth_pcd)

    # 计算每个重建点到地面真实点的最近距离
    distances = []
    for point in reconstruction_pcd.points:
        [_, idx, dists] = gt_tree.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(dists[0]))  # 存储欧几里得距离

    distances = np.array(distances)

    # 计算70%和80%的距离阈值
    thresholds_dist = np.percentile(distances, [100 * t for t in thresholds])  # 返回70%和80%位置的距离
    return thresholds_dist


def compute_completeness(ground_truth_pcd, reconstruction_pcd, threshold=5, num_samples=None):
    # 如果未指定采样点数，使用重建点云中的点数
    if num_samples is None:
        num_samples = len(reconstruction_pcd.points)

    # 确保采样点数不大于地面真实点云中的点数
    gt_points = np.asarray(ground_truth_pcd.points)
    num_samples = min(num_samples, len(gt_points))

    # 均匀采样地面真实点云上的点
    sampled_indices = np.random.choice(len(gt_points), size=num_samples, replace=False)
    sampled_gt_points = gt_points[sampled_indices]

    # 构建重建点云的KD树
    rec_tree = o3d.geometry.KDTreeFlann(reconstruction_pcd)

    # 对每个采样点，找出最近的重建点，并判断是否满足阈值
    covered_count = 0
    for point in sampled_gt_points:
        [_, idx, dists] = rec_tree.search_knn_vector_3d(point, 1)
        distance = np.sqrt(dists[0])
        if distance <= threshold:
            covered_count += 1

    completeness = covered_count / num_samples
    return completeness


def evaluate_reconstruction(rec_path, gt_path, voxel_size):
    """
    voxel_size: 下采样的网格大小。
    如果您的单位是米，0.1 代表 10cm。
    如果想更准但慢一点，可以设为 0.05 (5cm)。
    如果想极快，设为 0.2 或 0.5。
    """
    print(f"读取重建点云: {rec_path}")
    if not os.path.exists(rec_path):
        print(f"错误: 找不到文件 {rec_path}")
        return
    reconstruction_pcd = o3d.io.read_point_cloud(rec_path)

    print(f"读取GT点云: {gt_path}")
    if not os.path.exists(gt_path):
        print(f"错误: 找不到文件 {gt_path}")
        return
    ground_truth_pcd = o3d.io.read_point_cloud(gt_path)

    print("-" * 30)
    print(f"【原始】重建点数: {len(reconstruction_pcd.points)}")
    print(f"【原始】GT点数: {len(ground_truth_pcd.points)}")

    # ================= 优化开始：下采样 =================
    if voxel_size > 0:
        print(f"正在进行体素下采样 (Voxel Size = {voxel_size})...")
        reconstruction_pcd = reconstruction_pcd.voxel_down_sample(voxel_size=voxel_size)
        ground_truth_pcd = ground_truth_pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"【优化后】重建点数: {len(reconstruction_pcd.points)}")
        print(f"【优化后】GT点数: {len(ground_truth_pcd.points)}")
    # ================= 优化结束 =================

    print("-" * 30)

    # 计算几何精度
    dist_70, dist_80 = compute_geometric_accuracy(reconstruction_pcd, ground_truth_pcd)
    print(f"Accuracy (70%): {dist_70:.5f}")
    print(f"Accuracy (80%): {dist_80:.5f}")

    # 计算完整性
    completeness = compute_completeness(ground_truth_pcd, reconstruction_pcd, threshold=voxel_size)
    print(f"Completeness: {completeness * 100:.2f}%")


if __name__ == "__main__":
    # 读取点云文件
    # reconstruction_pcd = o3d.io.read_point_cloud(r"G:\result\01sub.ply")
    # ground_truth_pcd = o3d.io.read_point_cloud(r"F:\result\Block_all.ply")
    # reconstruction_pcd=r"G:\result\colmap-night\05fused.ply"
    reconstruction_pcd = r"G:\result\05.ply"
    ground_truth_pcd=r"F:\result\Block_all.ply"


    # 执行评估
    evaluate_reconstruction(reconstruction_pcd, ground_truth_pcd, voxel_size=0.2)
    # evaluate_reconstruction(reconstruction_pcd, ground_truth_pcd)
