# -*- coding: utf-8 -*-
"""
uav_station_grid_cm.py
按摄影测量关系（厘米制）从相机参数与重叠度求摄站间距，依次叠加生成网格（可蛇形排序），
并可选裁剪到边界多边形内，导出 UE 友好的 CSV (X,Y,Z,Yaw,Pitch,Roll)。

主要功能：
- 生成标准正射网格航线。
- 新增：通过 --oblique 生成五向倾斜航线（1个正射+4个倾斜），用于三维重建。
- 新增：--auto_rotate 航向优化，可与正射和倾斜模式配合使用，提高效率。
- 导出的 Z = base_z + H（单位：cm），满足“Z=--H”的要求。

用法（示例）:
  # 1. 生成标准正射航线，并自动优化航向
  python uav_station_grid_cm.py --use_default_boundary --H 10000 --auto_rotate --csv out_nadir.csv --plot

  # 2. 生成五向倾斜航线（1正射+4倾斜），并自动优化航向
  python uav_station_grid_cm.py --use_default_boundary --H 10000 --oblique --gimbal_pitch -45 --auto_rotate --csv out_oblique.csv --plot
"""

from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional
import argparse
import numpy as np
import sys
import os

try:
    from shapely.geometry import Polygon, Point
    from shapely.prepared import prep
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False

try:
    # --- 新增代码：在导入 pyplot 之前手动设置后端 ---
    import matplotlib
    matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'Agg' 等
    # ---------------------------------------------
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

except Exception:
    MPL_OK = False

try:
    import pandas as pd

    PANDAS_OK = True
except Exception:
    PANDAS_OK = False


# ------------------------------------------------------


# -------------------- 数据模型与核心计算 --------------------
@dataclass
class CameraModelCM:
    """相机/成像参数（单位均为厘米或像素）。"""
    f_cm: float  # 焦距 f（cm）
    sensor_w_cm: float  # 传感器宽（cm）
    sensor_h_cm: float  # 传感器高（cm）
    Nx: int  # 水平方向像素数
    Ny: int  # 垂直方向像素数
    H_cm: float  # 飞行高度 H（cm，地面到相机主点）

    def gsd_cm_per_px(self) -> float:
        """GSD = (H * p) / f，其中 p = sensor_w / Nx（cm/px）。"""
        p_cm = self.sensor_w_cm / float(self.Nx)
        return (self.H_cm * p_cm) / self.f_cm

    def footprint_cm(self) -> Tuple[float, float]:
        """单张影像在地面的覆盖尺寸（cm）"""
        gsd = self.gsd_cm_per_px()
        Wx = gsd * self.Nx  # 地面覆盖宽（横向）
        Wy = gsd * self.Ny  # 地面覆盖高（纵向）
        return Wx, Wy


def station_spacing_cm(
        cam: CameraModelCM,
        front_overlap: float,
        side_overlap: float
) -> Tuple[float, float, float]:
    """
    由重叠度将影像足迹折算为摄站间距（厘米）。
    返回：GSD(cm/px), Δalong(cm), Δcross(cm)
    """
    gsd = cam.gsd_cm_per_px()
    Wx, Wy = cam.footprint_cm()
    d_along = Wy * (1.0 - front_overlap)  # 前方重叠 -> 沿程步长
    d_cross = Wx * (1.0 - side_overlap)  # 旁向重叠 -> 旁向步长
    return gsd, d_along, d_cross


def _bbox_from_polygon_cm(boundary_points_cm: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in boundary_points_cm]
    ys = [p[1] for p in boundary_points_cm]
    return min(xs), min(ys), max(xs), max(ys)


def _get_optimal_rotation_rad(boundary_points_cm: List[Tuple[float, float]]) -> float:
    """使用 PCA 计算边界多边形的主方向，返回最佳旋转角度（弧度）。"""
    coords = np.array(boundary_points_cm)
    center = np.mean(coords, axis=0)
    centered_coords = coords - center
    cov_matrix = np.cov(centered_coords, rowvar=False)
    _, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_vector = eigenvectors[:, -1]
    angle_rad = np.arctan2(principal_vector[1], principal_vector[0])
    return angle_rad


def _rotate_points(points: np.ndarray, angle_rad: float, center: np.ndarray) -> np.ndarray:
    """围绕中心点旋转点集。"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return (points - center) @ rotation_matrix.T + center


def generate_stations_grid_cm(
        boundary_points_cm: List[Tuple[float, float]],
        d_along_cm: float,
        d_cross_cm: float,
        origin: Literal["bbox_min", "bbox_max", "custom"] = "bbox_min",
        custom_origin_cm: Optional[Tuple[float, float]] = None,
        snake: bool = True,
        clip_to_polygon: bool = True,
        auto_rotate: bool = False
) -> Tuple[np.ndarray, float]:
    """
    生成网格，返回站点坐标和网格的旋转角度（弧度）。
    返回：(N×2 的数组 [x_cm, y_cm], rotation_rad)
    """
    boundary_np = np.array(boundary_points_cm)

    if auto_rotate:
        rotation_angle_rad = _get_optimal_rotation_rad(boundary_points_cm)
        print(f"[INFO] 自动优化航向：旋转角度 {np.degrees(rotation_angle_rad):.2f}°")
        boundary_center = np.mean(boundary_np, axis=0)
        proc_boundary_points_cm = _rotate_points(boundary_np, -rotation_angle_rad, boundary_center).tolist()
    else:
        rotation_angle_rad = 0.0
        boundary_center = np.array([0, 0])  # 仅为定义变量
        proc_boundary_points_cm = boundary_points_cm

    minx, miny, maxx, maxy = _bbox_from_polygon_cm(proc_boundary_points_cm)

    if origin == "bbox_min":
        x0, y0 = minx, miny
    elif origin == "bbox_max":
        x0, y0 = maxx, maxy
    elif origin == "custom":
        if custom_origin_cm is None:
            raise ValueError("origin='custom' 时必须提供 custom_origin_cm")
        if auto_rotate:
            x0, y0 = _rotate_points(np.array([custom_origin_cm]), -rotation_angle_rad, boundary_center)[0]
        else:
            x0, y0 = custom_origin_cm
    else:
        raise ValueError("origin 取值必须为 'bbox_min' | 'bbox_max' | 'custom'")

    n_cols = int(np.ceil((maxx - minx) / max(d_cross_cm, 1e-6))) + 1
    n_rows = int(np.ceil((maxy - miny) / max(d_along_cm, 1e-6))) + 1

    xs = x0 + np.arange(0, n_cols) * d_cross_cm
    ys = y0 + np.arange(0, n_rows) * d_along_cm

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts_rotated = np.column_stack([X.ravel(), Y.ravel()])

    if auto_rotate:
        pts = _rotate_points(pts_rotated, rotation_angle_rad, boundary_center)
    else:
        pts = pts_rotated

    if clip_to_polygon:
        if not SHAPELY_OK:
            raise RuntimeError("clip_to_polygon=True 需要 shapely；请安装 shapely 或将其设为 False")
        poly = Polygon(boundary_points_cm)
        prepared = prep(poly)
        mask = np.array([prepared.contains(Point(p)) for p in pts])
        pts = pts[mask]
        if auto_rotate:
            pts_rotated = pts_rotated[mask]

    if snake and pts.size > 0:
        # 在旋转后的坐标系中进行蛇形排序，以保证航线是直的
        current_pts_for_sort = pts_rotated if auto_rotate else pts
        order = np.lexsort((current_pts_for_sort[:, 1], current_pts_for_sort[:, 0]))
        sorted_indices = order

        # 按列分组并反转
        unique_xs, counts = np.unique(current_pts_for_sort[order, 0], return_counts=True)
        final_indices = []
        start_idx = 0
        for i, count in enumerate(counts):
            end_idx = start_idx + count
            col_indices = sorted_indices[start_idx:end_idx]
            if i % 2 == 1:
                final_indices.extend(col_indices[::-1])
            else:
                final_indices.extend(col_indices)
            start_idx = end_idx
        pts = pts[final_indices]

    return pts, rotation_angle_rad


def shift_boundary(boundary_points: List[Tuple[float, float]], direction_yaw_deg: float, offset: float) -> List[
    Tuple[float, float]]:
    """根据相机朝向（世界系yaw）和偏移量，平移边界。"""
    if offset == 0:
        return boundary_points

    angle_rad = np.radians(direction_yaw_deg)
    # 飞机需要向相机朝向的相反方向移动
    shift_x = -offset * np.sin(angle_rad)
    shift_y = -offset * np.cos(angle_rad)

    return [(x + shift_x, y + shift_y) for x, y in boundary_points]


def export_csv_for_ue(path_csv: str, stations: np.ndarray, encoding: str = "utf-8"):
    """导出为 UE 友好的 CSV：列名 X,Y,Z,Yaw,Pitch,Roll（单位：cm/度）。"""
    if not PANDAS_OK:
        print("[ERROR] 需要 pandas 库来导出 CSV。请运行 `pip install pandas`。", file=sys.stderr)
        return
    df = pd.DataFrame(stations, columns=["X", "Y", "Z", "Yaw", "Pitch", "Roll"])
    df.to_csv(path_csv, index=False, encoding=encoding)
    print(f"[OK] 已导出 UE CSV: {os.path.abspath(path_csv)}")


# -------------------------- CLI 与主流程 --------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="无人机摄影测量网格生成器")
    # 相机/高度（厘米）
    p.add_argument("--f", type=float, default=3.5, help="焦距 f (cm)")
    p.add_argument("--sw", type=float, default=3.6, help="传感器宽 (cm)")
    p.add_argument("--sh", type=float, default=2.4, help="传感器高 (cm)")
    p.add_argument("--Nx", type=int, default=6000, help="横向像素数 Nx")
    p.add_argument("--Ny", type=int, default=4000, help="垂直像素数 Ny")
    p.add_argument("--H", type=float, default=25000.0, help="相对飞行高度 H (cm)")
    p.add_argument("--base_z", type=float, default=0.0, help="地面基准高度 Z (cm)")

    # 重叠度
    p.add_argument("--front", type=float, default=0.80, help="前方重叠度 (0-1)")
    p.add_argument("--side", type=float, default=0.70, help="旁向重叠度 (0-1)")

    # 航线模式
    p.add_argument("--oblique", action="store_true",  default=True,help="生成五向倾斜航线（1正射+4倾斜）")
    p.add_argument("--gimbal_pitch", type=float, default=-45.0, help="倾斜航线的相机俯仰角 (度)")

    # 边界与网格
    p.add_argument("--origin", choices=["bbox_min", "bbox_max", "custom"], default="bbox_min", help="叠加基准点")
    p.add_argument("--origin_x", type=float, default=None, help="自定义原点 X (cm)")
    p.add_argument("--origin_y", type=float, default=None, help="自定义原点 Y (cm)")
    p.add_argument("--no_snake", action="store_true", help="关闭蛇形排序")
    p.add_argument("--no_clip", action="store_true", help="不裁剪到多边形（更快）")
    p.add_argument("--auto_rotate", action="store_true", help="自动旋转网格以优化航向（推荐）")

    # 导出与展示
    p.add_argument("--csv", type=str, default="drone_stations_250.csv", help="输出 CSV 路径")
    p.add_argument("--plot", action="store_true",  default=True,help="显示示意图")

    # 内置示例边界
    p.add_argument("--use_default_boundary", action="store_true", help="使用内置示例边界点")
    return p.parse_args()


def main():
    args = parse_args()

    if args.use_default_boundary:
        boundary_points = [(-91500, 16150), (-38750, -62100), (39250, -62900), (78750, 15600), (26050, 68050)]
    else:
        print("未指定 --use_default_boundary，默认使用示例边界。", file=sys.stderr)
        boundary_points = [(-91500, 16150), (-38750, -62100), (39250, -62900), (78750, 15600), (26050, 68050)]

    cam = CameraModelCM(f_cm=args.f, sensor_w_cm=args.sw, sensor_h_cm=args.sh, Nx=args.Nx, Ny=args.Ny, H_cm=args.H)
    gsd, d_along_cm, d_cross_cm = station_spacing_cm(cam, args.front, args.side)
    print(f"[INFO] GSD = {gsd:.4f} cm/px | 沿程步长 = {d_along_cm:.2f} cm | 旁向步长 = {d_cross_cm:.2f} cm")

    custom_origin = (args.origin_x, args.origin_y) if args.origin == "custom" else None
    z_value = args.base_z + args.H

    all_missions_data = []  # 用于绘图
    all_stations_pose = []  # 用于导出

    # 定义航线任务
    if args.oblique:
        print("[INFO] 生成五向倾斜航线...")
        # (任务名, 俯仰角, 偏航角偏移, 颜色)
        # 偏航角偏移是相对于主航向的
        tasks = [
            ("Nadir", -90.0, 0.0, 'black'),
            ("Forward", args.gimbal_pitch, 0.0, 'red'),
            ("Backward", args.gimbal_pitch, 180.0, 'blue'),
            ("Right", args.gimbal_pitch, 90.0, 'green'),
            ("Left", args.gimbal_pitch, 270.0, 'purple')
        ]
    else:
        print("[INFO] 生成标准正射航线...")
        tasks = [("Nadir", -90.0, 0.0, 'blue')]

    # 为所有任务预先计算一次主航向
    base_rotation_rad = _get_optimal_rotation_rad(boundary_points) if args.auto_rotate else 0.0
    base_rotation_deg = np.degrees(base_rotation_rad)

    for name, pitch, yaw_offset_deg, color in tasks:
        print(f"\n--- 正在生成: {name} ({color}) 层 ---")

        # 计算相机姿态
        mission_yaw_deg = (base_rotation_deg + yaw_offset_deg) % 360
        mission_pitch_deg = pitch

        # 计算偏移量
        if pitch == -90.0:
            offset = 0.0
        else:
            angle_from_vertical_rad = np.radians(90.0 + pitch)  # pitch是负数
            offset = args.H * np.tan(angle_from_vertical_rad)
        print(f"[INFO] 姿态 Yaw={mission_yaw_deg:.2f}°, Pitch={pitch}° | 偏移量={offset:.2f} cm")

        # 根据相机朝向和偏移量，计算无人机的“飞行边界”
        # 注意：这里的相机朝向是世界坐标系下的最终朝向
        flight_boundary = shift_boundary(boundary_points, mission_yaw_deg, offset)

        try:
            # 在飞行边界上生成网格点
            stations_xy, _ = generate_stations_grid_cm(
                boundary_points_cm=flight_boundary,
                d_along_cm=d_along_cm,
                d_cross_cm=d_cross_cm,
                origin=args.origin,
                custom_origin_cm=custom_origin,
                snake=(not args.no_snake),
                clip_to_polygon=(not args.no_clip),
                auto_rotate=args.auto_rotate
            )
        except RuntimeError as e:
            print(f"[WARN] {e}\n[WARN] 自动切换为不裁剪模式。", file=sys.stderr)
            stations_xy, _ = generate_stations_grid_cm(
                boundary_points_cm=flight_boundary, d_along_cm=d_along_cm, d_cross_cm=d_cross_cm,
                origin=args.origin, custom_origin_cm=custom_origin, snake=(not args.no_snake),
                clip_to_polygon=False, auto_rotate=args.auto_rotate
            )

        if stations_xy.size == 0:
            print("[WARN] 该层未生成任何站点。")
            continue

        print(f"[INFO] 生成站点数: {len(stations_xy)}")
        all_missions_data.append({'pts': stations_xy, 'color': color, 'label': f'{name} ({len(stations_xy)})'})

        # 组合为完整的位姿信息 (X, Y, Z, Yaw, Pitch, Roll)
        z_col = np.full((len(stations_xy), 1), z_value)
        yaw_col = np.full((len(stations_xy), 1), mission_yaw_deg)
        pitch_col = np.full((len(stations_xy), 1), mission_pitch_deg)
        roll_col = np.zeros((len(stations_xy), 1))  # Roll通常为0

        full_pose = np.hstack([stations_xy, z_col, yaw_col, pitch_col, roll_col])
        all_stations_pose.append(full_pose)

    # 合并所有航点并导出
    if not all_stations_pose:
        print("\n[ERROR] 未能生成任何航点。", file=sys.stderr)
        sys.exit(1)

    final_stations = np.vstack(all_stations_pose)
    export_csv_for_ue(args.csv, final_stations)
    print(f"\n[INFO] 总计站点数: {len(final_stations)}")

    # 可视化
    if args.plot:
        if not MPL_OK:
            print("[WARN] 未安装 matplotlib，无法绘图。", file=sys.stderr)
        else:
            plt.figure(figsize=(10, 10))
            # 绘制原始边界
            b_pts = np.array(boundary_points + [boundary_points[0]])
            plt.plot(b_pts[:, 0], b_pts[:, 1], 'k-', label=f'Boundary Area')
            plt.fill(b_pts[:, 0], b_pts[:, 1], alpha=0.1, color='gray')

            # 分色绘制各层航点
            for mission in all_missions_data:
                pts = mission['pts']
                if pts.size > 0:
                    plt.scatter(pts[:, 0], pts[:, 1], s=8, color=mission['color'], label=mission['label'])
                    plt.plot(pts[:, 0], pts[:, 1], lw=0.5, color=mission['color'], alpha=0.7)

            plt.gca().set_aspect('equal', adjustable='box')
            plt.title("UAV Flight Trajectories (cm)")
            plt.xlabel("X (cm)")
            plt.ylabel("Y (cm)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()


if __name__ == "__main__":
    main()
