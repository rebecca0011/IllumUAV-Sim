# -*- coding: utf-8 -*-
"""
uav_station_grid_cm.py
按摄影测量关系（厘米制）从相机参数与重叠度求摄站间距，依次叠加生成网格（可蛇形排序），
并可选裁剪到边界多边形内，导出 UE 友好的 CSV (X,Y,Z)。

更新要点：
- Z 不再固定为 0，而是导出 Z = base_z + H（单位：cm），满足“Z=--H”的要求。
- 如地面不在 UE 的 Z=0，可通过 --base_z 指定地面基准高度（cm）。

用法（示例）:
  python uav_station_grid_cm.py --use_default_boundary --front 0.8 --side 0.7 --H 10000 --base_z 0 --csv out.csv --plot
"""

from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional
import argparse
import numpy as np
import sys
import os

# ----------------------- 可选依赖 -----------------------
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

# try:
#     import matplotlib.pyplot as plt
#     MPL_OK = True
# except Exception:
#     MPL_OK = False
# ------------------------------------------------------


# -------------------- 数据模型与核心计算 --------------------
@dataclass
class CameraModelCM:
    """相机/成像参数（单位均为厘米或像素）。"""
    f_cm: float                 # 焦距 f（cm）
    sensor_w_cm: float          # 传感器宽（cm）
    sensor_h_cm: float          # 传感器高（cm）
    Nx: int                     # 水平方向像素数
    Ny: int                     # 垂直方向像素数
    H_cm: float                 # 飞行高度 H（cm，地面到相机主点）

    def gsd_cm_per_px(self) -> float:
        """GSD = (H * p) / f，其中 p = sensor_w / Nx（cm/px）。"""
        p_cm = self.sensor_w_cm / float(self.Nx)
        return (self.H_cm * p_cm) / self.f_cm

    def footprint_cm(self) -> Tuple[float, float]:
        """单张影像在地面的覆盖尺寸（cm）"""
        gsd = self.gsd_cm_per_px()
        Wx = gsd * self.Nx   # 地面覆盖宽（横向）
        Wy = gsd * self.Ny   # 地面覆盖高（纵向）
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
    d_along = Wy * (1.0 - front_overlap)   # 前方重叠 -> 沿程步长
    d_cross = Wx * (1.0 - side_overlap)    # 旁向重叠 -> 旁向步长
    return gsd, d_along, d_cross


def _bbox_from_polygon_cm(boundary_points_cm: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in boundary_points_cm]
    ys = [p[1] for p in boundary_points_cm]
    return min(xs), min(ys), max(xs), max(ys)


def generate_stations_grid_cm(
    boundary_points_cm: List[Tuple[float, float]],
    d_along_cm: float,
    d_cross_cm: float,
    origin: Literal["bbox_min", "bbox_max", "custom"] = "bbox_min",
    custom_origin_cm: Optional[Tuple[float, float]] = None,
    snake: bool = True,
    clip_to_polygon: bool = True
) -> np.ndarray:
    """
    采用“依次叠加”的方式生成网格（厘米）。
    - origin = "bbox_min": 以边界外接框左下角为基准点（推荐）
    - origin = "bbox_max": 以外接框右上角为基准点
    - origin = "custom":   使用 custom_origin_cm 作为基准点
    - snake: 行走蛇形布设（每列交替反向），便于拟合往返航线
    - clip_to_polygon: 是否裁剪到边界多边形（需要 shapely；关闭可获得最快速度）
    返回：N×2 的数组，每行 [x_cm, y_cm]
    """
    minx, miny, maxx, maxy = _bbox_from_polygon_cm(boundary_points_cm)

    if origin == "bbox_min":
        x0, y0 = minx, miny
    elif origin == "bbox_max":
        x0, y0 = maxx, maxy
    elif origin == "custom":
        if custom_origin_cm is None:
            raise ValueError("origin='custom' 时必须提供 custom_origin_cm")
        x0, y0 = custom_origin_cm
    else:
        raise ValueError("origin 取值必须为 'bbox_min' | 'bbox_max' | 'custom'")

    # 列/行数量（向下取整后 +1，保证覆盖外接框）
    n_cols = int(np.floor((maxx - minx) / max(d_cross_cm, 1e-6))) + 1
    n_rows = int(np.floor((maxy - miny) / max(d_along_cm, 1e-6))) + 1

    # 从基准点“叠加”
    if origin == "bbox_max":
        xs = x0 - np.arange(0, n_cols) * d_cross_cm
        ys = y0 - np.arange(0, n_rows) * d_along_cm
    else:
        xs = x0 + np.arange(0, n_cols) * d_cross_cm
        ys = y0 + np.arange(0, n_rows) * d_along_cm

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])

    # 先做 bbox 裁剪（避免少量越界）
    in_bbox = (
        (pts[:, 0] >= minx) & (pts[:, 0] <= maxx) &
        (pts[:, 1] >= miny) & (pts[:, 1] <= maxy)
    )
    pts = pts[in_bbox]

    # 可选：裁剪到多边形内部
    if clip_to_polygon:
        if not SHAPELY_OK:
            raise RuntimeError("clip_to_polygon=True 需要 shapely；请安装 shapely 或将其设为 False")
        poly = Polygon(boundary_points_cm)
        prepared = prep(poly)
        mask = np.fromiter((prepared.contains(Point(xy[0], xy[1])) for xy in pts), dtype=bool, count=len(pts))
        pts = pts[mask]

    # 蛇形排序（按列蛇形：X 分组后交替反序 Y）
    if snake and pts.size > 0:
        order = np.lexsort((pts[:, 1], pts[:, 0]))  # 先按 x，再按 y 排
        sorted_pts = pts[order]
        unique_xs = np.unique(sorted_pts[:, 0])
        snake_list = []
        for i, vx in enumerate(unique_xs):
            col = sorted_pts[sorted_pts[:, 0] == vx]
            if i % 2 == 1:
                col = col[::-1]
            snake_list.append(col)
        pts = np.vstack(snake_list)

    return pts


def export_csv_for_ue(path_csv: str, pts_cm: np.ndarray, z_cm_value: float, encoding: str = "utf-8"):
    """
    导出为 UE 友好的 CSV：列名 X,Y,Z（单位：cm）。
    Z 由调用方传入（应为 base_z + H）。
    """
    import pandas as pd
    z_col = np.full((pts_cm.shape[0],), z_cm_value, dtype=float)
    df = pd.DataFrame({
        "X": pts_cm[:, 0],
        "Y": pts_cm[:, 1],
        "Z": z_col
    })
    df.to_csv(path_csv, index=False, encoding=encoding)


# -------------------------- CLI 与主流程 --------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UAV 摄影测量网格生成（厘米制，UE 友好 CSV，Z=base_z+H）")
    # 相机/高度（厘米）
    p.add_argument("--f", type=float, default=3.5, help="焦距 f（cm），默认 3.5（=35mm）")
    p.add_argument("--sw", type=float, default=3.6, help="传感器宽（cm），默认 3.6（=36mm）")
    p.add_argument("--sh", type=float, default=2.4, help="传感器高（cm），默认 2.4（=24mm）")
    p.add_argument("--Nx", type=int, default=6000, help="横向像素数 Nx，默认 6000")
    p.add_argument("--Ny", type=int, default=4000, help="垂直像素数 Ny，默认 4000")
    p.add_argument("--H",  type=float, default=25000.0, help="飞行高度 H（cm），默认 10000 (=100m)")
    p.add_argument("--base_z", type=float, default=0.0, help="地面基准高度（cm），最终 Z=base_z+H，默认 0")

    # 重叠度
    p.add_argument("--front", type=float, default=0.80, help="前方重叠 Fo（0~1），默认 0.80")
    p.add_argument("--side",  type=float, default=0.70, help="旁向重叠 So（0~1），默认 0.70")

    # 边界与网格
    p.add_argument("--origin", choices=["bbox_min", "bbox_max", "custom"], default="bbox_min",
                   help="叠加基准点，默认 bbox_min（外接框左下角）")
    p.add_argument("--origin_x", type=float, default=None, help="自定义原点 X（cm），当 origin=custom 时有效")
    p.add_argument("--origin_y", type=float, default=None, help="自定义原点 Y（cm），当 origin=custom 时有效")
    p.add_argument("--no_snake", action="store_true", help="关闭蛇形排序（默认开启）")
    p.add_argument("--no_clip", action="store_true", help="不裁剪到多边形（更快）")

    # 导出与展示
    p.add_argument("--csv", type=str, default="drone_H250.csv", help="输出 CSV 路径，默认 drone_stations_cm.csv")
    p.add_argument("--plot", action="store_true",default=True, help="显示示意图（需要 matplotlib）")

    # 内置示例边界（若需要可改为文件输入）
    p.add_argument("--use_default_boundary", action="store_true",
                   help="使用内置示例边界点（厘米）：[(-91500,16150), (-38750,-62100), (39250,-62900), (78750,15600), (26050,68050)]")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) 边界点（厘米）
    if args.use_default_boundary:
        boundary_points = [(-91500, 16150), (-38750, -62100), (39250, -62900), (78750, 15600), (26050, 68050)]
    else:
        # 这里你可以改为从文件或其他接口读取；为简洁起见，默认用示例
        print("未指定 --use_default_boundary，默认使用示例边界。", file=sys.stderr)
        boundary_points = [(-91500, 16150), (-38750, -62100), (39250, -62900), (78750, 15600), (26050, 68050)]

    # 2) 相机与高度
    cam = CameraModelCM(
        f_cm=args.f,
        sensor_w_cm=args.sw,
        sensor_h_cm=args.sh,
        Nx=args.Nx,
        Ny=args.Ny,
        H_cm=args.H
    )

    # 3) 计算 GSD 与摄站步长
    GSD, d_along_cm, d_cross_cm = station_spacing_cm(cam, front_overlap=args.front, side_overlap=args.side)
    print(f"[INFO] GSD = {GSD:.6f} cm/px")
    print(f"[INFO] 步长：沿程 Δalong = {d_along_cm:.3f} cm, 旁向 Δcross = {d_cross_cm:.3f} cm")

    # 4) 生成网格
    if args.origin == "custom":
        if args.origin_x is None or args.origin_y is None:
            print("[ERROR] origin=custom 时必须提供 --origin_x 与 --origin_y", file=sys.stderr)
            sys.exit(1)
        custom_origin = (args.origin_x, args.origin_y)
    else:
        custom_origin = None

    try:
        pts_cm = generate_stations_grid_cm(
            boundary_points_cm=boundary_points,
            d_along_cm=d_along_cm,
            d_cross_cm=d_cross_cm,
            origin=args.origin,
            custom_origin_cm=custom_origin,
            snake=(not args.no_snake),
            clip_to_polygon=(not args.no_clip)
        )
    except RuntimeError as e:
        print(f"[WARN] {e}\n[WARN] 自动切换为不裁剪模式（--no_clip）。", file=sys.stderr)
        pts_cm = generate_stations_grid_cm(
            boundary_points_cm=boundary_points,
            d_along_cm=d_along_cm,
            d_cross_cm=d_cross_cm,
            origin=args.origin,
            custom_origin_cm=custom_origin,
            snake=(not args.no_snake),
            clip_to_polygon=False
        )

    print(f"[INFO] 生成站点数: {len(pts_cm)}")

    # 5) 导出 CSV —— Z = base_z + H
    z_value = args.base_z + args.H
    export_csv_for_ue(args.csv, pts_cm, z_cm_value=z_value)
    print(f"[OK] 已导出 UE CSV: {os.path.abspath(args.csv)}（Z = base_z + H = {args.base_z} + {args.H}）")

    # 6) 可视化（可选）
    if args.plot:
        if not MPL_OK:
            print("[WARN] 未安装 matplotlib，无法绘图。", file=sys.stderr)
        else:
            xs = [p[0] for p in boundary_points]
            ys = [p[1] for p in boundary_points]
            xs.append(boundary_points[0][0])
            ys.append(boundary_points[0][1])

            plt.figure()
            plt.fill(xs, ys, alpha=0.3, label="Boundary (cm)")
            if pts_cm.size > 0:
                plt.scatter(pts_cm[:, 0], pts_cm[:, 1], s=8, label=f"Stations (cm)  Z={z_value:.0f}cm")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title("UAV Stations (cm)")
            plt.xlabel("X (cm)")
            plt.ylabel("Y (cm)")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    main()
