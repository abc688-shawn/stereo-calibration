#!/usr/bin/env python3
# =============================================================================
# 03_measure.py — 双目实时测距
#
# 使用方法：
#   python 03_measure.py
#
# 操作：
#   • 鼠标左键点击左视图画面的任意像素 → 打印该点的三维坐标与距离
#   • 鼠标右键       → 清除画面上的距离标注
#   • [q]           → 退出
#   • [d]           → 切换是否显示视差图（节省性能时可关）
#   • [r]           → 重置点击记录
#
# 说明：
#   距离单位与标定时 SQUARE_SIZE_MM 单位一致（默认毫米）。
#   测距精度受标定质量、视差匹配质量影响。
#   近距离（< 0.3 m）或远距离（> 基线×100 左右）测距误差会增大。
# =============================================================================

import sys
import cv2
import numpy as np
import config
import utils


# ---------------------------------------------------------------------------
# 全局状态（由鼠标回调和主循环共享）
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.points3d     = None          # 当前帧 reprojectImageTo3D 结果
        self.disparity    = None          # 当前帧视差图（/16 后，float32）
        self.click_info   = []            # 点击记录: list of (x, y, dist_mm, z_mm)
        self.show_disp    = True          # 是否显示视差图窗口
        self.frame_size   = None          # (width, height)


state = AppState()


def mouse_callback(event, x, y, flags, param):
    """鼠标回调：左键点击查距离，右键清除。"""
    if event == cv2.EVENT_LBUTTONDOWN:
        if state.points3d is None:
            return

        h, w = state.points3d.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        pt3d = state.points3d[y, x]   # (X, Y, Z) in mm
        X, Y, Z = float(pt3d[0]), float(pt3d[1]), float(pt3d[2])

        # 检查有效性
        disp_val = state.disparity[y, x] if state.disparity is not None else 0
        valid = utils.is_valid_disparity(
            disp_val,
            config.SGBM_MIN_DISPARITY,
            config.SGBM_NUM_DISPARITIES
        ) and np.isfinite(Z) and 0 < Z < 100_000

        if valid:
            dist_mm  = np.sqrt(X**2 + Y**2 + Z**2)
            z_mm     = Z
            dist_m   = dist_mm  / 1000.0
            z_m      = z_mm     / 1000.0
            state.click_info.append((x, y, dist_mm, z_mm))
            print(f"  点 ({x:4d}, {y:4d})  |  Z(深度)={z_m:.3f} m   "
                  f"欧氏距离={dist_m:.3f} m   "
                  f"[X={X/1000:.3f}m  Y={Y/1000:.3f}m  Z={Z/1000:.3f}m]")
        else:
            print(f"  点 ({x:4d}, {y:4d})  |  视差无效（遮挡/平滑区域/超出测量范围）")
            state.click_info.append((x, y, None, None))

    elif event == cv2.EVENT_RBUTTONDOWN:
        state.click_info.clear()
        print("  [已清除] 点击记录已清空。")


def draw_annotations(frame, click_info):
    """在左视图上绘制已点击位置和距离文字。"""
    vis = frame.copy()
    for x, y, dist_mm, z_mm in click_info:
        if dist_mm is not None:
            label = f"Z={z_mm/1000:.2f}m  D={dist_mm/1000:.2f}m"
            color = (0, 255, 128)
        else:
            label = "无效"
            color = (0, 80, 255)

        cv2.circle(vis, (x, y), 6, color, -1)
        cv2.circle(vis, (x, y), 8, (255, 255, 255), 1)

        # 文字背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx = min(x + 12, vis.shape[1] - tw - 5)
        ty = max(y - 8, th + 5)
        cv2.rectangle(vis, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2),
                      (20, 20, 20), -1)
        cv2.putText(vis, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return vis


def build_sgbm():
    """构造 StereoSGBM 匹配器。"""
    sgbm = cv2.StereoSGBM_create(
        minDisparity    = config.SGBM_MIN_DISPARITY,
        numDisparities  = config.SGBM_NUM_DISPARITIES,
        blockSize       = config.SGBM_BLOCK_SIZE,
        P1              = config.SGBM_P1,
        P2              = config.SGBM_P2,
        disp12MaxDiff   = config.SGBM_DISP12MAX_DIFF,
        uniquenessRatio = config.SGBM_UNIQUENESS_RATIO,
        speckleWindowSize = config.SGBM_SPECKLE_WIN_SIZE,
        speckleRange    = config.SGBM_SPECKLE_RANGE,
        mode            = config.SGBM_MODE,
    )
    return sgbm


def open_cameras():
    cap_l = cv2.VideoCapture(config.LEFT_CAM_ID)
    cap_r = cv2.VideoCapture(config.RIGHT_CAM_ID)
    for cap, cid in [(cap_l, config.LEFT_CAM_ID), (cap_r, config.RIGHT_CAM_ID)]:
        if not cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头 {cid}。请检查设备连接，并确认 config.py 中的索引正确。"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap_l, cap_r


def main():
    utils.ensure_dirs()

    print("=" * 60)
    print("  双目实时测距")
    print("=" * 60)
    print("  操作:")
    print("    鼠标左键  — 点击左视图查询该点距离")
    print("    鼠标右键  — 清除所有标注")
    print("    [d]      — 开/关视差图窗口")
    print("    [r]      — 清除点击记录")
    print("    [q]      — 退出")
    print("=" * 60)

    # 加载标定参数
    try:
        params = utils.load_stereo_params(config.PARAM_FILE)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    lm1, lm2, rm1, rm2, image_size = utils.build_rectify_maps(params)
    Q = params["Q"]

    # 打开相机
    try:
        cap_l, cap_r = open_cameras()
    except RuntimeError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    sgbm = build_sgbm()

    # 窗口设置
    WIN_LEFT = "Left Rectified  [Click to measure]"
    WIN_DISP = "Disparity Map"
    cv2.namedWindow(WIN_LEFT, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_LEFT, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    cv2.setMouseCallback(WIN_LEFT, mouse_callback)
    cv2.namedWindow(WIN_DISP, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_DISP, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    print("\n[信息] 实时测距中，点击左视图画面任意位置查询距离...\n")

    frame_count = 0
    while True:
        # 同步取帧
        ret_l = cap_l.grab()
        ret_r = cap_r.grab()
        if not ret_l or not ret_r:
            print("[警告] 取帧失败，跳过...")
            continue

        _, frame_l = cap_l.retrieve()
        _, frame_r = cap_r.retrieve()

        # 立体校正（去畸变 + 极线对齐）
        rect_l = cv2.remap(frame_l, lm1, lm2, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, rm1, rm2, cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        # 视差计算（每帧都算，保证点击时数据最新）
        disp_raw = sgbm.compute(gray_l, gray_r)   # int16, *16
        disp     = disp_raw.astype(np.float32) / 16.0

        # 三维重投影
        points3d = cv2.reprojectImageTo3D(disp, Q)

        # 更新全局状态（供鼠标回调读取）
        state.points3d  = points3d
        state.disparity = disp

        # 绘制左视图标注
        vis_left = draw_annotations(rect_l, state.click_info)

        # HUD 信息
        cv2.rectangle(vis_left, (0, 0), (vis_left.shape[1], 28), (20, 20, 20), -1)
        cv2.putText(vis_left, "LEFT RECTIFIED  |  Click to measure distance",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 180), 1)

        cv2.imshow(WIN_LEFT, vis_left)

        # 视差图可视化
        if state.show_disp:
            # 归一化并着色
            disp_vis = disp.copy()
            valid_mask = (disp_vis > config.SGBM_MIN_DISPARITY)
            if valid_mask.any():
                d_min = disp_vis[valid_mask].min()
                d_max = disp_vis[valid_mask].max()
                disp_vis = np.clip((disp_vis - d_min) / (d_max - d_min + 1e-6), 0, 1)
            disp_u8  = (disp_vis * 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)
            # TURBO: 蓝=近, 红=远；无效区域为深色
            disp_color[~valid_mask] = (20, 20, 20)

            cv2.rectangle(disp_color, (0, 0), (disp_color.shape[1], 28), (20, 20, 20), -1)
            cv2.putText(disp_color, "DISPARITY  (blue=near, red=far, dark=invalid)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(WIN_DISP, disp_color)

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            state.show_disp = not state.show_disp
            if not state.show_disp:
                cv2.destroyWindow(WIN_DISP)
            print(f"  视差图: {'显示' if state.show_disp else '关闭'}")
        elif key == ord('r'):
            state.click_info.clear()
            print("  [已清除] 点击记录已清空。")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print("\n测距程序已退出。")


if __name__ == "__main__":
    main()
