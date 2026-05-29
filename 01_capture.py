#!/usr/bin/env python3
# =============================================================================
# 01_capture.py — 双目棋盘格图像采集
#
# 使用方法：
#   python 01_capture.py
#
# 操作键：
#   s   — 保存当前左右帧（仅当两侧都检测到棋盘格时才保存，以保证质量）
#         若想强制保存（不检测），运行时加 --force 参数
#   q   — 退出
#
# 建议：
#   • 采集 15~25 对，覆盖画面不同位置（左/中/右、上/中/下）
#   • 变换标定板姿态（倾斜、旋转）
#   • 保持标定板在两个画面中都完整可见
#   • 避免运动模糊：移动后等一秒再按 s
# =============================================================================

import sys
import os
import glob
import cv2
import numpy as np
import config
import utils


def open_camera(cam_id: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {cam_id}，请检查设备连接和索引。\n"
            "（可在 config.py 中修改 LEFT_CAM_ID / RIGHT_CAM_ID）"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # 关闭自动曝光以减少左右曝光差异（部分相机不支持，忽略失败）
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    return cap


def next_index(directory: str) -> int:
    """自动推算下一张图的序号（从已有文件中找最大序号+1）。"""
    existing = glob.glob(os.path.join(directory, "*.png"))
    if not existing:
        return 0
    nums = []
    for f in existing:
        try:
            nums.append(int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 0


def detect_corners(frame_gray, board_size, flags=None):
    """检测棋盘格角点，返回 (found, corners)。"""
    if flags is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(frame_gray, board_size, flags)
    if found:
        crit = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            config.SUBPIX_CRITERIA[1],
            config.SUBPIX_CRITERIA[2]
        )
        corners = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), crit)
    return found, corners


def draw_preview(frame, found, corners, board_size):
    """在帧上叠加棋盘格检测结果和边框颜色（绿=检出，红=未检出）。"""
    vis = frame.copy()
    color = (0, 220, 0) if found else (0, 0, 220)
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, 4)
    if found and corners is not None:
        cv2.drawChessboardCorners(vis, board_size, corners, found)
    return vis


def main():
    force_save = "--force" in sys.argv

    utils.ensure_dirs()

    print("=" * 60)
    print("  双目标定图像采集工具")
    print("=" * 60)
    print(f"  左相机: {config.LEFT_CAM_ID}  右相机: {config.RIGHT_CAM_ID}")
    print(f"  分辨率: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print(f"  棋盘格内角点: {config.CHESSBOARD_SIZE}  方格大小: {config.SQUARE_SIZE_MM} mm")
    print(f"  强制保存模式: {'是（--force）' if force_save else '否（仅保存双侧检出的帧）'}")
    print("-" * 60)
    print("  操作: [s] 保存  [q] 退出")
    print("=" * 60)

    # 打开相机
    try:
        cap_l = open_camera(config.LEFT_CAM_ID,  config.FRAME_WIDTH, config.FRAME_HEIGHT)
        cap_r = open_camera(config.RIGHT_CAM_ID, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    except RuntimeError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    board_size  = config.CHESSBOARD_SIZE
    saved_count = next_index(config.LEFT_DIR)
    print(f"[信息] 目录中已有 {saved_count} 对图像，将从序号 {saved_count:02d} 继续。\n")

    while True:
        # 同步取帧：先 grab 再 retrieve，减少时间差
        ret_l = cap_l.grab()
        ret_r = cap_r.grab()
        if not ret_l or not ret_r:
            print("[警告] 取帧失败，跳过...")
            continue

        _, frame_l = cap_l.retrieve()
        _, frame_r = cap_r.retrieve()

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # 棋盘格检测
        found_l, corners_l = detect_corners(gray_l, board_size)
        found_r, corners_r = detect_corners(gray_r, board_size)
        both_found = found_l and found_r

        # 可视化叠加
        vis_l = draw_preview(frame_l, found_l, corners_l, board_size)
        vis_r = draw_preview(frame_r, found_r, corners_r, board_size)

        # 缩放为 640 宽（保持比例）用于显示，避免窗口过大
        disp_w = 640
        scale  = disp_w / config.FRAME_WIDTH
        disp_h = int(config.FRAME_HEIGHT * scale)
        small_l = cv2.resize(vis_l, (disp_w, disp_h))
        small_r = cv2.resize(vis_r, (disp_w, disp_h))

        # 拼接左右画面
        canvas = np.hstack([small_l, small_r])

        # 顶部状态栏
        status_color = (0, 220, 0) if both_found else (0, 180, 255)
        status_text  = (
            f"已采集: {saved_count} 对   "
            + ("双侧检出 ✓  按[s]保存" if both_found
               else ("左" if not found_l else "") +
                    ("右" if not found_r else "") + " 未检出")
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (30, 30, 30), -1)
        cv2.putText(canvas, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 中间分割线
        cv2.line(canvas, (disp_w, 0), (disp_w, disp_h + 36), (200, 200, 200), 2)
        cv2.putText(canvas, "LEFT",  (10,  disp_h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(canvas, "RIGHT", (disp_w + 10, disp_h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        canvas_full = np.zeros((disp_h + 50, disp_w * 2, 3), dtype=np.uint8)
        canvas_full[:36, :] = canvas[:36, :]
        canvas_full[36:36 + disp_h, :] = canvas[36:, :]

        cv2.imshow("Stereo Capture  [s]Save  [q]Quit", canvas_full)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if both_found or force_save:
                idx = saved_count
                l_path = os.path.join(config.LEFT_DIR,  f"left_{idx:02d}.png")
                r_path = os.path.join(config.RIGHT_DIR, f"right_{idx:02d}.png")
                cv2.imwrite(l_path, frame_l)
                cv2.imwrite(r_path, frame_r)
                saved_count += 1
                note = "" if both_found else " [强制]"
                print(f"  已保存{note}: {os.path.basename(l_path)}  {os.path.basename(r_path)}"
                      f"  (共 {saved_count} 对)")
                if saved_count >= 20:
                    print("  [提示] 已采集 20 对，数量充足，可运行 02_calibrate.py 进行标定。")
            else:
                print("  [跳过] 棋盘格未在两侧都检出，请调整角度再试（或加 --force 强制保存）。")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print(f"\n采集结束，共保存 {saved_count} 对标定图像。")
    if saved_count < 15:
        print(f"[建议] 当前只有 {saved_count} 对，建议至少采集 15 对以保证标定精度。")
    else:
        print("下一步：运行  python 02_calibrate.py  进行双目标定。")


if __name__ == "__main__":
    main()
