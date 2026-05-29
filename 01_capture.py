#!/usr/bin/env python3
# =============================================================================
# 01_capture.py — 双目棋盘格图像采集
#
# 相机类型：一体式双目模组，单 USB，输出左右拼接宽图（2560x720）
# 脚本自动从中间切割得到左右各 1280x720 的图像。
#
# 使用方法：
#   python 01_capture.py
#
# 操作键：
#   s   — 保存当前左右帧（仅当两侧都检测到棋盘格时才保存）
#         若想强制保存（不检测），运行时加 --force 参数
#   q   — 退出
# =============================================================================

import sys
import os
import glob
import cv2
import numpy as np
import config
import utils


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(config.CAM_ID)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {config.CAM_ID}，请检查设备连接和索引。\n"
            "（可在 config.py 中修改 CAM_ID）"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    # 丢弃前 10 帧预热，避免首帧全黑
    for _ in range(10):
        cap.read()
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != config.FRAME_WIDTH or actual_h != config.FRAME_HEIGHT:
        print(f"  [警告] 实际分辨率 {actual_w}x{actual_h} 与配置不符，"
              f"请将 config.py 中 FRAME_WIDTH/HEIGHT 改为 {actual_w}x{actual_h}。")
    return cap


def split_frame(frame):
    """从拼接帧中切割出左右图像。"""
    mid = frame.shape[1] // 2
    return frame[:, :mid].copy(), frame[:, mid:].copy()


def next_index(directory: str) -> int:
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


def detect_corners(frame_gray, board_size):
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
    vis = frame.copy()
    color = (0, 220, 0) if found else (0, 0, 220)
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, 4)
    if found and corners is not None:
        cv2.drawChessboardCorners(vis, board_size, corners, found)
    return vis


def main():
    force_save = "--force" in sys.argv
    utils.ensure_dirs()

    eye_w = config.FRAME_WIDTH // 2   # 单眼宽度
    eye_h = config.FRAME_HEIGHT

    print("=" * 60)
    print("  双目标定图像采集工具")
    print("=" * 60)
    print(f"  相机索引: {config.CAM_ID}  拼接输出: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print(f"  单眼分辨率: {eye_w}x{eye_h}")
    print(f"  棋盘格内角点: {config.CHESSBOARD_SIZE}  方格大小: {config.SQUARE_SIZE_MM} mm")
    print(f"  强制保存: {'是（--force）' if force_save else '否'}")
    print("-" * 60)
    print("  操作: [s] 保存  [q] 退出")
    print("=" * 60)

    try:
        cap = open_camera()
    except RuntimeError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    board_size  = config.CHESSBOARD_SIZE
    saved_count = next_index(config.LEFT_DIR)
    print(f"[信息] 目录中已有 {saved_count} 对图像，将从序号 {saved_count:02d} 继续。\n")

    # 显示用缩放宽度（单眼）
    disp_w = 640
    scale  = disp_w / eye_w
    disp_h = int(eye_h * scale)

    while True:
        ret, raw = cap.read()
        if not ret or raw is None:
            print("[警告] 取帧失败，跳过...")
            continue

        frame_l, frame_r = split_frame(raw)

        # 缩放后检测（快速预览）
        small_l = cv2.resize(frame_l, (disp_w, disp_h))
        small_r = cv2.resize(frame_r, (disp_w, disp_h))
        gray_sl = cv2.cvtColor(small_l, cv2.COLOR_BGR2GRAY)
        gray_sr = cv2.cvtColor(small_r, cv2.COLOR_BGR2GRAY)
        found_l, corners_l = detect_corners(gray_sl, board_size)
        found_r, corners_r = detect_corners(gray_sr, board_size)
        both_found = found_l and found_r

        small_l = draw_preview(small_l, found_l, corners_l, board_size)
        small_r = draw_preview(small_r, found_r, corners_r, board_size)

        # 状态栏
        status_color = (0, 220, 0) if both_found else (0, 180, 255)
        status_text = (
            f"已采集: {saved_count} 对   "
            + ("双侧检出 ✓  按[s]保存" if both_found
               else ("左" if not found_l else "") +
                    ("右" if not found_r else "") + " 未检出")
        )
        status_bar = np.zeros((36, disp_w * 2, 3), dtype=np.uint8)
        cv2.putText(status_bar, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        img_row = np.hstack([small_l, small_r])
        cv2.line(img_row, (disp_w, 0), (disp_w, disp_h - 1), (200, 200, 200), 2)

        label_bar = np.zeros((24, disp_w * 2, 3), dtype=np.uint8)
        cv2.putText(label_bar, "LEFT",  (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(label_bar, "RIGHT", (disp_w + 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        canvas = np.vstack([status_bar, img_row, label_bar])
        cv2.imshow("Stereo Capture  [s]Save  [q]Quit", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if both_found or force_save:
                # 全分辨率再确认一次
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                ok_l, _ = detect_corners(gray_l, board_size)
                ok_r, _ = detect_corners(gray_r, board_size)
                if not force_save and not (ok_l and ok_r):
                    print("  [跳过] 全分辨率检测未通过，请重试。")
                    continue
                idx = saved_count
                l_path = os.path.join(config.LEFT_DIR,  f"left_{idx:02d}.png")
                r_path = os.path.join(config.RIGHT_DIR, f"right_{idx:02d}.png")
                cv2.imwrite(l_path, frame_l)
                cv2.imwrite(r_path, frame_r)
                saved_count += 1
                note = "" if both_found else " [强制]"
                print(f"  已保存{note}: left_{idx:02d}.png  right_{idx:02d}.png  (共 {saved_count} 对)")
                if saved_count >= 20:
                    print("  [提示] 已采集 20 对，可运行 02_calibrate.py 进行标定。")
            else:
                print("  [跳过] 棋盘格未在两侧都检出，请调整角度再试（或加 --force 强制保存）。")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n采集结束，共保存 {saved_count} 对标定图像。")
    if saved_count < 15:
        print(f"[建议] 当前只有 {saved_count} 对，建议至少采集 15 对以保证标定精度。")
    else:
        print("下一步：运行  python 02_calibrate.py  进行双目标定。")


if __name__ == "__main__":
    main()
