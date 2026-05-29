#!/usr/bin/env python3
# =============================================================================
# 02_calibrate.py — 双目相机标定
#
# 使用方法：
#   python 02_calibrate.py
#
# 流程：
#   1. 读取 data/left / data/right 中成对的棋盘格图像
#   2. 单目标定左右相机（得到内参 K 和畸变 D）
#   3. 双目标定（得到旋转 R、平移 T、本质矩阵 E、基础矩阵 F）
#   4. 立体校正（得到 R1/R2/P1/P2/Q）
#   5. 保存参数到 output/stereo_params.yml
#   6. 生成并保存一张校正后的对比图，供肉眼验证极线对齐
#
# 质量判断：
#   • 单目重投影误差 < 0.5 px 为优秀
#   • 双目 stereoCalibrate RMS  < 1.0 px 为可接受
#   • 校正图中，棋盘格对应点应落在同一水平线上
# =============================================================================

import os
import sys
import glob
import cv2
import numpy as np
import config
import utils


def load_image_pairs():
    """
    扫描 LEFT_DIR / RIGHT_DIR，返回成对的图像路径列表。
    按文件名排序后两两配对（同序号为一对）。
    """
    left_imgs  = sorted(glob.glob(os.path.join(config.LEFT_DIR,  "*.png")))
    right_imgs = sorted(glob.glob(os.path.join(config.RIGHT_DIR, "*.png")))

    if len(left_imgs) == 0 or len(right_imgs) == 0:
        print("[错误] data/left 或 data/right 目录中没有图像。")
        print("       请先运行 01_capture.py 采集标定图像。")
        sys.exit(1)

    if len(left_imgs) != len(right_imgs):
        print(f"[警告] 左({len(left_imgs)})右({len(right_imgs)})图像数量不一致，"
              "将使用最小数量。")
    n = min(len(left_imgs), len(right_imgs))
    pairs = list(zip(left_imgs[:n], right_imgs[:n]))
    print(f"[信息] 找到 {n} 对候选图像。")
    return pairs


def find_corners_batch(pairs):
    """
    对每对图像检测棋盘格角点，只保留两侧都成功检测的对。
    返回:
        obj_points  — 世界坐标列表 (N × corner_count × 3)
        left_pts    — 左图角点列表 (N × corner_count × 2)
        right_pts   — 右图角点列表
        image_size  — (width, height)，取第一张图的尺寸
    """
    board_size    = config.CHESSBOARD_SIZE
    square_size   = config.SQUARE_SIZE_MM

    # 构造世界坐标模板（Z=0 平面，单位 mm）
    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    crit = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        config.SUBPIX_CRITERIA[1],
        config.SUBPIX_CRITERIA[2]
    )
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    obj_points = []
    left_pts   = []
    right_pts  = []
    image_size = None
    skipped    = 0

    for i, (lp, rp) in enumerate(pairs):
        img_l = cv2.imread(lp)
        img_r = cv2.imread(rp)
        if img_l is None or img_r is None:
            print(f"  [跳过] 读取失败: {os.path.basename(lp)} / {os.path.basename(rp)}")
            skipped += 1
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])

        found_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, flags)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, flags)

        if found_l and found_r:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), crit)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), crit)
            obj_points.append(objp)
            left_pts.append(corners_l)
            right_pts.append(corners_r)
            print(f"  [{i:02d}] ✓  {os.path.basename(lp)}")
        else:
            tag_l = "✓" if found_l else "✗"
            tag_r = "✓" if found_r else "✗"
            print(f"  [{i:02d}] 左{tag_l} 右{tag_r}  {os.path.basename(lp)}  [跳过]")
            skipped += 1

    valid = len(obj_points)
    print(f"\n有效对: {valid}  跳过: {skipped}")
    if valid < 10:
        print(f"[警告] 有效标定对只有 {valid} 对，标定精度可能不足，建议至少 15 对。")
        if valid == 0:
            print("[错误] 没有任何有效对，无法标定。")
            sys.exit(1)

    return obj_points, left_pts, right_pts, image_size


def mono_calibrate(obj_points, img_points, image_size, name="相机"):
    """单目标定，返回 (K, D, rms)，并打印信息。"""
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    print(f"\n[单目标定 - {name}]  RMS 重投影误差: {rms:.4f} px")
    print(f"  内参矩阵 K:\n{K}")
    print(f"  畸变系数 D: {D.ravel()}")
    return K, D, rms


def stereo_calibrate(obj_points, left_pts, right_pts,
                     K1, D1, K2, D2, image_size):
    """双目标定（固定内参），返回 (R, T, E, F, rms)。"""
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    flags = (cv2.CALIB_FIX_INTRINSIC)

    rms, K1_, D1_, K2_, D2_, R, T, E, F = cv2.stereoCalibrate(
        obj_points, left_pts, right_pts,
        K1, D1, K2, D2,
        image_size,
        criteria=crit,
        flags=flags
    )
    print(f"\n[双目标定]  RMS 重投影误差: {rms:.4f} px")
    if rms > 1.0:
        print("  [警告] RMS > 1.0 px，建议重新采集更多高质量图像再标定。")
    elif rms < 0.5:
        print("  [优秀] RMS < 0.5 px，标定质量很好！")
    else:
        print("  [良好] RMS 在可接受范围内。")

    print(f"  旋转矩阵 R:\n{R}")
    print(f"  平移向量 T (mm): {T.ravel()}")
    baseline_mm = np.linalg.norm(T)
    print(f"  基线长度: {baseline_mm:.2f} mm")
    return R, T, E, F, rms


def stereo_rectify(K1, D1, K2, D2, R, T, image_size):
    """立体校正，返回 (R1, R2, P1, P2, Q)。"""
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0       # alpha=0: 裁剪到有效像素；alpha=1: 保留全部像素（含黑边）
    )
    print("\n[立体校正]")
    print(f"  P1: {P1}")
    print(f"  P2: {P2}")
    print(f"  Q 矩阵:\n{Q}")
    print(f"  左有效 ROI: {roi_l}  右有效 ROI: {roi_r}")
    return R1, R2, P1, P2, Q


def save_verification_image(params, pairs):
    """
    取第一对有效图像，做校正并并排保存，画水平参考线，方便肉眼检验极线对齐。
    """
    if not pairs:
        return

    img_l = cv2.imread(pairs[0][0])
    img_r = cv2.imread(pairs[0][1])
    if img_l is None or img_r is None:
        return

    lm1, lm2, rm1, rm2, _ = utils.build_rectify_maps(params)
    rect_l = cv2.remap(img_l, lm1, lm2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, rm1, rm2, cv2.INTER_LINEAR)

    canvas = np.hstack([rect_l, rect_r])
    h = canvas.shape[0]
    # 画等间距水平线
    for y in range(0, h, h // 12):
        cv2.line(canvas, (0, y), (canvas.shape[1], y), (0, 200, 0), 1)

    out_path = os.path.join(config.OUTPUT_DIR, "rectification_check.png")
    cv2.imwrite(out_path, canvas)
    print(f"\n[验证图] 已保存: {out_path}")
    print("  请打开该图，检查棋盘格对应点是否落在同一水平绿线上。")


def main():
    utils.ensure_dirs()

    print("=" * 60)
    print("  双目相机标定")
    print("=" * 60)
    print(f"  棋盘格内角点: {config.CHESSBOARD_SIZE}")
    print(f"  方格大小: {config.SQUARE_SIZE_MM} mm")
    print("=" * 60)

    # 1. 加载图像对
    pairs = load_image_pairs()

    # 2. 检测角点
    print("\n--- 角点检测 ---")
    obj_points, left_pts, right_pts, image_size = find_corners_batch(pairs)
    print(f"图像尺寸: {image_size[0]}x{image_size[1]}")

    # 3. 单目标定
    print("\n--- 单目标定 ---")
    K1, D1, rms_l = mono_calibrate(obj_points, left_pts,  image_size, "左相机")
    K2, D2, rms_r = mono_calibrate(obj_points, right_pts, image_size, "右相机")

    # 4. 双目标定
    print("\n--- 双目标定 ---")
    R, T, E, F, rms_stereo = stereo_calibrate(
        obj_points, left_pts, right_pts, K1, D1, K2, D2, image_size
    )

    # 5. 立体校正
    R1, R2, P1, P2, Q = stereo_rectify(K1, D1, K2, D2, R, T, image_size)

    # 6. 保存参数
    params = {
        "image_width":  image_size[0],
        "image_height": image_size[1],
        "K1": K1, "D1": D1,
        "K2": K2, "D2": D2,
        "R":  R,  "T":  T,
        "E":  E,  "F":  F,
        "R1": R1, "R2": R2,
        "P1": P1, "P2": P2,
        "Q":  Q,
    }
    utils.save_stereo_params(config.PARAM_FILE, params)

    # 7. 生成校正验证图
    save_verification_image(params, pairs)

    print("\n" + "=" * 60)
    print("  标定完成！")
    print(f"  双目 RMS: {rms_stereo:.4f} px")
    print(f"  参数文件: {config.PARAM_FILE}")
    print("  下一步：运行  python 03_measure.py  进行测距")
    print("=" * 60)


if __name__ == "__main__":
    main()
