# =============================================================================
# utils.py — 公共工具函数
# =============================================================================

import os
import cv2
import numpy as np
import config


def ensure_dirs():
    """确保 data/left, data/right, output 目录存在。"""
    for d in (config.LEFT_DIR, config.RIGHT_DIR, config.OUTPUT_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 标定参数 IO
# ---------------------------------------------------------------------------

def save_stereo_params(path: str, params: dict):
    """
    将双目标定参数写入 YAML 文件。
    params 应包含以下键（与 02_calibrate.py 输出一致）:
        K1, D1, K2, D2      — 左/右内参、畸变
        R, T, E, F          — 双目旋转/平移/本质矩阵/基础矩阵
        R1, R2, P1, P2, Q   — 校正旋转、投影矩阵、视差-深度映射矩阵
        image_width, image_height — 标定时使用的图像尺寸
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width",  params["image_width"])
    fs.write("image_height", params["image_height"])
    for key in ("K1", "D1", "K2", "D2", "R", "T", "E", "F",
                "R1", "R2", "P1", "P2", "Q"):
        fs.write(key, params[key])
    if "roi" in params:
        fs.write("roi", params["roi"])
    fs.release()
    print(f"[utils] 标定参数已保存到: {path}")


def load_stereo_params(path: str) -> dict:
    """从 YAML 文件读取双目标定参数，返回字典。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到标定参数文件: {path}\n"
            "请先运行 02_calibrate.py 完成标定。"
        )
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    params = {
        "image_width":  int(fs.getNode("image_width").real()),
        "image_height": int(fs.getNode("image_height").real()),
    }
    for key in ("K1", "D1", "K2", "D2", "R", "T", "E", "F",
                "R1", "R2", "P1", "P2", "Q"):
        params[key] = fs.getNode(key).mat()
    roi_node = fs.getNode("roi")
    params["roi"] = roi_node.mat().astype(int).ravel().tolist() if not roi_node.empty() else None
    fs.release()
    return params


# ---------------------------------------------------------------------------
# 校正映射
# ---------------------------------------------------------------------------

def build_rectify_maps(params: dict):
    """
    根据标定参数构建左右校正重映射表。
    返回:
        left_map1, left_map2, right_map1, right_map2  — 传给 cv2.remap 使用
        image_size  — (width, height) 元组
    """
    w = params["image_width"]
    h = params["image_height"]
    image_size = (w, h)

    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        params["K1"], params["D1"], params["R1"], params["P1"],
        image_size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        params["K2"], params["D2"], params["R2"], params["P2"],
        image_size, cv2.CV_16SC2
    )
    return left_map1, left_map2, right_map1, right_map2, image_size


# ---------------------------------------------------------------------------
# 视差有效性检查
# ---------------------------------------------------------------------------

def is_valid_disparity(disp_value: float, min_disp: int, num_disp: int,
                        min_valid: float = 1.0) -> bool:
    """
    判断某像素的视差值是否有效（非遮挡/未匹配区域）。
    StereoSGBM 输出已 /16 后的值传入即可。
    """
    return (disp_value > max(min_valid, min_disp) and
            disp_value < min_disp + num_disp)
