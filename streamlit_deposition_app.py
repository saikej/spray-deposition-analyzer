import io
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False



import streamlit as st
import hmac

# ========== 密码验证 ==========
def check_password():
    """返回 True 表示用户已通过验证"""
    
    def login_form():
        with st.form("credentials"):
            st.text_input("用户名", key="username")
            st.text_input("密码", type="password", key="password")
            st.form_submit_button("登录", on_click=password_entered)

    def password_entered():
        """检查密码是否正确"""
        # 从 secrets 中读取用户名和密码
        if st.secrets["general"]["username"] == st.session_state["username"]:
            if hmac.compare_digest(
                st.session_state["password"],
                st.secrets["general"]["password"]
            ):
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        else:
            st.session_state["password_correct"] = False

    # 初始化状态
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    # 显示登录表单或返回验证状态
    if not st.session_state["password_correct"]:
        login_form()
        return False
    else:
        return True

# 验证通过后才运行主程序
if not check_password():
    st.stop()




A4_PORTRAIT_MM = (210.0, 297.0)
A4_LANDSCAPE_MM = (297.0, 210.0)


@dataclass
class ROIItem:
    box: np.ndarray
    center: Tuple[float, float]
    width_px: float
    height_px: float
    area_px: float
    source: str = "auto"
    row: int = 0
    col: int = 0
    label: str = ""
    roi_image: Optional[np.ndarray] = None
    roi_mask: Optional[np.ndarray] = None


@dataclass
class AnalysisParams:
    dpi: float
    px_per_mm: float
    min_spot_area_px: int
    inner_margin_pct: float
    deposition_thresh: Optional[float]
    use_watershed: bool
    droplet_mode: str
    diameter_model: str
    power_a: float
    power_b: float
    area_a: float
    area_b: float
    high_coverage_warning_pct: float



def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_image(uploaded_file) -> np.ndarray:
    pil_img = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def crop_by_percent(img: np.ndarray, left_pct: float, right_pct: float, top_pct: float, bottom_pct: float) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(w * left_pct / 100.0)
    x1 = int(w * (1.0 - right_pct / 100.0))
    y0 = int(h * top_pct / 100.0)
    y1 = int(h * (1.0 - bottom_pct / 100.0))
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    return img[y0:y1, x0:x1].copy()


def resolve_px_per_mm(calibration_mode: str, dpi: float, img_shape: Tuple[int, int, int]) -> float:
    h, w = img_shape[:2]
    if calibration_mode == "按输入 DPI 计算":
        return dpi / 25.4
    width_mm, height_mm = A4_PORTRAIT_MM if h >= w else A4_LANDSCAPE_MM
    return min(w / width_mm, h / height_mm)


def lab_distance_to_color(img_bgr: np.ndarray, ref_lab: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = lab - ref_lab.reshape(1, 1, 3)
    return np.sqrt(np.sum(delta * delta, axis=2))


def otsu_threshold_from_float(dist: np.ndarray) -> float:
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    threshold_val, _ = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    d_min, d_max = float(dist.min()), float(dist.max())
    if d_max - d_min < 1e-6:
        return d_min
    return d_min + (float(threshold_val) / 255.0) * (d_max - d_min)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def contour_to_box(contour: np.ndarray) -> Tuple[np.ndarray, float, float]:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    (w, h) = rect[1]
    return box, float(w), float(h)


def detect_colored_board(img_bgr: np.ndarray, hue_tol: int = 14, sat_min: int = 35) -> Tuple[np.ndarray, Tuple[int, int, int, int], int]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    sat_pixels = H[(S > sat_min) & (V > 80)]
    if sat_pixels.size == 0:
        return img_bgr.copy(), (0, 0, img_bgr.shape[1], img_bgr.shape[0]), 100

    hist = np.bincount(sat_pixels, minlength=180)
    hue_peak = int(np.argmax(hist))

    dh = np.minimum((H.astype(np.int16) - hue_peak) % 180, (hue_peak - H.astype(np.int16)) % 180)
    board_mask = ((dh <= hue_tol) & (S >= sat_min) & (V > 60)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr.copy(), (0, 0, img_bgr.shape[1], img_bgr.shape[0]), hue_peak

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    pad = 8
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img_bgr.shape[1] - x, w + 2 * pad)
    h = min(img_bgr.shape[0] - y, h + 2 * pad)
    return img_bgr[y:y + h, x:x + w].copy(), (x, y, w, h), hue_peak


def build_board_masks_from_hue(board_bgr: np.ndarray, hue_peak: int, hue_tol: int, sat_min: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据背景纸主色相，在背景纸区域内重新生成：
    1) blue_mask：蓝色区域掩膜
    2) paper_mask：最大蓝色连通域，即背景纸掩膜
    """
    hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    dh = np.minimum(
        (H.astype(np.int16) - int(hue_peak)) % 180,
        (int(hue_peak) - H.astype(np.int16)) % 180
    )

    blue_mask = ((dh <= int(hue_tol)) & (S >= int(sat_min)) & (V > 60)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_mask = np.zeros_like(blue_mask)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(paper_mask, [cnt], -1, 255, -1)

    return blue_mask, paper_mask


def detect_stickers_binarized(
    board_bgr: np.ndarray,
    board_hue_peak: int,
    hue_tol: int,
    sat_min: int,
    value_threshold: int,
    morph_kernel: int,
    min_area_pct: float,
    max_area_pct: float,
    min_rectangularity: float,
    aspect_min: float,
    aspect_max: float,
) -> Tuple[List[ROIItem], np.ndarray, np.ndarray]:
    """
    改进版贴纸检测：
    1. 根据背景纸主色相提取蓝色背景纸
    2. 在背景纸内部寻找“非蓝色 + 亮色”区域作为贴纸候选
    3. 用面积、矩形度、长宽比过滤掉手写字、箭头、杂点
    """
    hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    blue_mask, paper_mask = build_board_masks_from_hue(
        board_bgr, hue_peak=board_hue_peak, hue_tol=hue_tol, sat_min=sat_min
    )

    non_blue_mask = cv2.bitwise_and(cv2.bitwise_not(blue_mask), paper_mask)

    bright_mask = (V >= int(value_threshold)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(non_blue_mask, bright_mask)

    kernel_size = max(1, int(morph_kernel))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size + 4, kernel_size + 4))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    paper_area = float(max(np.count_nonzero(paper_mask), 1))
    min_area = paper_area * min_area_pct / 100.0
    max_area = paper_area * max_area_pct / 100.0

    rois: List[ROIItem] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        box, bw, bh = contour_to_box(cnt)
        if bw < 10 or bh < 10:
            continue

        long_side = max(float(bw), float(bh))
        short_side = min(float(bw), float(bh))
        if short_side <= 1e-6:
            continue

        rect_area = max(float(bw) * float(bh), 1e-6)
        rectangularity = area / rect_area
        aspect = long_side / short_side

        x, y, rw, rh = cv2.boundingRect(cnt)
        bbox_area = max(float(rw * rh), 1e-6)
        extent = area / bbox_area

        if rectangularity < min_rectangularity:
            continue
        if not (aspect_min <= aspect <= aspect_max):
            continue
        if extent < 0.65:
            continue

        center = tuple(np.mean(box, axis=0).tolist())
        rois.append(
            ROIItem(
                box=order_quad_points(box),
                center=center,
                width_px=long_side,
                height_px=short_side,
                area_px=float(area),
                source="auto",
            )
        )

    return rois, mask, blue_mask


def cluster_rows(rois: List[ROIItem], row_tol_factor: float = 0.70) -> List[List[ROIItem]]:
    if not rois:
        return []
    median_h = float(np.median([r.height_px for r in rois]))
    tol = max(10.0, median_h * row_tol_factor)
    rois_sorted = sorted(rois, key=lambda r: r.center[1])
    rows: List[List[ROIItem]] = []
    for roi in rois_sorted:
        placed = False
        for row in rows:
            row_y = np.mean([r.center[1] for r in row])
            if abs(roi.center[1] - row_y) <= tol:
                row.append(roi)
                placed = True
                break
        if not placed:
            rows.append([roi])
    for row in rows:
        row.sort(key=lambda r: r.center[0])
    rows.sort(key=lambda rr: np.mean([r.center[1] for r in rr]))
    return rows


def assign_grid_labels(rois: List[ROIItem]) -> List[ROIItem]:
    auto_rois = [r for r in rois if r.source == "auto"]
    manual_rois = [r for r in rois if r.source == "manual"]
    rows = cluster_rows(auto_rois)
    ordered: List[ROIItem] = []
    for i, row in enumerate(rows, start=1):
        for j, roi in enumerate(row, start=1):
            roi.row = i
            roi.col = j
            roi.label = f"R{i}C{j}"
            ordered.append(roi)
    for m_i, roi in enumerate(sorted(manual_rois, key=lambda r: (r.center[1], r.center[0])), start=1):
        roi.label = f"M{m_i}"
        ordered.append(roi)
    return ordered


def warp_roi(img_bgr: np.ndarray, box: np.ndarray, pad_px: int = 2) -> np.ndarray:
    box = order_quad_points(box)
    width_a = np.linalg.norm(box[2] - box[3])
    width_b = np.linalg.norm(box[1] - box[0])
    height_a = np.linalg.norm(box[1] - box[2])
    height_b = np.linalg.norm(box[0] - box[3])
    max_width = max(10, int(max(width_a, width_b)))
    max_height = max(10, int(max(height_a, height_b)))
    dst = np.array(
        [
            [pad_px, pad_px],
            [max_width - 1 - pad_px, pad_px],
            [max_width - 1 - pad_px, max_height - 1 - pad_px],
            [pad_px, max_height - 1 - pad_px],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    return cv2.warpPerspective(img_bgr, matrix, (max_width, max_height))


def crop_inner_margin(img_bgr: np.ndarray, margin_pct: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mx = int(w * margin_pct / 100.0)
    my = int(h * margin_pct / 100.0)
    x0, x1 = max(0, mx), max(mx + 1, w - mx)
    y0, y1 = max(0, my), max(my + 1, h - my)
    return img_bgr[y0:y1, x0:x1].copy()


def estimate_paper_lab(roi_bgr: np.ndarray, k: int = 3) -> np.ndarray:
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if len(lab) < k:
        return np.median(lab, axis=0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _compactness, labels, centers = cv2.kmeans(lab, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k).astype(np.float32)
    brightness = centers[:, 0]
    scores = counts * (1.0 + 0.30 * (brightness / 255.0))
    idx = int(np.argmax(scores))
    return centers[idx]


def separate_touching_spots(binary_mask: np.ndarray) -> np.ndarray:
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        return mask_u8
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return mask_u8
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask_u8, kernel, iterations=1)
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    out = np.zeros_like(mask_u8)
    out[markers > 1] = 255
    return out


def segment_deposition(roi_bgr: np.ndarray, threshold_override: Optional[float], min_spot_area_px: int, use_watershed: bool) -> Tuple[np.ndarray, float]:
    paper_lab = estimate_paper_lab(roi_bgr)
    dist = lab_distance_to_color(roi_bgr, paper_lab)
    threshold = threshold_override if threshold_override is not None else otsu_threshold_from_float(dist)
    mask = (dist > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if use_watershed:
        mask = separate_touching_spots(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for lab_idx in range(1, num_labels):
        area = stats[lab_idx, cv2.CC_STAT_AREA]
        if area >= min_spot_area_px:
            clean[labels == lab_idx] = 255
    return clean, float(threshold)


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    if len(values) == 0:
        return float("nan")
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative = np.cumsum(weights)
    cutoff = percentile / 100.0 * cumulative[-1]
    idx = np.searchsorted(cumulative, cutoff, side="left")
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def correct_diameter(stain_diameter_um: np.ndarray, stain_area_um2: np.ndarray, model: str, power_a: float, power_b: float, area_a: float, area_b: float) -> np.ndarray:
    if model == "不校正（直接使用雾滴等效直径）":
        return stain_diameter_um.copy()
    if model == "幂函数：d = a * stain_d^b":
        return power_a * np.power(np.maximum(stain_diameter_um, 0.0), power_b)
    if model == "面积幂函数：d = a * area^b":
        return area_a * np.power(np.maximum(stain_area_um2, 0.0), area_b)
    return stain_diameter_um.copy()


def compute_roi_metrics(roi_bgr: np.ndarray, dep_mask: np.ndarray, params: AnalysisParams) -> dict:
    h, w = roi_bgr.shape[:2]
    roi_area_px = h * w
    roi_area_mm2 = roi_area_px / (params.px_per_mm ** 2)
    roi_area_cm2 = roi_area_mm2 / 100.0
    dep_px = int(np.count_nonzero(dep_mask))
    dep_area_mm2 = dep_px / (params.px_per_mm ** 2)
    coverage_pct = 100.0 * dep_px / max(roi_area_px, 1)

    result = {
        "ROI_height_px": h,
        "ROI_width_px": w,
        "ROI_area_mm2": roi_area_mm2,
        "Deposit_area_mm2": dep_area_mm2,
        "Coverage_pct": coverage_pct,
        "Coverage_flag": "高覆盖率，可能存在严重重叠" if coverage_pct >= params.high_coverage_warning_pct else "正常",
    }

    if params.droplet_mode == "仅覆盖率/染色面积":
        return result

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dep_mask, connectivity=8)
    spot_areas_px = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= params.min_spot_area_px:
            spot_areas_px.append(area)

    if len(spot_areas_px) == 0:
        result.update({
            "Spot_count": 0,
            "Spot_density_per_cm2": 0.0,
            "Mean_stain_area_mm2": 0.0,
            "Mean_stain_diameter_um": 0.0,
            "Mean_output_diameter_um": 0.0,
            "D10_um": np.nan,
            "D50_um": np.nan,
            "D90_um": np.nan,
            "DV0.1_um": np.nan,
            "DV0.5_um": np.nan,
            "DV0.9_um": np.nan,
        })
        return result

    spot_areas_px = np.asarray(spot_areas_px, dtype=np.float64)
    spot_areas_mm2 = spot_areas_px / (params.px_per_mm ** 2)
    stain_diam_mm = np.sqrt(4.0 * spot_areas_mm2 / math.pi)
    stain_diam_um = stain_diam_mm * 1000.0
    stain_area_um2 = spot_areas_mm2 * 1_000_000.0

    droplet_diam_um = correct_diameter(
        stain_diam_um,
        stain_area_um2,
        params.diameter_model,
        params.power_a,
        params.power_b,
        params.area_a,
        params.area_b,
    )

    count = len(spot_areas_px)
    density = count / max(roi_area_cm2, 1e-9)
    d10 = float(np.percentile(droplet_diam_um, 10))
    d50 = float(np.percentile(droplet_diam_um, 50))
    d90 = float(np.percentile(droplet_diam_um, 90))
    vol_weights = np.power(np.maximum(droplet_diam_um, 0.0), 3)
    dv01 = weighted_percentile(droplet_diam_um, vol_weights, 10)
    dv05 = weighted_percentile(droplet_diam_um, vol_weights, 50)
    dv09 = weighted_percentile(droplet_diam_um, vol_weights, 90)

    result.update({
        "Spot_count": count,
        "Spot_density_per_cm2": density,
        "Mean_stain_area_mm2": float(np.mean(spot_areas_mm2)),
        "Mean_stain_diameter_um": float(np.mean(stain_diam_um)),
        "Mean_output_diameter_um": float(np.mean(droplet_diam_um)),
        "D10_um": d10,
        "D50_um": d50,
        "D90_um": d90,
        "DV0.1_um": dv01,
        "DV0.5_um": dv05,
        "DV0.9_um": dv09,
    })
    return result


def create_detection_overlay(img_bgr: np.ndarray, rois: List[ROIItem], color=(0, 180, 0)) -> np.ndarray:
    overlay = img_bgr.copy()

    font_scale = 1.5          # 编号字体大小
    font_thickness = 4        # 编号线宽
    poly_thickness = 5        # 检测框线宽
    text_pad_x = 12
    text_pad_y = 10

    for idx, roi in enumerate(rois, start=1):
        pts = roi.box.astype(np.int32)
        cv2.polylines(overlay, [pts], True, color, poly_thickness)

        # 这里只显示“序号”，不再显示 R1C1 之类标签
        label_text = f"{idx}"

        (tw, th), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )

        anchor_x = int(np.min(pts[:, 0]))
        anchor_y = int(np.min(pts[:, 1]))

        # 默认把编号框放在目标左上角上方；若越界则放到框内上沿
        box_x0 = max(0, anchor_x)
        box_y1 = anchor_y - 6
        box_y0 = box_y1 - th - baseline - 2 * text_pad_y

        if box_y0 < 0:
            box_y0 = max(0, anchor_y + 4)
            box_y1 = box_y0 + th + baseline + 2 * text_pad_y

        box_x1 = min(overlay.shape[1] - 1, box_x0 + tw + 2 * text_pad_x)

        cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), (255, 255, 255), -1)
        cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), color, 3)

        text_x = box_x0 + text_pad_x
        text_y = box_y1 - baseline - text_pad_y

        cv2.putText(
            overlay,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA
        )

    return overlay


def create_result_overlay(img_bgr: np.ndarray, rois: List[ROIItem], df: pd.DataFrame, color=(0, 180, 0)) -> np.ndarray:
    overlay = img_bgr.copy()
    coverage_map = {row["Label"]: row["Coverage_pct"] for _, row in df.iterrows()}
    for roi in rois:
        pts = roi.box.astype(np.int32)
        cv2.polylines(overlay, [pts], True, color, 3)
        cov = coverage_map.get(roi.label, np.nan)
        text = f"{roi.label}: {cov:.2f}%"
        x = int(np.min(pts[:, 0]))
        y = max(18, int(np.min(pts[:, 1])) - 6)
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return overlay


def build_results_table(rois: List[ROIItem], analysis_rows: List[dict]) -> pd.DataFrame:
    rows = []
    for idx, (roi, metrics) in enumerate(zip(rois, analysis_rows), start=1):
        rows.append({
            "Index": idx,
            "Label": roi.label,
            "Source": roi.source,
            "Row": roi.row,
            "Col": roi.col,
            "Center_x_px": roi.center[0],
            "Center_y_px": roi.center[1],
            "Sticker_area_px": roi.area_px,
            **metrics,
        })
    return pd.DataFrame(rows)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    output.seek(0)
    return output.read()


def image_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return encoded.tobytes()


def format_metric(value, precision=2):
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return "-"
        return f"{float(value):.{precision}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def render_selected_panel(df: pd.DataFrame, preview_data: List[Tuple[str, np.ndarray, np.ndarray]]):
    if df.empty:
        st.info("暂无结果")
        return

    options = [f"#{int(row['Index'])} | {row['Label']} | {row['Source']}" for _, row in df.iterrows()]
    selected_option = st.selectbox("查看区域详情", options)
    selected_index = int(selected_option.split("|")[0].replace("#", "").strip())
    row = df.loc[df["Index"] == selected_index].iloc[0]

    c1, c2 = st.columns(2)
    c1.metric("标签", str(row["Label"]))
    c2.metric("来源", str(row["Source"]))

    c3, c4 = st.columns(2)
    c3.metric("覆盖率 %", format_metric(row.get("Coverage_pct"), 2))
    if "Spot_density_per_cm2" in df.columns:
        c4.metric("雾滴密度 /cm²", format_metric(row.get("Spot_density_per_cm2"), 2))
    else:
        c4.metric("沉积面积 mm²", format_metric(row.get("Deposit_area_mm2"), 2))

    c5, c6 = st.columns(2)
    c5.metric("雾滴数", format_metric(row.get("Spot_count"), 0))
    c6.metric("DV0.5 μm", format_metric(row.get("DV0.5_um"), 2))

    detail_cols = [
        "Deposit_area_mm2", "ROI_area_mm2", "Coverage_flag", "Mean_output_diameter_um",
        "D10_um", "D50_um", "D90_um", "DV0.1_um", "DV0.5_um", "DV0.9_um",
        "Used_board_hue_peak", "Used_sticker_value_threshold", "Used_deposition_threshold"
    ]
    detail = []
    for k in detail_cols:
        if k in row.index:
            detail.append({"指标": k, "值": format_metric(row[k], 3)})
    if detail:
        st.dataframe(pd.DataFrame(detail), height=min(360, 35 * (len(detail) + 1)))

    lookup = {label: (img, mask) for label, img, mask in preview_data}
    if row["Label"] in lookup:
        roi_img, roi_mask = lookup[row["Label"]]
        q1, q2 = st.columns(2)
        with q1:
            st.image(bgr_to_rgb(roi_img), caption=f"{row['Label']} - 区域图")
        with q2:
            st.image(roi_mask, caption=f"{row['Label']} - 沉积掩膜")

    summary_cols = [c for c in ["Index", "Label", "Source", "Coverage_pct", "Spot_density_per_cm2", "DV0.5_um", "Spot_count"] if c in df.columns]
    st.markdown("**全部区域摘要**")
    st.dataframe(df[summary_cols], height=min(360, 35 * (len(df) + 1)))


def rectangles_from_canvas(json_data, scale_x: float, scale_y: float) -> List[ROIItem]:
    rois = []
    if not json_data or "objects" not in json_data:
        return rois
    for obj in json_data["objects"]:
        if obj.get("type") != "rect":
            continue
        left = float(obj.get("left", 0.0)) * scale_x
        top = float(obj.get("top", 0.0)) * scale_y
        width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0)) * scale_x
        height = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0)) * scale_y
        angle = math.radians(float(obj.get("angle", 0.0)))
        cx = left + width / 2.0
        cy = top + height / 2.0
        pts = np.array(
            [
                [-width / 2.0, -height / 2.0],
                [width / 2.0, -height / 2.0],
                [width / 2.0, height / 2.0],
                [-width / 2.0, height / 2.0],
            ],
            dtype=np.float32,
        )
        if abs(angle) > 1e-6:
            rot = np.array(
                [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
                dtype=np.float32
            )
            pts = pts @ rot.T
        pts[:, 0] += cx
        pts[:, 1] += cy
        rois.append(
            ROIItem(
                box=order_quad_points(pts),
                center=(cx, cy),
                width_px=width,
                height_px=height,
                area_px=width * height,
                source="manual",
            )
        )
    return rois


def app():
    st.set_page_config(page_title="蓝底贴纸喷雾沉积分析", layout="wide")
    st.title("蓝色背景纸多贴纸喷雾沉积分析")
    st.caption("先识别蓝色背景纸，再在背景纸内部分割浅色长方形贴纸并自动编号；同时支持手动补框。")

    with st.sidebar:
        st.header("1. 图像与标定")
        uploaded = st.file_uploader("上传扫描图像", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        calibration_mode = st.radio("尺度标定方式", ["按输入 DPI 计算", "按当前裁剪区域视为完整 A4 计算"], index=0)
        dpi = st.number_input("扫描分辨率（dpi）", min_value=50.0, max_value=2400.0, value=600.0, step=50.0)
        crop_left = st.slider("左裁剪 %", 0.0, 30.0, 0.0, 0.5)
        crop_right = st.slider("右裁剪 %", 0.0, 30.0, 0.0, 0.5)
        crop_top = st.slider("上裁剪 %", 0.0, 30.0, 0.0, 0.5)
        crop_bottom = st.slider("下裁剪 %", 0.0, 30.0, 0.0, 0.5)

        st.header("2. 自动检测：基于蓝色背景分割贴纸")
        value_threshold = st.slider("贴纸最小亮度阈值 V", 0, 255, 180, 1)
        hue_tol = st.slider("背景纸主色相容差", 6, 30, 14, 1)
        sat_min = st.slider("背景纸最小饱和度", 10, 120, 35, 1)
        morph_kernel = st.slider("形态学核大小", 1, 21, 7, 2)
        min_area_pct = st.slider("最小贴纸面积（占背景纸 %）", 0.01, 10.0, 1.0, 0.01)
        max_area_pct = st.slider("最大贴纸面积（占背景纸 %）", 0.1, 40.0, 8.0, 0.1)
        min_rectangularity = st.slider("最小矩形度", 0.50, 1.00, 0.80, 0.01)
        aspect_min = st.slider("最小长宽比", 0.80, 1.80, 1.00, 0.01)
        aspect_max = st.slider("最大长宽比", 1.00, 3.00, 2.20, 0.01)

        st.header("3. 沉积分析")
        droplet_mode = st.radio("输出模式", ["离散雾滴参数", "仅覆盖率/染色面积"], index=0)
        inner_margin_pct = st.slider("自动检测区域内边距裁剪 %", 0.0, 15.0, 2.0, 0.5)
        auto_dep_threshold = st.checkbox("沉积阈值自动（推荐）", value=True)
        dep_threshold = st.slider("沉积阈值：与贴纸本底的颜色距离", 0.0, 120.0, 18.0, 1.0)
        min_spot_area_px = st.slider("最小雾滴面积（px）", 1, 100, 4, 1)
        use_watershed = st.checkbox("尝试分离轻微粘连雾滴", value=False)
        high_coverage_warning_pct = st.slider("高覆盖率警戒值 %", 5.0, 60.0, 20.0, 1.0)

        st.header("4. 直径换算（可选）")
        diameter_model = st.selectbox(
            "输出直径模型",
            ["不校正（直接使用雾滴等效直径）", "幂函数：d = a * stain_d^b", "面积幂函数：d = a * area^b"],
            index=0,
        )
        power_a = st.number_input("a（幂函数）", value=1.0, step=0.01, format="%.4f")
        power_b = st.number_input("b（幂函数）", value=1.0, step=0.01, format="%.4f")
        area_a = st.number_input("a（面积幂函数）", value=1.0, step=0.01, format="%.4f")
        area_b = st.number_input("b（面积幂函数）", value=0.5, step=0.01, format="%.4f")

    if uploaded is None:
        st.info("先上传一张扫描图像。")
        return

    img_bgr = read_image(uploaded)
    cropped = crop_by_percent(img_bgr, crop_left, crop_right, crop_top, crop_bottom)
    board_bgr, board_rect, hue_peak = detect_colored_board(cropped, hue_tol=hue_tol, sat_min=sat_min)
    px_per_mm = resolve_px_per_mm(calibration_mode, dpi, board_bgr.shape)

    rois_auto, sticker_mask, blue_mask = detect_stickers_binarized(
        board_bgr,
        hue_peak,
        hue_tol,
        sat_min,
        value_threshold,
        morph_kernel,
        min_area_pct,
        max_area_pct,
        min_rectangularity,
        aspect_min,
        aspect_max,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**调试信息**")
    st.sidebar.write(f"背景纸主色相峰值 H: {hue_peak}")
    st.sidebar.write(f"自动检测贴纸数: {len(rois_auto)}")
    st.sidebar.write(f"背景纸尺寸: {board_bgr.shape[:2]}")
    st.sidebar.write(f"像素密度 px/mm: {px_per_mm:.3f}")

    st.subheader("自动检测结果")
    c1, c2, c3 = st.columns([1.25, 1.0, 1.0])
    with c1:
        st.image(bgr_to_rgb(board_bgr), caption="背景纸区域（已自动裁掉外围白边）")
    with c2:
        st.image(blue_mask, caption="蓝色背景掩膜")
    with c3:
        st.image(sticker_mask, caption="贴纸候选掩膜")

    if len(rois_auto) == 0:
        st.warning("自动检测到 0 个贴纸。建议优先调整：最小贴纸面积、最小亮度阈值 V、最大长宽比、最小矩形度。")

    st.subheader("手动框选补充区域")
    manual_rois: List[ROIItem] = []
    if HAS_CANVAS:
        display_width = min(900, board_bgr.shape[1])
        scale = display_width / board_bgr.shape[1]
        display_height = int(board_bgr.shape[0] * scale)
        st.caption("可直接在下图拖动矩形，补充漏检贴纸，或只分析贴纸内部任意子区域。")
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.05)",
            stroke_width=2,
            stroke_color="#00AA00",
            background_image=Image.fromarray(bgr_to_rgb(board_bgr)),
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key="manual_canvas_rect",
        )
        manual_rois = rectangles_from_canvas(canvas_result.json_data, 1.0 / scale, 1.0 / scale)
    else:
        st.info("当前环境缺少 streamlit-drawable-canvas，暂时无法直接画框。可安装：pip install streamlit-drawable-canvas")

    rois_all = assign_grid_labels(rois_auto + manual_rois)
    detection_overlay = create_detection_overlay(board_bgr, rois_all)

    analysis_params = AnalysisParams(
        dpi=dpi,
        px_per_mm=px_per_mm,
        min_spot_area_px=min_spot_area_px,
        inner_margin_pct=inner_margin_pct,
        deposition_thresh=None if auto_dep_threshold else dep_threshold,
        use_watershed=use_watershed,
        droplet_mode=droplet_mode,
        diameter_model=diameter_model,
        power_a=power_a,
        power_b=power_b,
        area_a=area_a,
        area_b=area_b,
        high_coverage_warning_pct=high_coverage_warning_pct,
    )

    analysis_rows = []
    preview_data = []
    for roi in rois_all:
        warped = warp_roi(board_bgr, roi.box) if roi.source == "auto" else warp_roi(board_bgr, roi.box, pad_px=0)
        inner = crop_inner_margin(warped, inner_margin_pct if roi.source == "auto" else 0.0)
        roi.roi_image = inner

        dep_mask, dep_thr = segment_deposition(
            inner,
            analysis_params.deposition_thresh,
            min_spot_area_px,
            use_watershed
        )
        roi.roi_mask = dep_mask

        metrics = compute_roi_metrics(inner, dep_mask, analysis_params)
        metrics["Used_board_hue_peak"] = hue_peak
        metrics["Used_sticker_value_threshold"] = value_threshold
        metrics["Used_deposition_threshold"] = dep_thr
        analysis_rows.append(metrics)
        preview_data.append((roi.label, inner, dep_mask))

    df = build_results_table(rois_all, analysis_rows)
    result_overlay = create_result_overlay(board_bgr, rois_all, df)

    st.subheader("检测框与指标")
    left, right = st.columns([1.55, 1.0])
    with left:
        st.image(bgr_to_rgb(detection_overlay), caption="绿色框 + 序号 + 标签")
        st.image(bgr_to_rgb(result_overlay), caption="覆盖率结果标注")
    with right:
        st.metric("总区域数", len(rois_all))
        st.metric("自动检测区域", len(rois_auto))
        st.metric("手动框选区域", len(manual_rois))
        render_selected_panel(df, preview_data)

    st.subheader("结果表")
    st.dataframe(df)

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    excel_bytes = dataframe_to_excel_bytes(df)
    overlay_bytes = image_to_png_bytes(result_overlay)

    d1, d2, d3 = st.columns(3)
    d1.download_button("下载 CSV", csv_bytes, file_name="deposition_results.csv", mime="text/csv")
    d2.download_button(
        "下载 Excel",
        excel_bytes,
        file_name="deposition_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    d3.download_button("下载结果标注图", overlay_bytes, file_name="deposition_overlay.png", mime="image/png")

    with st.expander("方法说明与调参建议"):
        st.markdown(
            "1. 程序先识别蓝色背景纸，再在背景纸内部寻找非蓝色、且足够亮的浅色长方形贴纸。\n"
            "2. 与单纯用饱和度阈值找贴纸相比，这种方式对蓝底+浅色贴纸场景更稳定。\n"
            "3. 自动检测不到时，优先调整：最小贴纸面积、贴纸最小亮度阈值 V、最大长宽比、最小矩形度。\n"
            "4. 雾滴沉积分割是在每张贴纸矫正后的局部图上完成的，自动按贴纸本底颜色估计阈值。\n"
            "5. 手动画框适合补框，或只分析贴纸内部局部区域。\n"
            "6. 面积和直径可按输入 DPI 换算，也可按当前裁剪区域视为完整 A4 自动换算。"
        )


if __name__ == "__main__":
    app()