import cv2
import numpy as np
import time

from vlm.detector.yoloe import YOLOEClient
from vlm.utils.get_itm_message import get_itm_message

yoloe_detector = YOLOEClient(port=12184)


def get_segmentation(segmented_img, idx, detections, img, label, score, color):
    object_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bbox_denorm = detections.boxes[idx] * np.array(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    )
    x1, y1, x2, y2 = [int(v) for v in bbox_denorm]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1] - 1)
    y2 = min(y2, img.shape[0] - 1)

    if idx < len(detections.masks) and detections.masks[idx] is not None:
        object_mask = detections.masks[idx].astype(np.uint8)
    else:
        object_mask[y1 : y2 + 1, x1 : x2 + 1] = 1

    contours, _ = cv2.findContours(
        object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        cv2.drawContours(segmented_img, [contour], 0, color, 4)

    cv2.rectangle(
        segmented_img,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    label_text = f"{label} ({score:.2f})"
    (text_width, text_height), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
    )
    label_x = x1
    label_y = y1 - text_height
    cv2.rectangle(
        segmented_img,
        (label_x, label_y - 30),
        (label_x + text_width, label_y + text_height),
        color,
        2,
    )
    cv2.putText(
        segmented_img,
        label_text,
        (label_x, label_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (255, 255, 255),
        1,
    )

    return segmented_img, object_mask


def _get_yoloe_params(cfg):
    yoloe_cfg = getattr(cfg, "yoloe", None)
    if yoloe_cfg is not None:
        return (
            getattr(yoloe_cfg, "confidence_threshold", 0.3),
            getattr(yoloe_cfg, "iou_threshold", 0.5),
            getattr(yoloe_cfg, "agnostic_nms", True),
        )

    legacy_yolo_cfg = getattr(cfg, "yolo", None)
    legacy_dino_cfg = getattr(cfg, "groundingDINO", None)
    confidence_threshold = getattr(
        legacy_yolo_cfg,
        "confidence_threshold_yolo",
        getattr(legacy_dino_cfg, "confidence_threshold_dino", 0.3),
    )
    iou_threshold = getattr(legacy_yolo_cfg, "iou_threshold_yolo", 0.5)
    agnostic_nms = getattr(legacy_yolo_cfg, "agnostic_nms", True)
    return confidence_threshold, iou_threshold, agnostic_nms


def _merge_labels(right_label, similar_answer):
    right_label_list = [label.strip() for label in right_label.split("|") if label.strip()]
    all_answer = right_label_list + [label.strip() for label in similar_answer if label.strip()]
    all_answer = list(dict.fromkeys(all_answer))
    return right_label_list, all_answer


def get_object(right_label, img, cfg, similar_answer, return_stats=False):
    score_list = []
    object_masks_list = []
    segmented_img = img.copy()
    label_list = []

    right_label_list, all_answer = _merge_labels(right_label, similar_answer)
    if not all_answer:
        if return_stats:
            return segmented_img, score_list, object_masks_list, label_list, {
                "yoloe_latency_ms": 0.0
            }
        return segmented_img, score_list, object_masks_list, label_list

    conf_thres, iou_thres, agnostic_nms = _get_yoloe_params(cfg)
    yoloe_start = time.perf_counter()
    detections = yoloe_detector.predict(
        img,
        classes=all_answer,
        agnostic_nms=agnostic_nms,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )
    yoloe_latency_ms = (time.perf_counter() - yoloe_start) * 1000.0

    for idx in range(len(detections.logits)):
        label_detected = detections.phrases[idx]
        score = detections.logits[idx].item()
        if label_detected in right_label_list:
            segmented_img, object_mask = get_segmentation(
                segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
            )
            score_list.append(score)
            object_masks_list.append(object_mask)
            label_list.append(0)
        elif label_detected in all_answer:
            segmented_img, object_mask = get_segmentation(
                segmented_img, idx, detections, img, label_detected, score, color=(0, 255, 0)
            )
            score_list.append(score)
            object_masks_list.append(object_mask)
            label_list.append(all_answer.index(label_detected) - len(right_label_list) + 1)

    if return_stats:
        return segmented_img, score_list, object_masks_list, label_list, {
            "yoloe_latency_ms": yoloe_latency_ms
        }
    return segmented_img, score_list, object_masks_list, label_list


def get_object_with_itm(label, img, cfg):
    score_list = []
    object_masks_list = []
    cosine_list = []
    itm_score_list = []
    segmented_img = img.copy()

    conf_thres, iou_thres, agnostic_nms = _get_yoloe_params(cfg)
    detections = yoloe_detector.predict(
        img,
        classes=[label],
        agnostic_nms=agnostic_nms,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )

    for idx in range(len(detections.logits)):
        label_detected = detections.phrases[idx]
        score = detections.logits[idx].item()
        if label_detected != label:
            continue

        segmented_img, object_mask = get_segmentation(
            segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
        )
        img_detected = crop_and_expand_box(img, detections, idx)
        cosine, itm_score = get_itm_message(img_detected, label)
        print(f"cosine: {cosine:.3f}, itm_score: {itm_score:.3f}")
        score_list.append(score)
        object_masks_list.append(object_mask)
        cosine_list.append(cosine)
        itm_score_list.append(itm_score)

    return segmented_img, score_list, object_masks_list, cosine_list, itm_score_list


def crop_and_expand_box(img, detections, idx, expand_pixels=0.4):
    x_min, y_min, x_max, y_max = detections.boxes[idx]
    x_min = int(x_min * img.shape[1])
    y_min = int(y_min * img.shape[0])
    x_max = int(x_max * img.shape[1])
    y_max = int(y_max * img.shape[0])

    x_min = max(int(x_min * (1 - expand_pixels)), 0)
    y_min = max(int(y_min * (1 - expand_pixels)), 0)
    x_max = min(int(x_max * (1 + expand_pixels)), img.shape[1] - 1)
    y_max = min(int(y_max * (1 + expand_pixels)), img.shape[0] - 1)

    img_detected = img[y_min : y_max + 1, x_min : x_max + 1]
    return img_detected
