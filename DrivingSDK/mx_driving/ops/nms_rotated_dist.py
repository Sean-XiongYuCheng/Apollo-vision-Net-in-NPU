"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

Integrated NMS Rotated module.

nms_type=0: 3D IoU (Python CPU)
nms_type=1: 2D IoU / IoA (Python CPU)
nms_type=2: Projection distance (NPU accelerated via nms_rotated_dist)
"""

import torch
from torch import nn
from torch.autograd import Function
import math
import mx_driving._C

VEHICLES = 4
MAX_LOG_AREA = 3.218876
ALPHA = 0.8


class NmsRotatedDistFunction(Function):
    @staticmethod
    def forward(ctx, boxes, scores, iou_threshold: float):
        if boxes.device.type != 'npu' or scores.device.type != 'npu':
            raise ValueError("Input boxes and scores must be on NPU")

        order = scores.sort(0, descending=True)[1]
        boxes = boxes[order].contiguous()
        keep, num_out = mx_driving._C.nms_rotated_dist(boxes, iou_threshold)  # pylint: disable=c-extension-no-member
        return order[keep[:num_out].long()].contiguous()


nms_rotated_dist = NmsRotatedDistFunction.apply


def _nms_rotated_dist_npu(boxes_5d, scores, threshold):
    boxes_npu = boxes_5d.to('npu')
    scores_npu = scores.to('npu')
    keep = nms_rotated_dist(boxes_npu, scores_npu, threshold)
    return keep.cpu().tolist()


def get_rotated_vertices_cpu(box):
    x_ctr = box[0]
    y_ctr = box[1]
    length = box[2]
    w = box[3]
    a = box[4]
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    cos_a_2 = cos_a * 0.5
    sin_a_2 = sin_a * 0.5

    pts = torch.zeros(4, 2, dtype=torch.float32, device=box.device)
    pts[0, 0] = x_ctr - sin_a_2 * w - cos_a_2 * length
    pts[0, 1] = y_ctr + cos_a_2 * w - sin_a_2 * length
    pts[1, 0] = x_ctr + sin_a_2 * w - cos_a_2 * length
    pts[1, 1] = y_ctr - cos_a_2 * w - sin_a_2 * length
    pts[2, 0] = x_ctr * 2.0 - pts[0, 0]
    pts[2, 1] = y_ctr * 2.0 - pts[0, 1]
    pts[3, 0] = x_ctr * 2.0 - pts[1, 0]
    pts[3, 1] = y_ctr * 2.0 - pts[1, 1]

    return pts


def dot_2d_cpu(A, B):
    return A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1]


def cross_2d_cpu(A, B):
    return A[:, 0] * B[:, 1] - B[:, 0] * A[:, 1]


def get_intersection_points_cpu(pts1, pts2):
    vec1 = torch.zeros(4, 2, dtype=torch.float32, device=pts1.device)
    vec2 = torch.zeros(4, 2, dtype=torch.float32, device=pts2.device)

    for i in range(4):
        vec1[i, 0] = pts1[(i + 1) % 4, 0] - pts1[i, 0]
        vec1[i, 1] = pts1[(i + 1) % 4, 1] - pts1[i, 1]
        vec2[i, 0] = pts2[(i + 1) % 4, 0] - pts2[i, 0]
        vec2[i, 1] = pts2[(i + 1) % 4, 1] - pts2[i, 1]

    intersections = []

    for i in range(4):
        for j in range(4):
            det = vec2[j, 0] * vec1[i, 1] - vec1[i, 0] * vec2[j, 1]

            if abs(float(det)) <= 1e-14:
                continue

            vec12_x = pts2[j, 0] - pts1[i, 0]
            vec12_y = pts2[j, 1] - pts1[i, 1]

            t1 = (vec2[j, 0] * vec12_y - vec12_x * vec2[j, 1]) / det
            t2 = (vec1[i, 0] * vec12_y - vec12_x * vec1[i, 1]) / det

            if 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:
                intersections.append(
                    torch.tensor(
                        [pts1[i, 0] + vec1[i, 0] * t1, pts1[i, 1] + vec1[i, 1] * t1],
                        dtype=torch.float32,
                        device=pts1.device,
                    )
                )

    for i in range(4):
        AB_x = vec2[0, 0]
        AB_y = vec2[0, 1]
        DA_x = vec2[3, 0]
        DA_y = vec2[3, 1]
        ABdotAB = AB_x * AB_x + AB_y * AB_y
        ADdotAD = DA_x * DA_x + DA_y * DA_y

        AP_x = pts1[i, 0] - pts2[0, 0]
        AP_y = pts1[i, 1] - pts2[0, 1]
        APdotAB = AP_x * AB_x + AP_y * AB_y
        APdotAD = -(AP_x * DA_x + AP_y * DA_y)

        if 0.0 <= APdotAB <= ABdotAB and 0.0 <= APdotAD <= ADdotAD:
            intersections.append(pts1[i].clone())

    for i in range(4):
        AB_x = vec1[0, 0]
        AB_y = vec1[0, 1]
        DA_x = vec1[3, 0]
        DA_y = vec1[3, 1]
        ABdotAB = AB_x * AB_x + AB_y * AB_y
        ADdotAD = DA_x * DA_x + DA_y * DA_y

        AP_x = pts2[i, 0] - pts1[0, 0]
        AP_y = pts2[i, 1] - pts1[0, 1]
        APdotAB = AP_x * AB_x + AP_y * AB_y
        APdotAD = -(AP_x * DA_x + AP_y * DA_y)

        if 0.0 <= APdotAB <= ABdotAB and 0.0 <= APdotAD <= ADdotAD:
            intersections.append(pts2[i].clone())

    if len(intersections) == 0:
        return torch.zeros(0, 2, dtype=torch.float32, device=pts1.device)

    return torch.stack(intersections)


def convex_sort_point_cpu(p):
    num_in = p.shape[0]
    if num_in <= 1:
        return p

    center_x = 0.0
    center_y = 0.0
    for i in range(num_in):
        center_x = center_x + p[i, 0]
        center_y = center_y + p[i, 1]
    center_x = center_x / float(num_in)
    center_y = center_y / float(num_in)

    angles = []
    for i in range(num_in):
        angles.append(math.atan2(float(p[i, 1] - center_y), float(p[i, 0] - center_x)))

    idxs = list(range(num_in))
    for i in range(1, num_in):
        angle = angles[i]
        idx = idxs[i]
        j = i - 1
        while j >= 0 and angles[j] > angle:
            angles[j + 1] = angles[j]
            idxs[j + 1] = idxs[j]
            j = j - 1
        angles[j + 1] = angle
        idxs[j + 1] = idx

    q = torch.zeros_like(p)
    for i in range(num_in):
        q[i, 0] = p[idxs[i], 0]
        q[i, 1] = p[idxs[i], 1]

    return q


def polygon_area_cpu(q):
    m = q.shape[0]
    if m <= 2:
        return 0.0

    area = 0.0
    q0x = float(q[0, 0])
    q0y = float(q[0, 1])
    for i in range(1, m - 1):
        v1x = float(q[i, 0]) - q0x
        v1y = float(q[i, 1]) - q0y
        v2x = float(q[i + 1, 0]) - q0x
        v2y = float(q[i + 1, 1]) - q0y
        cross = abs(v1x * v2y - v2x * v1y)
        area = area + cross

    return area / 2.0


def rotated_boxes_intersection_cpu(box1, box2):
    pts1 = get_rotated_vertices_cpu(box1)
    pts2 = get_rotated_vertices_cpu(box2)

    intersectPts = get_intersection_points_cpu(pts1, pts2)

    if intersectPts.shape[0] <= 2:
        return 0.0

    orderedPts = convex_sort_point_cpu(intersectPts)
    return polygon_area_cpu(orderedPts)


def rotated_boxes_intersection_with_dist_cpu(box1, box2):
    dist_x = box1[0] - box2[0]
    dist_y = box1[1] - box2[1]
    cos_a = math.cos(box1[4])
    sin_a = math.sin(box1[4])
    dist_l = abs(dist_x * cos_a - dist_y * sin_a)
    dist_w = abs(dist_x * sin_a + dist_y * cos_a)

    return (dist_l > box1[2]) or (dist_w > box1[3])


def boxes_is_overlap_with_coarse_cpu(box1, box2):
    dist_x = abs(box1[0] - box2[0])
    dist_y = abs(box1[1] - box2[1])

    if dist_x > (box1[2] + box2[2]) * 0.5:
        return False
    if dist_y > (box1[3] + box2[3]) * 0.5:
        return False
    return True


def single_box_iou_rotated_cpu(box1_raw, box2_raw, nms_type, delete_small):
    center_shift_x = (box1_raw[0] + box2_raw[0]) / 2.0
    center_shift_y = (box1_raw[1] + box2_raw[1]) / 2.0

    box1 = box1_raw.clone()
    box2 = box2_raw.clone()

    box1[0] -= center_shift_x
    box1[1] -= center_shift_y
    box2[0] -= center_shift_x
    box2[1] -= center_shift_y

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    if area1 < 1e-14 or area2 < 1e-14:
        return 0.0

    if nms_type == 2:
        large = rotated_boxes_intersection_with_dist_cpu(box1, box2)
        return 0.0 if large else 0.5
    else:
        if not boxes_is_overlap_with_coarse_cpu(box1, box2):
            return 0.0

        intersection = rotated_boxes_intersection_cpu(box1, box2)

        baseS = 1.0
        if nms_type == 0:
            baseS = area1 + area2 - intersection
        elif nms_type == 1:
            baseS = area1

        if delete_small == 1:
            iou1 = intersection / baseS if baseS > 1e-14 else 0.0
            iou2 = intersection / min(area1, area2) if min(area1, area2) > 1e-14 else 0.0
            iou = max(iou1, iou2)
            return iou
        else:
            return intersection / baseS if baseS > 1e-14 else 0.0


def filter_boxes_with_conf_vcs_range_cpu(
    scores,
    boxes,
    labels,
    modify_labels,
    filter_scores,
    num_valid,
    num_class_on_batch,
    vcs_range,
    confidences,
    category,
    num_class,
    box_size,
    num_boxes,
    task_type,
    delete_small,
):
    for b in range(scores.shape[0]):
        for idx in range(num_boxes):
            label = int(labels[b, idx])
            conf = scores[b, idx]
            x = boxes[b, idx, 0]
            y = boxes[b, idx, 1]
            length = boxes[b, idx, 3]
            w = boxes[b, idx, 4]

            modify_label = VEHICLES if (label <= VEHICLES and task_type == 0) else label
            modify_labels[b, idx] = modify_label

            valid_idx = b * num_class + modify_label if category == 1 else b

            x_valid = vcs_range[0] <= x <= vcs_range[2]
            y_valid = vcs_range[1] <= y <= vcs_range[3]
            label_valid = label < num_class
            conf_valid = conf >= confidences[label]

            if not x_valid or not y_valid or not label_valid or not conf_valid:
                filter_scores[b, idx] = 0.0
            else:
                if delete_small == 1:
                    score = conf
                    if modify_label == VEHICLES:
                        area = length * w
                        score = ALPHA * conf + (1 - ALPHA) * (math.log(area) / MAX_LOG_AREA)
                    filter_scores[b, idx] = score
                else:
                    filter_scores[b, idx] = conf
                num_valid.view(-1)[valid_idx] += 1

            num_class_on_batch.view(-1)[valid_idx] += 1


def nms_rotated_cpu(pred_bboxes, pred_scores, pred_labels, pred_queries, nms_cfg):
    orig_bboxes_ndim = pred_bboxes.ndim
    orig_scores_ndim = pred_scores.ndim

    if orig_bboxes_ndim == 2:
        pred_bboxes = pred_bboxes.unsqueeze(0)
        if pred_queries is not None:
            pred_queries = pred_queries.unsqueeze(0)
    if orig_scores_ndim == 2:
        pred_scores = pred_scores.unsqueeze(-1)
        pred_labels = pred_labels.unsqueeze(-1)
    elif orig_scores_ndim == 1:
        pred_scores = pred_scores.unsqueeze(0).unsqueeze(-1)
        pred_labels = pred_labels.unsqueeze(0).unsqueeze(-1)

    nms_type = nms_cfg.get('nms_type', 2)
    iou_threshold = nms_cfg.get('iou_threshold', 0.2)
    max_num_output = nms_cfg.get('max_num_output', 20)
    vcs_range_cfg = nms_cfg.get('vcs_range', [-50, -50, 150, 50])
    class_confidences = nms_cfg.get('class_confidences', [0] * 19)
    category = nms_cfg.get('category', 0)
    with_nms_labels = nms_cfg.get('with_nms_labels', None)
    task_type = nms_cfg.get('task_type', 1)
    delete_small = nms_cfg.get('delete_small', 0)

    if nms_cfg.get('class_ids') is not None:
        with_nms_labels = nms_cfg.get('class_ids')
        category = 1

    batch, num_boxes, box_size = pred_bboxes.shape
    vcs_range = vcs_range_cfg

    pred_labels = pred_labels.to(torch.int32)

    if with_nms_labels is not None:
        with_labels = torch.tensor(with_nms_labels, dtype=torch.int32, device=pred_bboxes.device)
    else:
        with_labels = None

    num_class = len(class_confidences)

    nms_bboxes = torch.zeros(
        [batch, max_num_output, box_size],
        dtype=pred_bboxes.dtype,
        device=pred_bboxes.device,
    )
    nms_scores = torch.zeros(
        [batch, max_num_output, 1],
        dtype=pred_scores.dtype,
        device=pred_bboxes.device,
    )
    nms_labels = torch.zeros([batch, max_num_output, 1], dtype=torch.int32, device=pred_bboxes.device)

    if pred_queries is not None:
        _, _, queries_size = pred_queries.shape
        nms_queries = torch.zeros(
            [batch, max_num_output, queries_size],
            dtype=pred_queries.dtype,
            device=pred_bboxes.device,
        )
    else:
        nms_queries = None

    nms_valid_num = torch.zeros([batch, 1], dtype=torch.int32, device=pred_bboxes.device)

    confidences = torch.tensor(class_confidences, dtype=torch.float32, device=pred_bboxes.device)

    filter_scores = torch.zeros([batch, num_boxes], dtype=torch.float32, device=pred_bboxes.device)
    num_valid = torch.zeros(
        [batch, num_class] if category == 1 else [batch], dtype=torch.int32, device=pred_bboxes.device
    )
    num_class_on_batch = torch.zeros(
        [batch, num_class] if category == 1 else [batch], dtype=torch.int32, device=pred_bboxes.device
    )
    modify_labels = torch.zeros([batch, num_boxes], dtype=torch.int32, device=pred_bboxes.device)

    queries_size = pred_queries.shape[2] if pred_queries is not None else 0

    filter_scores_input = pred_scores.squeeze(-1) if pred_scores.shape[-1] == 1 else pred_scores
    filter_labels_input = pred_labels.squeeze(-1) if pred_labels.shape[-1] == 1 else pred_labels

    filter_boxes_with_conf_vcs_range_cpu(
        filter_scores_input,
        pred_bboxes,
        filter_labels_input,
        modify_labels,
        filter_scores,
        num_valid,
        num_class_on_batch,
        vcs_range,
        confidences,
        category,
        num_class,
        box_size,
        num_boxes,
        task_type,
        delete_small,
    )

    if category == 1:
        nms_rotated_cpu_with_category(
            pred_bboxes,
            pred_scores,
            pred_labels,
            pred_queries,
            nms_bboxes,
            nms_scores,
            nms_labels,
            nms_queries,
            nms_valid_num,
            filter_scores,
            num_valid,
            num_class_on_batch,
            modify_labels,
            with_labels,
            nms_type,
            iou_threshold,
            max_num_output,
            box_size,
            queries_size,
            num_class,
            num_boxes,
            batch,
            delete_small,
        )
    else:
        nms_rotated_cpu_wo_category(
            pred_bboxes,
            pred_scores,
            pred_labels,
            pred_queries,
            nms_bboxes,
            nms_scores,
            nms_labels,
            nms_queries,
            nms_valid_num,
            filter_scores,
            num_valid,
            num_class_on_batch,
            modify_labels,
            nms_type,
            iou_threshold,
            max_num_output,
            box_size,
            queries_size,
            num_class,
            num_boxes,
            batch,
            delete_small,
        )

    nms_result = []
    add_valid_num = True

    if not torch.onnx.is_in_onnx_export():
        if batch == 1:
            add_valid_num = False
            valid_num = nms_valid_num.item()
            nms_bboxes = nms_bboxes[:, :valid_num, :]
            nms_scores = nms_scores[:, :valid_num, :]
            nms_labels = nms_labels[:, :valid_num, :]
            nms_queries = nms_queries[:, :valid_num, :] if nms_queries is not None else None

        if orig_bboxes_ndim == 2:
            nms_bboxes = nms_bboxes.squeeze(0)
            nms_queries = nms_queries.squeeze(0) if nms_queries is not None else None

        if orig_scores_ndim == 2:
            nms_scores = nms_scores.squeeze(-1)
            nms_labels = nms_labels.squeeze(-1)
        elif orig_scores_ndim == 1:
            nms_scores = nms_scores.squeeze(-1).squeeze(0)
            nms_labels = nms_labels.squeeze(-1).squeeze(0)

    nms_result = [nms_bboxes, nms_scores, nms_labels]

    if nms_queries is not None:
        nms_result.append(nms_queries)

    if add_valid_num:
        nms_result.append(nms_valid_num)

    return tuple(nms_result)


def _suppress_rotated_by_iou(
    cls_sorted_indices,
    pred_bboxes,
    batch_idx,
    nms_type,
    delete_small,
    iou_threshold,
):
    """Greedy NMS suppression for the non-NPU (else) branch."""
    device = pred_bboxes.device
    keep = [True] * len(cls_sorted_indices)

    for i in range(len(cls_sorted_indices)):
        if not keep[i]:
            continue

        box1_idx = cls_sorted_indices[i]
        box1_raw = pred_bboxes[batch_idx, box1_idx, :]
        box1 = torch.tensor(
            [box1_raw[0], box1_raw[1], box1_raw[3], box1_raw[4], box1_raw[6]],
            dtype=box1_raw.dtype,
            device=device,
        )

        for j in range(i + 1, len(cls_sorted_indices)):
            if not keep[j]:
                continue

            box2_idx = cls_sorted_indices[j]
            box2_raw = pred_bboxes[batch_idx, box2_idx, :]
            box2 = torch.tensor(
                [box2_raw[0], box2_raw[1], box2_raw[3], box2_raw[4], box2_raw[6]],
                dtype=box2_raw.dtype,
                device=device,
            )

            iou = single_box_iou_rotated_cpu(box1, box2, nms_type, delete_small)

            if iou > iou_threshold:
                keep[j] = False

    return keep


def _write_kept_box_outputs(
    batch_idx,
    current_valid,
    orig_idx,
    box_size,
    queries_size,
    pred_bboxes,
    pred_scores,
    pred_labels,
    pred_queries,
    nms_bboxes,
    nms_scores,
    nms_labels,
    nms_queries,
):
    """Copy a single kept box into the NMS output buffers, return next valid index."""
    for k in range(box_size):
        nms_bboxes[batch_idx, current_valid, k] = pred_bboxes[batch_idx, orig_idx, k]

    nms_scores[batch_idx, current_valid, 0] = pred_scores[batch_idx, orig_idx, 0]
    nms_labels[batch_idx, current_valid, 0] = pred_labels[batch_idx, orig_idx, 0]

    if pred_queries is not None:
        for k in range(queries_size):
            nms_queries[batch_idx, current_valid, k] = pred_queries[batch_idx, orig_idx, k]

    return current_valid + 1


def nms_rotated_cpu_with_category(
    pred_bboxes,
    pred_scores,
    pred_labels,
    pred_queries,
    nms_bboxes,
    nms_scores,
    nms_labels,
    nms_queries,
    nms_valid_num,
    filter_scores,
    num_valid,
    num_class_on_batch,
    modify_labels,
    with_labels,
    nms_type,
    iou_threshold,
    max_num_output,
    box_size,
    queries_size,
    num_class,
    num_boxes,
    batch,
    delete_small,
):
    device = pred_bboxes.device

    for b in range(batch):
        batch_scores = []
        batch_indices = []
        for i in range(num_boxes):
            batch_scores.append(float(filter_scores[b, i]))
            batch_indices.append(i)

        sorted_pairs = sorted(zip(batch_scores, batch_indices, range(len(batch_indices))), key=lambda x: (-x[0], x[2]))
        all_sorted_indices = [idx for _, idx, _ in sorted_pairs]

        sorted_labels = []
        for idx in all_sorted_indices:
            sorted_labels.append(int(modify_labels[b, idx]))

        # 使用 Python 稳定排序，确保相同 label 时按原始顺序排列
        pairs = list(enumerate(sorted_labels))
        sorted_pairs = sorted(pairs, key=lambda x: x[1])  # 按 label 升序，稳定排序
        labels_order = torch.tensor([idx for idx, _ in sorted_pairs], dtype=torch.int32, device=device)

        sorted_by_label = [all_sorted_indices[i] for i in labels_order.tolist()]

        class_count = torch.zeros(num_class, dtype=torch.int32, device=device)

        for idx in range(num_boxes):
            cls = sorted_labels[idx]
            if cls < num_class:
                class_count[cls] += 1

        num_class_prefix_sum = torch.zeros(num_class, dtype=torch.int32, device=device)
        for cls in range(1, num_class):
            num_class_prefix_sum[cls] = num_class_prefix_sum[cls - 1] + class_count[cls - 1]

        for cls in range(num_class):
            cls_valid_num = int(num_valid[b, cls])
            if cls_valid_num < 1:
                continue

            if num_class_prefix_sum[cls] + cls_valid_num > num_boxes:
                continue

            cls_indices = []
            cls_scores = []
            for i in range(cls_valid_num):
                global_idx = sorted_by_label[num_class_prefix_sum[cls] + i]
                cls_indices.append(global_idx)
                cls_scores.append(float(filter_scores[b, global_idx]))

            cls_sorted_pairs = sorted(
                zip(cls_scores, cls_indices, range(len(cls_indices))), key=lambda x: (-x[0], x[2])
            )
            cls_sorted_indices = [idx for _, idx, _ in cls_sorted_pairs]
            cls_scores_sorted = [sc for sc, _, _ in cls_sorted_pairs]

            if nms_type == 2:
                cls_boxes_5d = torch.stack(
                    [
                        pred_bboxes[b, cls_sorted_indices, 0],
                        pred_bboxes[b, cls_sorted_indices, 1],
                        pred_bboxes[b, cls_sorted_indices, 3],
                        pred_bboxes[b, cls_sorted_indices, 4],
                        pred_bboxes[b, cls_sorted_indices, 6],
                    ],
                    dim=1,
                )

                cls_scores_t = torch.tensor(cls_scores_sorted, dtype=torch.float32, device=device)

                keep_indices = _nms_rotated_dist_npu(cls_boxes_5d, cls_scores_t, iou_threshold)
                keep_set = set(keep_indices)
            else:
                keep = _suppress_rotated_by_iou(
                    cls_sorted_indices, pred_bboxes, b, nms_type, delete_small, iou_threshold
                )

            current_valid = int(nms_valid_num[b, 0])
            for i in range(len(cls_sorted_indices)):
                in_with_labels = (with_labels is None) or (cls in with_labels.tolist())
                if in_with_labels:
                    suppressed = (i not in keep_set) if nms_type == 2 else (not keep[i])
                    if suppressed:
                        continue

                if current_valid >= max_num_output:
                    continue

                orig_idx = cls_sorted_indices[i]
                current_valid = _write_kept_box_outputs(
                    b,
                    current_valid,
                    orig_idx,
                    box_size,
                    queries_size,
                    pred_bboxes,
                    pred_scores,
                    pred_labels,
                    pred_queries,
                    nms_bboxes,
                    nms_scores,
                    nms_labels,
                    nms_queries,
                )

            nms_valid_num[b, 0] = current_valid


def nms_rotated_cpu_wo_category(
    pred_bboxes,
    pred_scores,
    pred_labels,
    pred_queries,
    nms_bboxes,
    nms_scores,
    nms_labels,
    nms_queries,
    nms_valid_num,
    filter_scores,
    num_valid,
    num_class_on_batch,
    modify_labels,
    nms_type,
    iou_threshold,
    max_num_output,
    box_size,
    queries_size,
    num_class,
    num_boxes,
    batch,
    delete_small,
):
    device = pred_bboxes.device

    for b in range(batch):
        b_valid_num = int(num_valid[b])

        if b_valid_num < 1:
            continue

        indices = []
        scores_list = []

        for i in range(num_boxes):
            if filter_scores[b, i] > 0:
                indices.append(i)
                scores_list.append(float(filter_scores[b, i]))

        sorted_pairs = sorted(zip(scores_list, indices, range(len(indices))), key=lambda x: (-x[0], x[2]))
        sorted_indices = [idx for _, idx, _ in sorted_pairs]
        sorted_scores = [sc for sc, _, _ in sorted_pairs]

        if nms_type == 2:
            sorted_boxes_5d = torch.stack(
                [
                    pred_bboxes[b, sorted_indices, 0],
                    pred_bboxes[b, sorted_indices, 1],
                    pred_bboxes[b, sorted_indices, 3],
                    pred_bboxes[b, sorted_indices, 4],
                    pred_bboxes[b, sorted_indices, 6],
                ],
                dim=1,
            )

            sorted_scores_t = torch.tensor(sorted_scores, dtype=torch.float32, device=device)

            keep_indices = _nms_rotated_dist_npu(sorted_boxes_5d, sorted_scores_t, iou_threshold)
            keep_set = set(keep_indices)
        else:
            keep = [True] * len(sorted_indices)

            for i in range(len(sorted_indices)):
                if not keep[i]:
                    continue

                box1_idx = sorted_indices[i]
                box1_raw = pred_bboxes[b, box1_idx, :]
                box1 = torch.tensor(
                    [box1_raw[0], box1_raw[1], box1_raw[3], box1_raw[4], box1_raw[6]],
                    dtype=box1_raw.dtype,
                    device=device,
                )

                for j in range(i + 1, len(sorted_indices)):
                    if not keep[j]:
                        continue

                    box2_idx = sorted_indices[j]
                    box2_raw = pred_bboxes[b, box2_idx, :]
                    box2 = torch.tensor(
                        [box2_raw[0], box2_raw[1], box2_raw[3], box2_raw[4], box2_raw[6]],
                        dtype=box2_raw.dtype,
                        device=device,
                    )

                    iou = single_box_iou_rotated_cpu(box1, box2, nms_type, delete_small)

                    if iou > iou_threshold:
                        keep[j] = False

        num_to_keep = 0

        for i in range(len(sorted_indices)):
            if nms_type == 2:
                if i not in keep_set:
                    continue
            else:
                if not keep[i]:
                    continue

            if num_to_keep >= max_num_output:
                continue

            orig_idx = sorted_indices[i]

            for k in range(box_size):
                nms_bboxes[b, num_to_keep, k] = pred_bboxes[b, orig_idx, k]

            nms_scores[b, num_to_keep, 0] = pred_scores[b, orig_idx, 0]
            nms_labels[b, num_to_keep, 0] = pred_labels[b, orig_idx, 0]

            if pred_queries is not None:
                for k in range(queries_size):
                    nms_queries[b, num_to_keep, k] = pred_queries[b, orig_idx, k]

            num_to_keep += 1

        nms_valid_num[b, 0] = num_to_keep


class NmsRotatedDistModel(nn.Module):
    def __init__(self, nms_cfg):
        super().__init__()
        self.nms_cfg = nms_cfg

    def forward(self, bboxes, scores, labels, pred_queries=None):
        return nms_rotated_cpu(bboxes, scores, labels, pred_queries, self.nms_cfg)
