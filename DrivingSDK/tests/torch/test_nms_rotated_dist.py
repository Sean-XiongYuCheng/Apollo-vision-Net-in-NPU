import numpy as np
import torch
import torch_npu
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving
from mx_driving import NmsRotatedDistModel

torch.npu.config.allow_internal_format = False
torch_npu.npu.set_compile_mode(jit_compile=False)

MAX_PRINT_ITEMS = 5  # 打印项数上限


def get_nms_config():
    return {
        'do_ct_nms': True,
        'ct_dist_scale': 0.5,
        'class_ids': [0, 5],
        'rot_match_threshold': 0.5,
        'kernel_from_singlebox': False,
        'export_nms_rotated': True,
    }


def get_user_exact_data():
    scores_data = np.array(
        [
            [
                0.5281,
                0.2190,
                0.1011,
                0.0415,
                0.0309,
                0.0303,
                0.0295,
                0.0293,
                0.0213,
                0.0208,
                0.0197,
                0.0196,
                0.0194,
                0.0193,
                0.0174,
                0.0170,
                0.0168,
                0.0162,
                0.0162,
                0.0161,
            ],
            [
                0.4178,
                0.1769,
                0.0704,
                0.0418,
                0.0400,
                0.0379,
                0.0272,
                0.0224,
                0.0197,
                0.0192,
                0.0181,
                0.0178,
                0.0146,
                0.0142,
                0.0141,
                0.0140,
                0.0139,
                0.0139,
                0.0137,
                0.0137,
            ],
            [
                0.5018,
                0.1116,
                0.0612,
                0.0573,
                0.0525,
                0.0496,
                0.0426,
                0.0226,
                0.0211,
                0.0201,
                0.0198,
                0.0190,
                0.0187,
                0.0177,
                0.0174,
                0.0154,
                0.0153,
                0.0145,
                0.0143,
                0.0135,
            ],
            [
                0.4188,
                0.1464,
                0.1123,
                0.0878,
                0.0756,
                0.0662,
                0.0409,
                0.0240,
                0.0203,
                0.0195,
                0.0189,
                0.0184,
                0.0175,
                0.0175,
                0.0150,
                0.0146,
                0.0146,
                0.0134,
                0.0133,
                0.0131,
            ],
        ],
        dtype=np.float32,
    )

    labels_data = np.array(
        [
            [7, 7, 4, 7, 7, 7, 7, 2, 0, 0, 5, 7, 0, 7, 7, 7, 7, 7, 0, 0],
            [7, 7, 4, 7, 7, 0, 7, 2, 0, 0, 0, 0, 7, 7, 7, 7, 0, 7, 7, 0],
            [7, 7, 7, 7, 7, 4, 7, 0, 7, 7, 0, 7, 0, 0, 4, 7, 7, 0, 7, 0],
            [7, 7, 7, 7, 5, 4, 7, 0, 0, 7, 0, 0, 0, 7, 7, 7, 5, 7, 0, 7],
        ],
        dtype=np.int32,
    )

    scores = torch.from_numpy(scores_data)
    labels = torch.from_numpy(labels_data)

    torch.manual_seed(42)
    batch, num_boxes, box_size = 4, 20, 9
    bboxes = torch.zeros(batch, num_boxes, box_size)

    for b in range(batch):
        base_x, base_y = 10.0, 10.0
        idx_0 = [i for i, label in enumerate(labels_data[b]) if label == 0]
        idx_5 = [i for i, label in enumerate(labels_data[b]) if label == 5]

        for i in range(num_boxes):
            label = labels_data[b, i]

            bboxes[b, i, 2] = 2.0  # w_i = 2.0
            bboxes[b, i, 3] = 1.5  # h_i = 1.5
            bboxes[b, i, 4] = 1.0
            bboxes[b, i, 5] = 0.0
            bboxes[b, i, 6] = 0.0
            bboxes[b, i, 7:9] = 0.0

            # NMS抑制条件: dist_w <= w_i(2.0) AND dist_h <= h_i(1.5)
            # 为避免抑制: 同label框的x间隔>=3.0(>w_i), y间隔>=2.0(>h_i)
            # 不同label框自然分散在不同区域(label=0在左侧, label=5在右侧, 其他在中间)
            if label == 0 and len(idx_0) > 0:
                pos = idx_0.index(i)
                bboxes[b, i, 0] = base_x + pos * 3.0
                bboxes[b, i, 1] = base_y + pos * 2.0
            elif label == 5 and len(idx_5) > 0:
                pos = idx_5.index(i)
                bboxes[b, i, 0] = base_x + 30 + pos * 3.0
                bboxes[b, i, 1] = base_y + pos * 2.0
            else:
                bboxes[b, i, 0] = base_x + 10 + torch.rand(1).item() * 40
                bboxes[b, i, 1] = base_y + 10 + torch.rand(1).item() * 20

    return bboxes, scores, labels


def _truncate(data, max_items=MAX_PRINT_ITEMS):
    """截断数据，超过上限只打印前几个"""
    if hasattr(data, 'tolist'):
        data = data.tolist()
    if isinstance(data, (list, tuple)) and len(data) > max_items:
        return f"[{', '.join(str(x) for x in data[:max_items])}, ... ({len(data)} total)]"
    return data


def nms_rotated_dist_cpu(boxes, scores, threshold):
    scores = scores.reshape(-1)
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous().clone()

    keep = torch.ones(boxes.shape[0], dtype=torch.bool)

    for i in range(boxes.shape[0]):
        if not keep[i]:
            continue

        x_ctr_i = boxes[i, 0]
        y_ctr_i = boxes[i, 1]
        w_i = boxes[i, 2]
        h_i = boxes[i, 3]
        angle_i = boxes[i, 4]
        cos_i = torch.cos(angle_i)
        sin_i = torch.sin(angle_i)

        for j in range(i + 1, boxes.shape[0]):
            if not keep[j]:
                continue

            dist_x = x_ctr_i - boxes[j, 0]
            dist_y = y_ctr_i - boxes[j, 1]

            dist_w = torch.abs(dist_x * cos_i - dist_y * sin_i)
            dist_h = torch.abs(dist_x * sin_i + dist_y * cos_i)

            if dist_w <= w_i and dist_h <= h_i and threshold < 0.5:
                keep[j] = False

    keep_indices = order[keep].long()
    return keep_indices


def _build_default_nms_cfg(iou_threshold=0.2):
    return {
        'nms_type': 2,
        'iou_threshold': iou_threshold,
        'max_num_output': 200,
        'vcs_range': [-1000, -1000, 2000, 2000],
        'class_confidences': [0.0] * 20,
        'max_batch': 4,
        'category': 0,
        'task_type': 1,
        'delete_small': 0,
    }


class TestNmsRotatedDistFilter(TestCase):
    def driving_npu_to_exec(self, boxes, scores, threshold):
        keep = mx_driving.nms_rotated_dist(boxes, scores, threshold)
        return keep.cpu()

    def cpu_reference_exec(self, boxes, scores, threshold):
        boxes_cpu = boxes.cpu().to(torch.float32)
        scores_cpu = scores.cpu().to(torch.float32)
        return nms_rotated_dist_cpu(boxes_cpu, scores_cpu, threshold)

    def test_nms_rotated_dist_float32(self):
        shape_format = [
            [[np.float32, -1, [5, 5]], [np.float32, -1, [5]], 0.2],
            [[np.float32, -1, [100, 5]], [np.float32, -1, [100]], 0.2],
            [[np.float32, -1, [500, 5]], [np.float32, -1, [500]], 0.2],
            [[np.float32, -1, [800, 5]], [np.float32, -1, [800]], 0.2],
            [[np.float32, -1, [1000, 5]], [np.float32, -1, [1000]], 0.2],
        ]
        for item in shape_format:
            boxes_cpu, boxes_npu = create_common_tensor(item[0], 0, 3)
            scores_cpu, scores_npu = create_common_tensor(item[1], 0, 1)
            threshold = item[2]
            out_cpu = self.cpu_reference_exec(boxes_cpu, scores_cpu, threshold)
            out_npu = self.driving_npu_to_exec(boxes_npu, scores_npu, threshold)
            scores_cpu_flat = scores_cpu.reshape(-1)
            scores_npu_flat = scores_npu.cpu().reshape(-1)
            cpu_scores = scores_cpu_flat[out_cpu] if out_cpu.numel() > 0 else torch.tensor([])
            npu_scores = scores_npu_flat[out_npu] if out_npu.numel() > 0 else torch.tensor([])

            # 打印 CPU 和 NPU 结果对比
            cpu_out_list = _truncate(out_cpu.tolist() if out_cpu.numel() > 0 else [])
            cpu_scores_list = _truncate(cpu_scores.tolist() if cpu_scores.numel() > 0 else [])
            npu_out_list = _truncate(out_npu.tolist() if out_npu.numel() > 0 else [])
            npu_scores_list = _truncate(npu_scores.tolist() if npu_scores.numel() > 0 else [])
            print(f"[Test shape={item[0][2]}] CPU keep={cpu_out_list}, scores={cpu_scores_list}")
            print(f"[Test shape={item[0][2]}] NPU keep={npu_out_list}, scores={npu_scores_list}")

            self.assertEqual(out_cpu, out_npu)

    def test_nms_rotated_dist_model_nms_type_2(self):
        thresholds = [0.2, 0.5, 0.8]
        box_counts = [5, 100, 500]

        for threshold in thresholds:
            for num_boxes in box_counts:
                nms_cfg = _build_default_nms_cfg(iou_threshold=threshold)
                nms_cfg['max_num_output'] = num_boxes
                model = NmsRotatedDistModel(nms_cfg)

                boxes_9d = torch.rand(1, num_boxes, 9, dtype=torch.float32) * 100.0  # [1, num_boxes, 9] [0, 100)
                scores = torch.rand(1, num_boxes, dtype=torch.float32) * 0.99 + 0.01  # [1, num_boxes]    (0, 1.0]
                labels = torch.zeros(1, num_boxes, dtype=torch.int32)  # [1, num_boxes]    全0

                with torch.no_grad():
                    result = model(boxes_9d, scores, labels)

                nms_bboxes, nms_scores, _ = result
                npu_valid = nms_bboxes.shape[1]

                boxes_5d = boxes_9d[0, :, [0, 1, 3, 4, 6]]
                scores_1d = scores[0, :]
                cpu_keep = nms_rotated_dist_cpu(boxes_5d, scores_1d, threshold)

                npu_boxes_5d = nms_bboxes[0, :, [0, 1, 3, 4, 6]]
                cpu_boxes_5d = boxes_5d[cpu_keep]

                # 打印 CPU 和 NPU 结果对比
                cpu_scores = scores_1d[cpu_keep] if len(cpu_keep) > 0 else torch.tensor([])
                npu_scores = nms_scores[0, :npu_valid]
                cpu_scores_list = _truncate(cpu_scores.tolist() if cpu_scores.numel() > 0 else [])
                npu_scores_list = _truncate(npu_scores.tolist() if npu_valid > 0 else [])
                print(f"[Test Scenario] threshold={threshold}, num_boxes={num_boxes}")
                print(f"[CPU Result] keep_count={len(cpu_keep)}, scores={cpu_scores_list}")
                print(f"[NPU Result] keep count={npu_valid}, scores={npu_scores_list}")

                self.assertEqual(len(cpu_keep), npu_valid, f"Keep count mismatch: CPU={len(cpu_keep)}, NPU={npu_valid}")

                _, npu_sorted_idx = npu_boxes_5d[:, 0].sort()
                _, cpu_sorted_idx = cpu_boxes_5d[:, 0].sort()
                npu_sorted = npu_boxes_5d[npu_sorted_idx]
                cpu_sorted = cpu_boxes_5d[cpu_sorted_idx]

                self.assertRtolEqual(cpu_sorted, npu_sorted)

    def test_nms_rotated_dist_model_user_exact_data(self):
        nms_cfg = get_nms_config()
        bboxes, scores, labels = get_user_exact_data()

        model = NmsRotatedDistModel(nms_cfg)

        with torch.no_grad():
            result = model(bboxes, scores, labels)

        nms_bboxes = result[0]
        nms_scores = result[1]
        nms_labels = result[2]
        nms_valid = result[3] if len(result) > 3 else None

        for b in range(nms_bboxes.shape[0]):
            valid_num = int(nms_valid[b, 0].item()) if nms_valid is not None else nms_bboxes.shape[1]
            self.assertEqual(valid_num, 20)
            print(f"[Batch {b}] Check passed: output box count = {valid_num}")

            batch_labels = nms_labels[b, :valid_num].numpy()
            batch_scores = nms_scores[b, :valid_num].numpy()

            for label_id in np.unique(batch_labels):
                mask = batch_labels == label_id
                label_scores = batch_scores[mask]
                self.assertTrue(np.all(label_scores[:-1] >= label_scores[1:]))
                print(f"[Batch {b}] Check passed: label={label_id}, scores sorted descending")


if __name__ == '__main__':
    run_tests()
