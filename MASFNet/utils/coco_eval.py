import math
import re
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _pycocotools_version():
    try:
        try:
            from importlib.metadata import version
        except ImportError:
            import pkg_resources
            return pkg_resources.get_distribution('pycocotools').version
        return version('pycocotools')
    except Exception:
        return 'unknown'


def condition_from_path(path):
    stem = path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    match = re.match(r'(.+?)_(?:train|valid|test)_\d+$', stem, re.IGNORECASE)
    return match.group(1) if match else 'unknown'


def build_coco_ground_truth(annotation_lines, class_names):
    images = []
    annotations = []
    records = {}
    annotation_id = 1

    for image_id, line in enumerate(annotation_lines, 1):
        fields = line.split()
        if not fields:
            continue
        image_path = fields[0]
        with Image.open(image_path) as image:
            width, height = image.size
        condition = condition_from_path(image_path)
        images.append({
            'id': image_id,
            'file_name': image_path.rsplit('/', 1)[-1],
            'width': width,
            'height': height,
        })
        records[image_id] = {
            'path': image_path,
            'condition': condition,
            'width': width,
            'height': height,
        }

        for box_text in fields[1:]:
            x1, y1, x2, y2, class_id = map(float, box_text.split(','))
            box_width, box_height = x2 - x1, y2 - y1
            if not 0 <= class_id < len(class_names) or box_width <= 0 or box_height <= 0:
                raise ValueError('Invalid ground-truth box in {}'.format(image_path))
            annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': int(class_id) + 1,
                'bbox': [x1, y1, box_width, box_height],
                'area': box_width * box_height,
                'iscrowd': 0,
            })
            annotation_id += 1

    coco = COCO()
    coco.dataset = {
        'info': {'description': 'XWOD bounding-box evaluation'},
        'images': images,
        'annotations': annotations,
        'categories': [
            {'id': index + 1, 'name': name, 'supercategory': 'object'}
            for index, name in enumerate(class_names)
        ],
    }
    with redirect_stdout(StringIO()):
        coco.createIndex()
    return coco, records


def _safe_metric(value):
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _empty_detections(coco_gt):
    coco_dt = COCO()
    coco_dt.dataset = {
        'info': coco_gt.dataset['info'],
        'images': coco_gt.dataset['images'],
        'categories': coco_gt.dataset['categories'],
        'annotations': [],
    }
    with redirect_stdout(StringIO()):
        coco_dt.createIndex()
    return coco_dt


def _coco_metrics(coco_gt, detections, image_ids, max_detections):
    with redirect_stdout(StringIO()):
        if detections:
            coco_dt = coco_gt.loadRes(detections)
        else:
            coco_dt = _empty_detections(coco_gt)
        evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = sorted(coco_gt.getCatIds())
        evaluator.params.iouThrs = IOU_THRESHOLDS
        evaluator.params.maxDets = [1, min(10, max_detections), max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    annotations = coco_gt.dataset['annotations']
    image_id_set = set(image_ids)
    instance_count = sum(annotation['image_id'] in image_id_set
                         for annotation in annotations)
    result = {
        'map_50_95': _safe_metric(evaluator.stats[0]),
        'map_50': _safe_metric(evaluator.stats[1]),
        'map_75': _safe_metric(evaluator.stats[2]),
        'ar_1': _safe_metric(evaluator.stats[6]),
        'ar_10': _safe_metric(evaluator.stats[7]),
        'ar_100': _safe_metric(evaluator.stats[8]),
        'images': len(image_ids),
        'instances': instance_count,
    }

    precision = evaluator.eval['precision']
    per_class = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        category = coco_gt.cats[category_id]
        values = precision[:, :, category_index, 0, -1]
        ap_50_95 = _safe_metric(values[values >= 0].mean()) if (values >= 0).any() else None
        values_50 = precision[0, :, category_index, 0, -1]
        ap_50 = _safe_metric(values_50[values_50 >= 0].mean()) if (values_50 >= 0).any() else None
        per_class[category['name']] = {
            'ap_50_95': ap_50_95,
            'ap_50': ap_50,
            'instances': sum(
                annotation['category_id'] == category_id and annotation['image_id'] in image_id_set
                for annotation in annotations),
        }
    result['per_class'] = per_class
    return result


def _xywh_to_xyxy(box):
    x, y, width, height = box
    return x, y, x + width, y + height


def _iou(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def _precision_recall(coco_gt, detections, image_ids, score_threshold, iou_threshold):
    image_id_set = set(image_ids)
    ground_truth = defaultdict(list)
    predictions = defaultdict(list)
    for annotation in coco_gt.dataset['annotations']:
        if annotation['image_id'] in image_id_set:
            ground_truth[(annotation['image_id'], annotation['category_id'])].append(
                _xywh_to_xyxy(annotation['bbox']))
    for detection in detections:
        if (detection['image_id'] in image_id_set and
                detection['score'] >= score_threshold):
            predictions[(detection['image_id'], detection['category_id'])].append(detection)

    true_positives = false_positives = false_negatives = localization_errors = 0
    for key in set(ground_truth) | set(predictions):
        boxes = ground_truth[key]
        matched = set()
        for detection in sorted(predictions[key], key=lambda item: item['score'], reverse=True):
            predicted_box = _xywh_to_xyxy(detection['bbox'])
            overlaps = [_iou(predicted_box, box) for box in boxes]
            candidates = [(overlap, index) for index, overlap in enumerate(overlaps)
                          if index not in matched]
            best_overlap, best_index = max(candidates, default=(0.0, -1))
            if best_overlap >= iou_threshold:
                true_positives += 1
                matched.add(best_index)
            else:
                false_positives += 1
                if (max(overlaps, default=0.0) >= 0.1 and
                        not any(overlap >= iou_threshold for overlap in overlaps)):
                    localization_errors += 1
        false_negatives += len(boxes) - len(matched)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    return {
        'precision': precision,
        'recall': recall,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'localization_errors': localization_errors,
    }


def limit_detections(detections, score_floor, max_per_image):
    by_image = defaultdict(list)
    for detection in detections:
        if float(detection['score']) >= score_floor:
            by_image[int(detection['image_id'])].append(detection)
    limited = []
    for image_detections in by_image.values():
        image_detections.sort(key=lambda item: float(item['score']), reverse=True)
        limited.extend(image_detections[:max_per_image])
    return limited


def evaluate_coco_predictions(coco_gt, detections, records, class_names,
                              score_threshold=0.25, iou_threshold=0.5,
                              prediction_score_floor=0.001,
                              max_detections=100,
                              include_condition_breakdown=True):
    detections = limit_detections(detections, prediction_score_floor, max_detections)
    image_ids = sorted(records)
    result = _coco_metrics(coco_gt, detections, image_ids, max_detections)
    result.update(_precision_recall(
        coco_gt, detections, image_ids, score_threshold, iou_threshold))

    result['per_condition'] = {}
    if include_condition_breakdown:
        by_condition = defaultdict(list)
        for image_id, record in records.items():
            by_condition[record['condition']].append(image_id)
        for condition, condition_ids in sorted(by_condition.items()):
            metrics = _coco_metrics(
                coco_gt, detections, condition_ids, max_detections)
            metrics.update(_precision_recall(
                coco_gt, detections, condition_ids, score_threshold, iou_threshold))
            result['per_condition'][condition] = metrics

    result['evaluation'] = {
        'script': 'pycocotools.COCOeval',
        'pycocotools_version': _pycocotools_version(),
        'iou_type': 'bbox',
        'iou_thresholds': [round(float(value), 2) for value in IOU_THRESHOLDS],
        'max_detections': max_detections,
        'prediction_score_floor': prediction_score_floor,
        'precision_recall_score_threshold': score_threshold,
        'precision_recall_iou_threshold': iou_threshold,
    }
    return result
