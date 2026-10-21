import csv
import os
from collections import defaultdict

from PIL import Image, ImageDraw


def _xyxy(box):
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


def _analyze_image(annotations, predictions, iou_threshold):
    unmatched = set(range(len(annotations)))
    prediction_status = []
    true_positives = false_positives = localization_errors = 0
    for prediction in sorted(predictions, key=lambda item: item['score'], reverse=True):
        category = prediction['category_id']
        box = _xyxy(prediction['bbox'])
        overlaps = [
            _iou(box, _xyxy(annotation['bbox'])) if annotation['category_id'] == category else 0.0
            for annotation in annotations
        ]
        candidates = [(overlap, index) for index, overlap in enumerate(overlaps)
                      if index in unmatched]
        best_overlap, best_index = max(candidates, default=(0.0, -1))
        if best_overlap >= iou_threshold:
            unmatched.remove(best_index)
            prediction_status.append('TP')
            true_positives += 1
        elif max(overlaps, default=0.0) >= 0.1 and not any(
                overlap >= iou_threshold for overlap in overlaps):
            prediction_status.append('LOCALIZATION')
            false_positives += 1
            localization_errors += 1
        else:
            prediction_status.append('FP')
            false_positives += 1

    return {
        'prediction_status': prediction_status,
        'matched_ground_truth': set(range(len(annotations))) - unmatched,
        'tp': true_positives,
        'fp': false_positives,
        'fn': len(unmatched),
        'localization': localization_errors,
    }


def _draw_case(record, annotations, predictions, analysis, class_names, output_path):
    with Image.open(record['path']) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image)

    for index, annotation in enumerate(annotations):
        box = _xyxy(annotation['bbox'])
        matched = index in analysis['matched_ground_truth']
        color = (25, 170, 65) if matched else (220, 40, 170)
        name = class_names[annotation['category_id'] - 1]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], max(0, box[1] - 14)), 'GT ' + name, fill=color)

    ordered = sorted(predictions, key=lambda item: item['score'], reverse=True)
    colors = {'TP': (25, 170, 65), 'FP': (220, 35, 35), 'LOCALIZATION': (235, 130, 20)}
    for prediction, status in zip(ordered, analysis['prediction_status']):
        box = _xyxy(prediction['bbox'])
        color = colors[status]
        name = class_names[prediction['category_id'] - 1]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], box[1]), '{} {} {:.2f}'.format(status, name, prediction['score']),
                  fill=color)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def save_qualitative_examples(coco_gt, records, predictions_by_image, class_names,
                              output_dir, per_condition=3, iou_threshold=0.5):
    annotations_by_image = defaultdict(list)
    for annotation in coco_gt.dataset['annotations']:
        annotations_by_image[annotation['image_id']].append(annotation)

    analyses = {}
    by_condition = defaultdict(list)
    for image_id, record in records.items():
        annotations = annotations_by_image[image_id]
        predictions = predictions_by_image.get(image_id, [])
        analysis = _analyze_image(annotations, predictions, iou_threshold)
        analyses[image_id] = analysis
        by_condition[record['condition']].append(image_id)

    summary = defaultdict(lambda: {
        'images': 0, 'instances': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'localization': 0,
    })
    manifest_rows = []
    for condition, image_ids in sorted(by_condition.items()):
        successes = []
        failures = []
        for image_id in image_ids:
            record = records[image_id]
            analysis = analyses[image_id]
            annotations = annotations_by_image[image_id]
            counts = summary[condition]
            counts['images'] += 1
            counts['instances'] += len(annotations)
            counts['tp'] += analysis['tp']
            counts['fp'] += analysis['fp'] - analysis['localization']
            counts['fn'] += analysis['fn']
            counts['localization'] += analysis['localization']
            errors = analysis['fp'] + analysis['fn']
            row = (image_id, record, annotations,
                   predictions_by_image.get(image_id, []), analysis, errors)
            if analysis['tp'] and not errors:
                successes.append(row)
            if errors:
                failures.append(row)

        successes.sort(key=lambda row: (-row[4]['tp'], row[1]['path']))
        failures.sort(key=lambda row: (-row[5], row[1]['path']))
        selected = [('success', row) for row in successes[:per_condition]]
        selected.extend(('failures', row) for row in failures[:per_condition])
        for kind, row in selected:
            image_id, record, annotations, predictions, analysis, _ = row
            output_path = os.path.join(output_dir, kind, condition,
                                       os.path.basename(record['path']))
            _draw_case(record, annotations, predictions, analysis, class_names, output_path)
            manifest_rows.append({
                'condition': condition,
                'kind': kind,
                'image': record['path'],
                'output': output_path,
                'tp': analysis['tp'],
                'fp': analysis['fp'],
                'fn': analysis['fn'],
                'localization_errors': analysis['localization'],
            })

    with open(os.path.join(output_dir, 'qualitative_manifest.csv'), 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'condition', 'kind', 'image', 'output', 'tp', 'fp', 'fn', 'localization_errors'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(os.path.join(output_dir, 'failure_summary.csv'), 'w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['condition', 'images', 'instances', 'tp', 'fp', 'fn',
                         'localization_errors', 'dominant_failure'])
        for condition, counts in sorted(summary.items()):
            error_types = {key: counts[key] for key in ('fp', 'fn', 'localization')}
            dominant = max(error_types, key=error_types.get) if any(error_types.values()) else 'none'
            writer.writerow([condition, counts['images'], counts['instances'], counts['tp'],
                             counts['fp'], counts['fn'], counts['localization'], dominant])

    with open(os.path.join(output_dir, 'failure_patterns.md'), 'w') as stream:
        stream.write('# Failure patterns by weather condition\n\n')
        for condition, counts in sorted(summary.items()):
            errors = {'false positives': counts['fp'], 'missed objects': counts['fn'],
                      'poor localization': counts['localization']}
            dominant = max(errors, key=errors.get) if any(errors.values()) else None
            if dominant is None:
                description = 'No errors were recorded at IoU {:.2f}.'.format(iou_threshold)
            else:
                description = '{} were the most frequent issue ({} cases).'.format(
                    dominant, errors[dominant])
            stream.write('- **{}**: {} images, {} objects; {}\n'.format(
                condition, counts['images'], counts['instances'], description))
            examples = [row['output'] for row in manifest_rows
                        if row['condition'] == condition and row['kind'] == 'failures']
            if examples:
                stream.write('  Failure examples: {}\n'.format(', '.join(
                    '`{}`'.format(path) for path in examples)))
        stream.write('\nReview the failure images to add visual explanations '
                     'for weather-specific causes such as contrast, occlusion, or scale.\n')
    return dict(summary)
