"""Score a COCO-format detection JSON with the shared XWOD evaluator."""

import argparse
import json
import os
from collections import defaultdict

from config import *
from tools.prepare_xwod import load_data_config, load_split
from utils.coco_eval import (build_coco_ground_truth, evaluate_coco_predictions,
                             limit_detections)
from utils.qualitative import save_qualitative_examples


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True,
                        help='COCO detections: image_id, category_id, bbox [x,y,w,h], score')
    parser.add_argument('--model', required=True, help='Model name for the comparison table')
    parser.add_argument('--split', choices=('valid', 'test'), default='valid')
    parser.add_argument('--final-selected', action='store_true',
                        help='confirm that validation selection is complete before test scoring')
    parser.add_argument('--metadata', help='Optional JSON with training/efficiency metadata')
    parser.add_argument('--output', help='Output metrics JSON path')
    args = parser.parse_args()
    if args.split == 'test' and not args.final_selected:
        parser.error('test scoring requires --final-selected after validation selection')

    data_yaml = os.path.abspath(os.path.join(ROOT, '..', 'dataset', 'data.yaml'))
    class_names, _ = load_data_config(data_yaml)
    annotation_lines, _, _ = load_split(data_yaml, args.split, input_shape)
    coco_gt, records = build_coco_ground_truth(annotation_lines, class_names)

    with open(args.predictions, encoding='utf-8') as stream:
        prediction_data = json.load(stream)
    detections = prediction_data.get('predictions', []) if isinstance(prediction_data, dict) else prediction_data
    id_by_file = {os.path.basename(record['path']): image_id
                  for image_id, record in records.items()}
    normalized = []
    for detection in detections:
        detection = dict(detection)
        if 'image_id' not in detection:
            file_name = os.path.basename(detection.pop('file_name'))
            detection['image_id'] = id_by_file[file_name]
        else:
            detection['image_id'] = int(detection['image_id'])
        detection['category_id'] = int(detection['category_id'])
        if not 1 <= detection['category_id'] <= len(class_names):
            raise ValueError('category_id must be one-based and follow the XWOD class order')
        normalized.append(detection)
    normalized = limit_detections(normalized, confidence_threshold, max_detections)
    predictions_by_image = defaultdict(list)
    for detection in normalized:
        predictions_by_image[detection['image_id']].append(detection)
    metrics = evaluate_coco_predictions(
        coco_gt, normalized, records, class_names,
        score_threshold=pr_confidence_threshold,
        iou_threshold=pr_iou_threshold,
        prediction_score_floor=confidence_threshold,
        max_detections=max_detections,
        include_condition_breakdown=True)

    model_metadata = {}
    if args.metadata:
        with open(args.metadata, encoding='utf-8') as stream:
            model_metadata = json.load(stream)
    expected_settings = {
        'confidence_threshold': confidence_threshold,
        'nms_iou_threshold': nms_iou_threshold,
        'max_detections': max_detections,
        'precision_recall_confidence_threshold': pr_confidence_threshold,
        'precision_recall_iou_threshold': pr_iou_threshold,
        'input_size': list(input_shape),
        'letterbox': True,
    }
    inference_settings = dict(expected_settings)
    inference_settings.update(model_metadata.get('inference_settings', {}))
    for name, expected in expected_settings.items():
        actual = inference_settings.get(name)
        if actual != expected:
            raise ValueError('Model inference setting {}={} differs from shared value {}'.format(
                name, actual, expected))
    result = {
        'schema_version': 1,
        'model': args.model,
        'split': args.split,
        'metrics': metrics,
        'training': model_metadata.get('training', {}),
        'model_size': model_metadata.get('model_size', {}),
        'efficiency': model_metadata.get('efficiency', {}),
        'inference_settings': inference_settings,
    }
    output_path = args.output or os.path.join(
        ROOT, 'results', '{}_{}_metrics.json'.format(args.model.replace('/', '_'), args.split))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    qualitative_dir = os.path.splitext(output_path)[0] + '_qualitative'
    result['failure_summary'] = save_qualitative_examples(
        coco_gt, records, predictions_by_image, class_names, qualitative_dir,
        iou_threshold=pr_iou_threshold)
    with open(output_path, 'w') as stream:
        json.dump(result, stream, indent=2)
    print('Wrote {} COCO metrics to {}'.format(args.split, output_path))


if __name__ == '__main__':
    main()
