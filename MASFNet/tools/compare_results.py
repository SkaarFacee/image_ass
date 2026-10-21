"""Create a common comparison table from COCO evaluation JSON files."""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def _load(path):
    with open(path, encoding='utf-8') as stream:
        result = json.load(stream)
    if result.get('split') != 'test':
        raise ValueError('{} is not a final test result'.format(path))
    if 'metrics' not in result:
        raise ValueError('{} has no COCO metrics'.format(path))
    return result


def _equal_evaluation_settings(results):
    keys = ('script', 'pycocotools_version', 'iou_type', 'iou_thresholds', 'max_detections',
            'prediction_score_floor', 'precision_recall_score_threshold',
            'precision_recall_iou_threshold')
    reference = results[0]['metrics'].get('evaluation', {})
    missing = [key for key in keys if key not in reference]
    if missing:
        raise ValueError('COCO settings are missing: {}'.format(', '.join(missing)))
    for result in results[1:]:
        current = result['metrics'].get('evaluation', {})
        differences = [key for key in keys
                       if key not in current or current.get(key) != reference.get(key)]
        if differences:
            raise ValueError('COCO settings differ for {}: {}'.format(
                result['model'], ', '.join(differences)))

    inference_keys = ('confidence_threshold', 'nms_iou_threshold', 'max_detections',
                      'precision_recall_confidence_threshold',
                      'precision_recall_iou_threshold', 'input_size', 'letterbox')
    reference = results[0].get('inference_settings', {})
    missing = [key for key in inference_keys if key not in reference]
    if missing:
        raise ValueError('Inference settings are missing: {}'.format(', '.join(missing)))
    for result in results[1:]:
        current = result.get('inference_settings', {})
        differences = [key for key in inference_keys
                       if key not in current or current.get(key) != reference.get(key)]
        if differences:
            raise ValueError('Inference settings differ for {}: {}'.format(
                result['model'], ', '.join(differences)))


def _fps_settings(result):
    efficiency = result.get('efficiency', {})
    benchmark = efficiency.get('model_only', {})
    return (
        efficiency.get('device'),
        benchmark.get('device_index'),
        benchmark.get('cuda_visible_devices'),
        benchmark.get('image_size'),
        benchmark.get('batch_size'),
        benchmark.get('warmup_iterations'),
        benchmark.get('timed_iterations'),
        benchmark.get('timing'),
        benchmark.get('preprocessing_included'),
        benchmark.get('postprocessing_included'),
    )


def _number(value):
    return '' if value is None else value


def _comparison_row(result):
    metrics = result['metrics']
    efficiency = result.get('efficiency', {})
    benchmark = efficiency.get('model_only', {})
    training = result.get('training', {})
    model_size = result.get('model_size', {})
    return {
        'model': result['model'],
        'mAP50:95': _number(metrics.get('map_50_95')),
        'mAP50': _number(metrics.get('map_50')),
        'precision@0.25': _number(metrics.get('precision')),
        'recall@0.25': _number(metrics.get('recall')),
        'AR@100': _number(metrics.get('ar_100')),
        'model_only_fps': _number(efficiency.get('model_only_fps')),
        'end_to_end_fps': _number(efficiency.get('end_to_end_fps')),
        'gpu_hours': _number(training.get('gpu_hours')),
        'total_parameters': _number(model_size.get('total_parameters')),
        'trainable_parameters': _number(model_size.get('trainable_parameters')),
        'state_dict_size_mb': _number(model_size.get('state_dict_size_mb')),
        'gpu': efficiency.get('device', ''),
        'image_size': 'x'.join(map(str, benchmark.get('image_size', []))),
    }


def _detailed_rows(results):
    rows = []
    for result in results:
        metrics = result['metrics']
        for class_name, values in metrics.get('per_class', {}).items():
            rows.append({
                'model': result['model'], 'breakdown': 'class', 'name': class_name,
                'map_50_95': values.get('ap_50_95'), 'map_50': values.get('ap_50'),
                'precision': '', 'recall': '', 'instances': values.get('instances', ''),
            })
        for condition, values in metrics.get('per_condition', {}).items():
            rows.append({
                'model': result['model'], 'breakdown': 'weather', 'name': condition,
                'map_50_95': values.get('map_50_95'), 'map_50': values.get('map_50'),
                'precision': values.get('precision'), 'recall': values.get('recall'),
                'instances': values.get('instances', ''),
            })
            for class_name, class_values in values.get('per_class', {}).items():
                rows.append({
                    'model': result['model'],
                    'breakdown': 'weather_class',
                    'name': '{}:{}'.format(condition, class_name),
                    'map_50_95': class_values.get('ap_50_95'),
                    'map_50': class_values.get('ap_50'),
                    'precision': '', 'recall': '',
                    'instances': class_values.get('instances', ''),
                })
    return rows


def _markdown(results, rows, fps_comparable):
    columns = list(rows[0])
    lines = ['# XWOD model comparison', '',
             '| ' + ' | '.join(columns) + ' |',
             '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for result, row in zip(results, rows):
        values = [str(row[column]) for column in columns]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines.extend(['', 'COCO metrics use the shared pycocotools evaluator and identical settings.'])
    lines.append('FPS conditions matched: {}.'.format('yes' if fps_comparable else 'no'))

    model_names = [result['model'] for result in results]
    conditions = sorted({condition for result in results
                         for condition in result_metrics(result).get('per_condition', {})})
    if conditions:
        lines.extend(['', '## mAP by weather condition', '',
                      '| Condition | ' + ' | '.join(model_names) + ' |',
                      '| ' + ' | '.join(['---'] * (len(model_names) + 1)) + ' |'])
        for condition in conditions:
            values = []
            for result in results:
                metric = result_metrics(result).get('per_condition', {}).get(condition, {})
                value = metric.get('map_50_95')
                values.append('' if value is None else '{:.4f}'.format(value))
            lines.append('| {} | {} |'.format(condition, ' | '.join(values)))

    classes = sorted({name for result in results
                      for name in result_metrics(result).get('per_class', {})})
    if classes:
        lines.extend(['', '## AP by object class', '',
                      '| Class | ' + ' | '.join(
                          '{} AP50:95 / AP50'.format(name) for name in model_names) + ' |',
                      '| ' + ' | '.join(['---'] * (len(model_names) + 1)) + ' |'])
        for class_name in classes:
            values = []
            for result in results:
                metric = result_metrics(result).get('per_class', {}).get(class_name, {})
                ap_50_95, ap_50 = metric.get('ap_50_95'), metric.get('ap_50')
                values.append('{:.4f} / {:.4f}'.format(ap_50_95, ap_50)
                              if ap_50_95 is not None and ap_50 is not None else '')
            lines.append('| {} | {} |'.format(class_name, ' | '.join(values)))

    metric_directions = {
        'mAP50:95': True, 'mAP50': True, 'precision@0.25': True,
        'recall@0.25': True, 'AR@100': True, 'model_only_fps': True,
        'total_parameters': False,
    }
    lines.extend(['', '## Relative strengths and weaknesses', ''])
    for result, row in zip(results, rows):
        strengths, weaknesses = [], []
        for column, higher_is_better in metric_directions.items():
            values = [(other[column], other['model']) for other in rows
                      if isinstance(other[column], (int, float))]
            if len(values) < 2:
                continue
            best = (max if higher_is_better else min)(values)[0]
            worst = (min if higher_is_better else max)(values)[0]
            if row[column] == best:
                strengths.append(column)
            if row[column] == worst:
                weaknesses.append(column)
        condition_scores = []
        for condition, condition_metrics in result_metrics(result).get('per_condition', {}).items():
            score = condition_metrics.get('map_50_95')
            if score is not None:
                condition_scores.append((score, condition))
        weather = ''
        if condition_scores:
            strongest = max(condition_scores)
            weakest = min(condition_scores)
            weather = ' Strongest condition: {} ({:.3f}); weakest: {} ({:.3f}).'.format(
                strongest[1], strongest[0], weakest[1], weakest[0])
        lines.append('- **{}**: strongest relative metrics: {}; weakest: {}.{}'.format(
            row['model'], ', '.join(strengths) or 'not best on reported metrics',
            ', '.join(weaknesses) or 'not worst on reported metrics', weather))
    return '\n'.join(lines) + '\n'


def result_metrics(result):
    return result.get('metrics', {})


def _plot_comparison(results, output_dir):
    labels = [result['model'] for result in results]
    map_50_95 = [result_metrics(result).get('map_50_95') for result in results]
    map_50 = [result_metrics(result).get('map_50') for result in results]
    x = list(range(len(labels)))
    width = 0.38
    plt.figure(figsize=(max(7, len(labels) * 1.4), 5))
    plt.bar([value - width / 2 for value in x],
            [value or 0 for value in map_50_95], width, label='mAP@0.5:0.95')
    plt.bar([value + width / 2 for value in x],
            [value or 0 for value in map_50], width, label='mAP@0.5')
    plt.xticks(x, labels, rotation=20, ha='right')
    plt.ylabel('COCO AP')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_map.png'))
    plt.close()

    conditions = sorted({condition for result in results
                         for condition in result_metrics(result).get('per_condition', {})})
    if not conditions:
        return
    width = 0.8 / max(len(results), 1)
    plt.figure(figsize=(max(8, len(conditions) * 1.2), 5))
    for model_index, result in enumerate(results):
        scores = [result_metrics(result).get('per_condition', {}).get(condition, {}).get(
            'map_50_95') or 0 for condition in conditions]
        offsets = [index - 0.4 + width * (model_index + 0.5)
                   for index in range(len(conditions))]
        plt.bar(offsets, scores, width, label=result['model'])
    plt.xticks(range(len(conditions)), conditions, rotation=20, ha='right')
    plt.ylabel('COCO mAP@0.5:0.95')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'condition_map.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', help='Final test_metrics.json files')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    results = [_load(path) for path in args.results]
    _equal_evaluation_settings(results)
    fps_settings = [_fps_settings(result) for result in results]
    fps_fields_present = all(
        settings[0] not in (None, '') and settings[1] is not None and
        all(value is not None for value in settings[2:])
        for settings in fps_settings)
    fps_comparable = fps_fields_present and all(
        settings == fps_settings[0] for settings in fps_settings[1:])
    rows = [_comparison_row(result) for result in results]

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'comparison.csv')
    with open(csv_path, 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    detail_rows = _detailed_rows(results)
    if detail_rows:
        detail_path = os.path.join(args.output_dir, 'comparison_detailed.csv')
        with open(detail_path, 'w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(detail_rows[0]))
            writer.writeheader()
            writer.writerows(detail_rows)
    markdown_path = os.path.join(args.output_dir, 'comparison.md')
    with open(markdown_path, 'w') as stream:
        stream.write(_markdown(results, rows, fps_comparable))
    _plot_comparison(results, args.output_dir)
    print('Wrote comparison tables and plots to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
