"""Rank training trials using validation results only."""

import argparse
import csv
import json
import os


def _read_run(run_dir):
    with open(os.path.join(run_dir, 'experiment.json'), encoding='utf-8') as stream:
        experiment = json.load(stream)
    validation = experiment.get('validation', {})
    metrics = validation.get('best_metrics') or {}
    if not metrics:
        raise ValueError('{} has no selected validation metrics'.format(run_dir))

    validation_loss = None
    loss_path = os.path.join(run_dir, 'loss.csv')
    if os.path.exists(loss_path):
        with open(loss_path, encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if int(row['epoch']) == validation.get('selected_epoch'):
                    validation_loss = float(row['validation_loss'])
                    break

    parameters = experiment.get('hyperparameters', {})
    return {
        'run': os.path.normpath(run_dir),
        'validation_map_50_95': metrics.get('map_50_95'),
        'validation_map_50': metrics.get('map_50'),
        'validation_precision': metrics.get('precision'),
        'validation_recall': metrics.get('recall'),
        'validation_loss_at_best_epoch': validation_loss,
        'selected_epoch': validation.get('selected_epoch'),
        'epochs': parameters.get('epochs'),
        'batch_size': parameters.get('batch_size'),
        'learning_rate': parameters.get('learning_rate'),
        'optimizer': parameters.get('optimizer'),
        'scheduler': parameters.get('scheduler'),
        'image_size': 'x'.join(map(str, parameters.get('image_size', []))),
        'random_seed': parameters.get('random_seed'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', help='Training run directories under logs-XWOD')
    parser.add_argument('--output-dir', default='results/validation_tuning')
    args = parser.parse_args()

    rows = [_read_run(path) for path in args.run_dirs]
    rows.sort(key=lambda row: (row['validation_map_50_95']
                               if row['validation_map_50_95'] is not None else -1),
              reverse=True)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'validation_trials.csv')
    with open(csv_path, 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = os.path.join(args.output_dir, 'validation_trials.md')
    with open(markdown_path, 'w') as stream:
        stream.write('# Validation-only training trials\n\n')
        stream.write('Ranked by COCO validation mAP@0.5:0.95. The held-out test '
                     'split is not read by this report.\n\n')
        columns = list(rows[0])
        stream.write('| ' + ' | '.join(columns) + ' |\n')
        stream.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
        for row in rows:
            stream.write('| ' + ' | '.join(str(row[column]) for column in columns) + ' |\n')
    print('Wrote validation-only trial reports to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
