import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import *
from nets.yolo import MSFNet_Head
from nets.yolo_training import YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init
from tools.prepare_xwod import load_data_config, load_split, prepare
from utils.callbacks import EvalCallback, LossHistory
from utils.coco_eval import build_coco_ground_truth, evaluate_coco_predictions
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.inference import benchmark_forward, predict_manifest
from utils.pretrained import load_cspdarknet_weights
from utils.qualitative import save_qualitative_examples
from utils.utils import get_anchors, get_classes, seed_everything
from utils.utils_fit import fit_one_epoch
from utils.utils_bbox import DecodeBox


ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = (
    'config.py', 'train.py', 'nets/yolo.py', 'nets/yolo_training.py',
    'tools/fps.py', 'tools/prepare_xwod.py', 'tools/score_coco_predictions.py',
    'tools/compare_results.py', 'tools/compare_validation_runs.py',
    'utils/callbacks.py', 'utils/coco_eval.py',
    'utils/inference.py', 'utils/pretrained.py', 'utils/qualitative.py',
    'utils/utils_fit.py',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def loading_data():
    data_yaml = os.path.abspath(os.path.join(ROOT, '..', 'dataset', 'data.yaml'))
    class_names, _ = load_data_config(data_yaml)
    prepare(data_yaml, os.path.join(ROOT, 'dataset_split'), ['train', 'valid'],
            input_shape, 6, anchor_seed)

    classes, num_classes = get_classes(os.path.join(ROOT, classes_path))
    anchors, num_anchors = get_anchors(os.path.join(ROOT, anchors_path))
    if classes != class_names or num_classes != 6 or num_anchors != 6:
        raise ValueError('XWOD requires the YAML class order and exactly six anchors')

    with open(os.path.join(ROOT, train_annotation_path), encoding='utf-8') as stream:
        train_lines = stream.readlines()
    with open(os.path.join(ROOT, val_annotation_path), encoding='utf-8') as stream:
        val_lines = stream.readlines()

    dataset = YoloDataset(train_lines, input_shape, num_classes,
                          epoch_length=UnFreeze_Epoch, train=True)
    loader = DataLoader(dataset, batch_size=Unfreeze_batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=Cuda, drop_last=True,
                        collate_fn=yolo_dataset_collate)
    val_dataset = YoloDataset(val_lines, input_shape, num_classes,
                              epoch_length=UnFreeze_Epoch, train=False)
    val_loader = DataLoader(val_dataset, batch_size=Unfreeze_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=Cuda, drop_last=False,
                            collate_fn=yolo_dataset_collate)
    if not len(loader) or not len(val_loader):
        raise ValueError('Training and validation splits must contain data')

    return {
        'data_yaml': data_yaml,
        'class_names': classes,
        'num_classes': num_classes,
        'anchors': anchors,
        'train_loader': loader,
        'val_loader': val_loader,
        'val_lines': val_lines,
    }


def loading_model(data):
    if Cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is enabled in config.py, but no CUDA device is available')

    model = MSFNet_Head(anchors_mask, data['num_classes'], pretrained=False, phi=phi)
    weights_init(model)
    transfer_report = load_cspdarknet_weights(model, os.path.join(ROOT, model_path))
    model = model.to(torch.device('cuda' if Cuda else 'cpu'))
    return model, transfer_report


def train_model(model, data, transfer_report):
    batch_size = Unfreeze_batch_size
    learning_rate = Init_lr * batch_size / 64
    min_learning_rate = Min_lr * batch_size / 64
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay,
                                nesterov=True)
    scheduler = get_lr_scheduler(lr_decay_type, learning_rate,
                                 min_learning_rate, UnFreeze_Epoch)
    yolo_loss = YOLOLoss(data['anchors'], data['num_classes'], input_shape,
                         Cuda, anchors_mask)

    run_name = datetime.datetime.now().strftime('loss_%Y%m%d_%H%M%S_%f')
    log_dir = os.path.join(ROOT, save_dir, run_name)
    loss_history = LossHistory(log_dir)
    device = next(model.parameters()).device
    eval_callback = EvalCallback(
        data['val_lines'], data['class_names'], input_shape, data['anchors'],
        anchors_mask, device, log_dir, eval_period, confidence_threshold,
        nms_iou_threshold, max_detections, pr_confidence_threshold,
        pr_iou_threshold)

    started_at = datetime.datetime.utcnow().isoformat() + 'Z'
    training_start = time.perf_counter()
    for epoch in range(UnFreeze_Epoch):
        set_optimizer_lr(optimizer, scheduler, epoch)
        fit_one_epoch(
            model, yolo_loss, loss_history, eval_callback, optimizer, epoch,
            len(data['train_loader']), data['train_loader'], data['val_loader'],
            UnFreeze_Epoch, log_dir)
    elapsed_seconds = time.perf_counter() - training_start

    if eval_callback.best_epoch is None:
        raise RuntimeError('Validation did not select a best checkpoint')
    best_path = os.path.join(log_dir, 'best_epoch_weights.pth')
    parameters = list(model.parameters())
    state = model.state_dict()
    gpu_count = 1 if Cuda else 0
    pretrained_path = os.path.join(ROOT, model_path)
    metadata = {
        'model': 'MASFNet-XWOD',
        'dataset': 'XWOD',
        'data_yaml': os.path.relpath(data['data_yaml'], ROOT),
        'data_yaml_sha256': sha256_file(data['data_yaml']),
        'official_splits': {'train': 'train', 'validation': 'valid', 'test': 'test'},
        'split_image_counts': {
            'train': len(data['train_loader'].dataset),
            'validation': len(data['val_loader'].dataset),
            'test': None,
        },
        'manifest_sha256': {
            'train': sha256_file(os.path.join(ROOT, train_annotation_path)),
            'validation': sha256_file(os.path.join(ROOT, val_annotation_path)),
        },
        'class_names': data['class_names'],
        'anchors': data['anchors'].tolist(),
        'anchors_mask': anchors_mask,
        'anchor_fit': {'source_split': 'train', 'random_seed': anchor_seed},
        'input_size': list(input_shape),
        'pretrained': {
            'file': os.path.relpath(pretrained_path, ROOT),
            'source': pretrained_source_url,
            'sha256': sha256_file(pretrained_path),
            'strategy': (
                'Initialize MASFNet, load shape-matched COCO tensors only into '
                'the shared CSPDarknet stages, then fine-tune all parameters on XWOD.'),
            'loaded_tensor_count': transfer_report['loaded_count'],
            'expected_tensor_count': transfer_report['expected_count'],
            'skipped_tensor_count': len(transfer_report['skipped']),
        },
        'hyperparameters': {
            'epochs': UnFreeze_Epoch,
            'batch_size': Unfreeze_batch_size,
            'learning_rate': learning_rate,
            'minimum_learning_rate': min_learning_rate,
            'base_learning_rate': Init_lr,
            'base_minimum_learning_rate': Min_lr,
            'optimizer': 'SGD',
            'nesterov': True,
            'scheduler': lr_decay_type,
            'scheduler_settings': {
                'warmup_iters_ratio': 0.05,
                'warmup_lr_ratio': 0.1,
                'no_aug_iter_ratio': 0.05,
            },
            'momentum': momentum,
            'weight_decay': weight_decay,
            'image_size': list(input_shape),
            'random_seed': seed,
            'data_loader_workers': num_workers,
            'drop_last': True,
            'validation_interval': eval_period,
        },
        'model_size': {
            'total_parameters': sum(parameter.numel() for parameter in parameters),
            'trainable_parameters': sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad),
            'state_dict_size_mb': sum(
                tensor.numel() * tensor.element_size() for tensor in state.values()) / (1024 ** 2),
            'checkpoint_size_mb': os.path.getsize(best_path) / (1024 ** 2),
        },
        'training': {
            'started_utc': started_at,
            'elapsed_seconds': elapsed_seconds,
            'gpu_count': gpu_count,
            'gpu_hours': elapsed_seconds * gpu_count / 3600.0 if Cuda else None,
            'device': torch.cuda.get_device_name(device) if Cuda else 'cpu',
            'device_index': torch.cuda.current_device() if Cuda else None,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'epochs_completed': UnFreeze_Epoch,
        },
        'software': {
            'python': platform.python_version(),
            'pytorch': str(torch.__version__),
            'cuda': torch.version.cuda,
            'pycocotools': eval_callback.best_metrics['evaluation']['pycocotools_version'],
        },
        'code_sha256': {
            path: sha256_file(os.path.join(ROOT, path)) for path in CODE_FILES
        },
        'validation': {
            'selection_metric': 'COCO mAP@0.5:0.95',
            'selected_epoch': eval_callback.best_epoch,
            'best_metrics': eval_callback.best_metrics,
        },
        'inference_settings': {
            'confidence_threshold': confidence_threshold,
            'nms_iou_threshold': nms_iou_threshold,
            'max_detections': max_detections,
            'precision_recall_confidence_threshold': pr_confidence_threshold,
            'precision_recall_iou_threshold': pr_iou_threshold,
            'input_size': list(input_shape),
            'letterbox': True,
            'coco_iou_thresholds': [round(float(value), 2)
                                    for value in eval_callback.best_metrics['evaluation']['iou_thresholds']],
        },
        'checkpoint': os.path.relpath(best_path, ROOT),
    }
    with open(os.path.join(log_dir, 'experiment.json'), 'w') as stream:
        json.dump(metadata, stream, indent=2)
    print('Best validation COCO mAP@0.5:0.95={:.4f} at epoch {}'.format(
        eval_callback.best_map, eval_callback.best_epoch))
    return {'checkpoint': best_path, 'run_dir': log_dir, 'metadata': metadata}


def eval_model(data, run):
    if run['metadata'].get('test_metrics') is not None:
        raise RuntimeError('This run has already used the held-out test split')
    if (run['metadata'].get('class_names') != data['class_names'] or
            run['metadata'].get('anchors_mask') != anchors_mask or
            run['metadata'].get('input_size') != list(input_shape)):
        raise ValueError('Evaluation configuration differs from the selected training run')
    if run['metadata'].get('data_yaml_sha256') != sha256_file(data['data_yaml']):
        raise ValueError('Dataset YAML changed after the selected run was trained')
    test_lines, _, _ = load_split(data['data_yaml'], 'test', input_shape)
    coco_gt, records = build_coco_ground_truth(test_lines, data['class_names'])
    device = torch.device('cuda' if Cuda else 'cpu')
    model = MSFNet_Head(anchors_mask, data['num_classes'], pretrained=False, phi=phi)
    model.load_state_dict(torch.load(run['checkpoint'], map_location='cpu'), strict=True)
    model.to(device).eval()
    bbox_util = DecodeBox(data['anchors'], data['num_classes'], input_shape, anchors_mask)
    predictions, predictions_by_image, elapsed = predict_manifest(
        model, test_lines, bbox_util, input_shape, data['num_classes'], device,
        confidence_threshold, nms_iou_threshold, max_detections)
    metrics = evaluate_coco_predictions(
        coco_gt, predictions, records, data['class_names'],
        score_threshold=pr_confidence_threshold, iou_threshold=pr_iou_threshold,
        prediction_score_floor=confidence_threshold,
        max_detections=max_detections,
        include_condition_breakdown=True)
    benchmark = benchmark_forward(
        model, input_shape, device, fps_warmup_iterations, fps_timed_iterations)
    gpu_name = torch.cuda.get_device_name(device) if Cuda else 'cpu'
    efficiency = {
        'device': gpu_name,
        'model_only_fps': benchmark['fps'],
        'model_only': benchmark,
        'end_to_end_fps': len(records) / max(elapsed, 1e-12),
        'end_to_end': {
            'elapsed_seconds': elapsed,
            'images': len(records),
            'batch_size': 1,
            'image_size': list(input_shape),
            'timing': 'wall clock over the full split',
            'preprocessing_included': True,
            'postprocessing_included': True,
            'warmup_iterations': 0,
        },
    }
    inference_settings = run['metadata']['inference_settings']
    inference_settings['fps_benchmark'] = benchmark
    inference_settings['fps_device'] = gpu_name

    qualitative_dir = os.path.join(run['run_dir'], 'qualitative')
    failure_summary = save_qualitative_examples(
        coco_gt, records, predictions_by_image, data['class_names'], qualitative_dir,
        iou_threshold=pr_iou_threshold)
    final_checkpoint = os.path.join(ROOT, checkpoint_path)
    os.makedirs(os.path.dirname(final_checkpoint), exist_ok=True)
    shutil.copy2(run['checkpoint'], final_checkpoint)
    run['checkpoint'] = final_checkpoint
    run['metadata']['checkpoint'] = os.path.relpath(final_checkpoint, ROOT)
    run['metadata']['model_size']['checkpoint_size_mb'] = (
        os.path.getsize(final_checkpoint) / (1024 ** 2))
    run['metadata']['checkpoint_sha256'] = sha256_file(final_checkpoint)
    result = {
        'schema_version': 1,
        'model': run['metadata']['model'],
        'split': 'test',
        'checkpoint': run['metadata']['checkpoint'],
        'checkpoint_sha256': run['metadata']['checkpoint_sha256'],
        'metrics': metrics,
        'efficiency': efficiency,
        'training': run['metadata']['training'],
        'model_size': run['metadata']['model_size'],
        'inference_settings': inference_settings,
        'failure_summary': failure_summary,
    }
    with open(os.path.join(run['run_dir'], 'test_predictions_coco.json'), 'w') as stream:
        json.dump(predictions, stream)
    with open(os.path.join(run['run_dir'], 'test_metrics.json'), 'w') as stream:
        json.dump(result, stream, indent=2)

    run['metadata']['test_metrics'] = metrics
    run['metadata']['split_image_counts']['test'] = len(records)
    run['metadata']['manifest_sha256']['test'] = hashlib.sha256(
        ''.join(test_lines).encode('utf-8')).hexdigest()
    run['metadata']['efficiency'] = efficiency
    with open(os.path.join(run['run_dir'], 'experiment.json'), 'w') as stream:
        json.dump(run['metadata'], stream, indent=2)
    with open(os.path.join(ROOT, metadata_path), 'w') as stream:
        json.dump(run['metadata'], stream, indent=2)
    print('Held-out test COCO mAP@0.5:0.95={:.4f}, mAP@0.5={:.4f}'.format(
        metrics['map_50_95'] or 0.0, metrics['map_50'] or 0.0))
    print('Model-only FPS: {:.2f} on {}'.format(benchmark['fps'], gpu_name))
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Train on train/validation; evaluate test only after selecting a run')
    parser.add_argument('--eval-only', metavar='RUN_DIR',
                        help='run final test evaluation for a validation-selected training run')
    args = parser.parse_args()
    seed_everything(seed)

    if args.eval_only:
        run_dir = args.eval_only
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(ROOT, run_dir)
        with open(os.path.join(run_dir, 'experiment.json'), encoding='utf-8') as stream:
            metadata = json.load(stream)
        data_yaml = os.path.abspath(os.path.join(ROOT, '..', 'dataset', 'data.yaml'))
        class_names, _ = load_data_config(data_yaml)
        if class_names != metadata.get('class_names'):
            raise ValueError('Dataset class order differs from the selected training run')
        checkpoint = metadata['checkpoint']
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(ROOT, checkpoint)
        data = {
            'data_yaml': data_yaml,
            'class_names': class_names,
            'num_classes': len(class_names),
            'anchors': np.asarray(metadata['anchors'], dtype=np.float32),
        }
        print('Evaluating the validation-selected checkpoint on test')
        eval_model(data, {'checkpoint': checkpoint, 'run_dir': run_dir, 'metadata': metadata})
        return

    print('Loading data')
    data = loading_data()
    print('Loading model')
    model, transfer_report = loading_model(data)
    print('Training using train and validation only')
    run = train_model(model, data, transfer_report)
    print('Validation-selected run: {}'.format(run['run_dir']))
    print('The test split was not read. Use --eval-only after choosing a run.')


if __name__ == '__main__':
    main()
