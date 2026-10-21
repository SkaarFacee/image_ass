import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils.coco_eval import build_coco_ground_truth, evaluate_coco_predictions
from utils.inference import predict_manifest
from utils.utils_bbox import DecodeBox


class LossHistory:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.train_losses = []
        self.val_losses = []
        os.makedirs(self.log_dir, exist_ok=True)

    def append_loss(self, epoch, train_loss, val_loss):
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        loss_path = os.path.join(self.log_dir, 'loss.csv')
        write_header = not os.path.exists(loss_path)
        with open(loss_path, 'a') as stream:
            writer = csv.writer(stream)
            if write_header:
                writer.writerow(['epoch', 'train_loss', 'validation_loss'])
            writer.writerow([epoch, train_loss, val_loss])
        self.plot()

    def plot(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure()
        plt.plot(epochs, self.train_losses, label='training loss')
        plt.plot(epochs, self.val_losses, label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'loss.png'))
        plt.close()


class EvalCallback:
    def __init__(self, annotation_lines, class_names, input_shape, anchors,
                 anchors_mask, device, log_dir, period, confidence_threshold,
                 nms_iou_threshold, max_detections, pr_confidence_threshold,
                 pr_iou_threshold):
        self.annotation_lines = annotation_lines
        self.class_names = class_names
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.device = device
        self.log_dir = log_dir
        self.period = period
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.pr_confidence_threshold = pr_confidence_threshold
        self.pr_iou_threshold = pr_iou_threshold
        self.coco_gt, self.records = build_coco_ground_truth(
            annotation_lines, class_names)
        self.bbox_util = DecodeBox(anchors, len(class_names), input_shape, anchors_mask)
        self.best_epoch = None
        self.best_map = None
        self.best_metrics = None
        self.history = []

    def on_epoch_end(self, epoch, model):
        if epoch % self.period:
            return None
        detections, _, inference_seconds = predict_manifest(
            model, self.annotation_lines, self.bbox_util, self.input_shape,
            len(self.class_names), self.device, self.confidence_threshold,
            self.nms_iou_threshold, self.max_detections)
        metrics = evaluate_coco_predictions(
            self.coco_gt, detections, self.records, self.class_names,
            score_threshold=self.pr_confidence_threshold,
            iou_threshold=self.pr_iou_threshold,
            prediction_score_floor=self.confidence_threshold,
            max_detections=self.max_detections,
            include_condition_breakdown=False)
        metrics['epoch'] = epoch
        metrics['inference_seconds'] = inference_seconds
        self.history.append(metrics)

        score = metrics['map_50_95']
        if score is not None and (self.best_map is None or score > self.best_map):
            self.best_map = score
            self.best_epoch = epoch
            self.best_metrics = metrics

        self._write_history()
        print('Validation COCO mAP@0.5:0.95={:.4f}, mAP@0.5={:.4f}'.format(
            metrics['map_50_95'] or 0.0, metrics['map_50'] or 0.0))
        return metrics

    def _write_history(self):
        csv_path = os.path.join(self.log_dir, 'validation_metrics.csv')
        with open(csv_path, 'w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['epoch', 'map_50_95', 'map_50', 'precision', 'recall'])
            for row in self.history:
                writer.writerow([
                    row['epoch'], row['map_50_95'], row['map_50'],
                    row['precision'], row['recall']])
        with open(os.path.join(self.log_dir, 'validation_metrics.json'), 'w') as stream:
            json.dump(self.history, stream, indent=2)

        plt.figure()
        plt.plot([row['epoch'] for row in self.history],
                 [row['map_50_95'] for row in self.history], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('COCO mAP@0.5:0.95')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'validation_map.png'))
        plt.close()
