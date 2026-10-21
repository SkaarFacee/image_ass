import os
import time

import numpy as np
import torch
from PIL import Image

from utils.utils import cvtColor, preprocess_input, resize_image


def _image_tensor(image, input_shape, device, letterbox=True):
    image = cvtColor(image)
    image_shape = np.asarray([image.height, image.width])
    resized = resize_image(image, (input_shape[1], input_shape[0]), letterbox)
    pixels = np.asarray(resized, dtype=np.float32)
    pixels = np.transpose(preprocess_input(pixels), (2, 0, 1))[None]
    return torch.from_numpy(pixels).to(device), image_shape


def predict_manifest(model, annotation_lines, bbox_util, input_shape, num_classes,
                     device, confidence_threshold, nms_iou_threshold, max_detections,
                     letterbox=True):
    detections = []
    per_image = {}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():
        for image_id, line in enumerate(annotation_lines, 1):
            fields = line.split()
            if not fields:
                continue
            image_path = fields[0]
            with Image.open(image_path) as source:
                image = source.copy()
            tensor, image_shape = _image_tensor(image, input_shape, device, letterbox)
            outputs = bbox_util.decode_box(model(tensor))
            results = bbox_util.non_max_suppression(
                torch.cat(outputs, 1), num_classes, input_shape, image_shape,
                letterbox, conf_thres=confidence_threshold,
                nms_thres=nms_iou_threshold)[0]

            image_detections = []
            if results is not None:
                height, width = image_shape
                for row in results:
                    top, left, bottom, right = map(float, row[:4])
                    left = min(max(left, 0.0), float(width))
                    right = min(max(right, 0.0), float(width))
                    top = min(max(top, 0.0), float(height))
                    bottom = min(max(bottom, 0.0), float(height))
                    if right <= left or bottom <= top:
                        continue
                    image_detections.append({
                        'image_id': image_id,
                        'category_id': int(row[6]) + 1,
                        'bbox': [left, top, right - left, bottom - top],
                        'score': float(row[4] * row[5]),
                    })
            image_detections.sort(key=lambda item: item['score'], reverse=True)
            image_detections = image_detections[:max_detections]
            detections.extend(image_detections)
            per_image[image_id] = image_detections

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    return detections, per_image, elapsed_seconds


def benchmark_forward(model, input_shape, device, warmup_iterations=50,
                      timed_iterations=200):
    model.eval()
    sample = torch.rand(1, 3, input_shape[0], input_shape[1], device=device)
    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(sample)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(timed_iterations):
                model(sample)
            end.record()
            torch.cuda.synchronize(device)
            elapsed_seconds = start.elapsed_time(end) / 1000.0
        else:
            start = time.perf_counter()
            for _ in range(timed_iterations):
                model(sample)
            elapsed_seconds = time.perf_counter() - start

    return {
        'fps': timed_iterations / max(elapsed_seconds, 1e-12),
        'batch_size': 1,
        'warmup_iterations': warmup_iterations,
        'timed_iterations': timed_iterations,
        'image_size': list(input_shape),
        'timing': 'CUDA events' if device.type == 'cuda' else 'time.perf_counter',
        'preprocessing_included': False,
        'postprocessing_included': False,
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'device_index': torch.cuda.current_device() if device.type == 'cuda' else None,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }
