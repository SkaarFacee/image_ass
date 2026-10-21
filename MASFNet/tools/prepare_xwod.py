"""Convert YOLO box/polygon XWOD labels to MASFNet manifests and anchors."""

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
SPLIT_KEYS = {'train': 'train', 'valid': 'val', 'test': 'test'}


def load_data_config(data_yaml):
    data_yaml = Path(data_yaml).resolve()
    with data_yaml.open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f'{data_yaml} must contain a YAML mapping')

    class_names = data.get('names')
    if not isinstance(class_names, list) or not class_names or not all(
            isinstance(name, str) and name for name in class_names):
        raise ValueError(f'{data_yaml} must define a non-empty names list')
    if len(set(class_names)) != len(class_names):
        raise ValueError('Class names must be unique')
    if 'nc' in data and int(data['nc']) != len(class_names):
        raise ValueError(f"nc={data['nc']} does not match {len(class_names)} class names")

    root = Path(data.get('path', '.'))
    if not root.is_absolute():
        root = data_yaml.parent / root
    image_dirs = {}
    for split, key in SPLIT_KEYS.items():
        relative = data.get(key)
        if relative is None:
            continue
        image_dir = Path(relative)
        if not image_dir.is_absolute():
            image_dir = root / image_dir
        image_dirs[split] = image_dir.resolve()
    return class_names, image_dirs


def parse_yolo_label_line(line, width, height, num_classes, input_shape, context):
    fields = line.split()
    if not fields:
        raise ValueError(f'{context}: empty annotation line')
    try:
        class_id = int(fields[0])
    except ValueError as exc:
        raise ValueError(f'{context}: invalid class id') from exc

    polygon = len(fields) != 5
    if not polygon:
        try:
            center_x, center_y, box_width, box_height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f'{context}: non-numeric YOLO annotation') from exc
    else:
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f'{context}: expected a box or a polygon with at least 3 points')
        try:
            coordinates = np.asarray(list(map(float, fields[1:])), dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f'{context}: non-numeric YOLO polygon') from exc
        if not np.isfinite(coordinates).all() or (coordinates < 0).any() or (coordinates > 1).any():
            raise ValueError(f'{context}: polygon points must be finite and normalized to [0, 1]')
        x_coordinates = coordinates[0::2]
        y_coordinates = coordinates[1::2]
        left, right = x_coordinates.min(), x_coordinates.max()
        top, bottom = y_coordinates.min(), y_coordinates.max()
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        box_width, box_height = right - left, bottom - top

    if not 0 <= class_id < num_classes:
        raise ValueError(f'{context}: class id {class_id} is outside [0, {num_classes - 1}]')
    values = (center_x, center_y, box_width, box_height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'{context}: box contains a non-finite value')
    if (not 0 <= center_x <= 1 or not 0 <= center_y <= 1 or
            not 0 < box_width <= 1 or not 0 < box_height <= 1):
        raise ValueError(f'{context}: normalized centers and sizes must be in [0, 1]')

    left = center_x - box_width / 2
    right = center_x + box_width / 2
    top = center_y - box_height / 2
    bottom = center_y + box_height / 2
    if right <= 0 or bottom <= 0 or left >= 1 or top >= 1:
        raise ValueError(f'{context}: box does not intersect the image')
    clipped_pixels = max(
        0.0, -left * width, (right - 1) * width,
        -top * height, (bottom - 1) * height)
    left = min(1.0, max(0.0, left))
    right = min(1.0, max(0.0, right))
    top = min(1.0, max(0.0, top))
    bottom = min(1.0, max(0.0, bottom))

    x_min = min(width, max(0, int(round(min(1.0, max(0.0, left)) * width))))
    x_max = min(width, max(0, int(round(min(1.0, max(0.0, right)) * width))))
    y_min = min(height, max(0, int(round(min(1.0, max(0.0, top)) * height))))
    y_max = min(height, max(0, int(round(min(1.0, max(0.0, bottom)) * height))))
    if x_max <= x_min:
        if x_min == width:
            x_min -= 1
        else:
            x_max = x_min + 1
    if y_max <= y_min:
        if y_min == height:
            y_min -= 1
        else:
            y_max = y_min + 1

    input_height, input_width = input_shape
    pixel_box = (x_min, y_min, x_max, y_max, class_id)
    anchor_box = ((right - left) * input_width, (bottom - top) * input_height)
    return pixel_box, anchor_box, clipped_pixels, polygon


def load_split(data_yaml, split, input_shape):
    class_names, image_dirs = load_data_config(data_yaml)
    if split not in image_dirs:
        raise ValueError(f'{data_yaml} does not define the {split} split')
    image_dir = image_dirs[split]
    labels_dir = image_dir.parent / 'labels'
    if not image_dir.is_dir():
        raise FileNotFoundError(f'{split} image directory does not exist: {image_dir}')
    if not labels_dir.is_dir():
        raise FileNotFoundError(f'{split} labels directory does not exist: {labels_dir}')

    images = sorted((path for path in image_dir.iterdir()
                     if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda path: path.name)
    if not images:
        raise ValueError(f'No supported images found in {image_dir}')
    if len({path.stem for path in images}) != len(images):
        raise ValueError(f'{image_dir} contains duplicate image stems')

    label_paths = {path.stem: path for path in labels_dir.glob('*.txt')}
    image_stems = {path.stem for path in images}
    missing = sorted(image_stems - label_paths.keys())
    orphaned = sorted(label_paths.keys() - image_stems)
    if missing or orphaned:
        details = []
        if missing:
            details.append(f'missing labels for {missing[:5]}')
        if orphaned:
            details.append(f'labels without images for {orphaned[:5]}')
        raise ValueError(f'{split} image/label pairing failed: {"; ".join(details)}')

    lines = []
    anchor_boxes = []
    objects = 0
    empty_images = 0
    clipped_boxes = 0
    max_clipping_px = 0.0
    polygon_objects = 0
    for image_path in images:
        image_path = image_path.resolve()
        if any(character.isspace() for character in str(image_path)):
            raise ValueError(f'MASFNet manifests cannot represent whitespace in paths: {image_path}')
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            raise ValueError(f'Could not read {image_path}') from exc
        if width <= 0 or height <= 0:
            raise ValueError(f'{image_path} has invalid dimensions {width}x{height}')

        manifest_boxes = []
        label_path = label_paths[image_path.stem]
        with label_path.open(encoding='utf-8') as stream:
            for line_number, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    continue
                context = f'{label_path}:{line_number}'
                pixel_box, anchor_box, clipped, polygon = parse_yolo_label_line(
                    raw_line, width, height, len(class_names), input_shape, context)
                manifest_boxes.append(','.join(map(str, pixel_box)))
                anchor_boxes.append(anchor_box)
                objects += 1
                if clipped:
                    clipped_boxes += 1
                    max_clipping_px = max(max_clipping_px, clipped)
                polygon_objects += polygon

        if not manifest_boxes:
            empty_images += 1
        lines.append(' '.join([str(image_path)] + manifest_boxes))

    return lines, np.asarray(anchor_boxes, dtype=np.float64).reshape(-1, 2), {
        'images': len(images), 'objects': objects, 'empty_images': empty_images,
        'clipped_boxes': clipped_boxes, 'max_clipping_px': max_clipping_px,
        'polygon_objects': polygon_objects,
    }


def wh_iou(boxes, anchors):
    intersection = np.minimum(boxes[:, None, :], anchors[None, :, :]).prod(axis=2)
    box_areas = boxes.prod(axis=1)[:, None]
    anchor_areas = anchors.prod(axis=1)[None, :]
    return intersection / np.maximum(box_areas + anchor_areas - intersection, 1e-12)


def fit_anchors(boxes, count=6, seed=0, max_iterations=1000):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 2 or len(boxes) < count:
        raise ValueError(f'Need at least {count} width/height boxes to fit anchors')
    if not np.isfinite(boxes).all() or (boxes <= 0).any():
        raise ValueError('Anchor boxes must be finite and positive')

    rng = np.random.default_rng(seed)
    centers = [boxes[int(rng.integers(len(boxes)))].copy()]
    nearest_distance = 1 - wh_iou(boxes, np.asarray(centers)).max(axis=1)
    while len(centers) < count:
        probabilities = nearest_distance ** 2
        total = probabilities.sum()
        if total <= 0:
            selected = {tuple(center) for center in centers}
            candidates = [index for index, box in enumerate(boxes)
                          if tuple(box) not in selected]
            next_index = candidates[0] if candidates else len(centers) % len(boxes)
        else:
            next_index = int(rng.choice(len(boxes), p=probabilities / total))
        centers.append(boxes[next_index].copy())
        nearest_distance = np.minimum(
            nearest_distance, 1 - wh_iou(boxes, np.asarray([centers[-1]])).ravel())

    centers = np.asarray(centers)
    assignments = None
    for _ in range(max_iterations):
        new_assignments = wh_iou(boxes, centers).argmax(axis=1)
        if assignments is not None and np.array_equal(assignments, new_assignments):
            break
        updated = centers.copy()
        for index in range(count):
            members = boxes[new_assignments == index]
            if len(members):
                updated[index] = np.median(members, axis=0)
        centers = updated
        assignments = new_assignments

    areas = centers.prod(axis=1)
    order = np.lexsort((centers[:, 1], centers[:, 0], areas))
    return centers[order]


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def prepare(data_yaml, output_dir, splits, input_shape=(416, 416),
            anchor_count=6, seed=0):
    class_names, _ = load_data_config(data_yaml)
    results = {}
    for split in splits:
        results[split] = load_split(data_yaml, split, input_shape)

    output_dir = Path(output_dir)
    if 'train' in results:
        train_boxes = results['train'][1]
        anchors = fit_anchors(train_boxes, count=anchor_count, seed=seed)
        write_text_atomic(output_dir.parent / 'model_data' / 'xwod_classes.txt',
                          '\n'.join(class_names) + '\n')
        anchor_text = ', '.join(f'{width:.6f},{height:.6f}' for width, height in anchors)
        write_text_atomic(output_dir.parent / 'model_data' / 'xwod_anchors.txt',
                          anchor_text + '\n')
        print(f'Fitted {len(anchors)} anchors from {len(train_boxes)} training boxes (seed={seed})')
        print('Anchors:', anchors.tolist())

    for split, (lines, _, stats) in results.items():
        output_path = output_dir / f'{split}_XWOD.txt'
        write_text_atomic(output_path, '\n'.join(lines) + '\n')
        print(f'{split}: {stats["images"]} images, {stats["objects"]} boxes, '
              f'{stats["polygon_objects"]} polygon labels converted, '
              f'{stats["empty_images"]} empty labels, {stats["clipped_boxes"]} boxes clipped '
              f'(max {stats["max_clipping_px"]:.2f} px) -> {output_path}')


def main():
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-yaml', type=Path,
                        default=workspace_root / 'dataset' / 'data.yaml')
    parser.add_argument('--output-dir', type=Path, default=project_root / 'dataset_split')
    parser.add_argument('--splits', nargs='+', choices=tuple(SPLIT_KEYS),
                        default=['train', 'valid'])
    parser.add_argument('--input-size', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'),
                        default=(416, 416))
    parser.add_argument('--anchors', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.anchors != 6:
        parser.error('MASFNet-XWOD uses exactly six anchors')
    if 'test' in args.splits and set(args.splits) != {'test'}:
        parser.error('Prepare train/valid before training; prepare test separately at final evaluation')
    prepare(args.data_yaml, args.output_dir, args.splits, args.input_size,
            args.anchors, args.seed)


if __name__ == '__main__':
    main()
