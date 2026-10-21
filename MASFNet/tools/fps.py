import json
import os

import torch

from config import *
from nets.yolo import MSFNet_Head
from utils.inference import benchmark_forward
from utils.utils import get_classes


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    if Cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is enabled in config.py, but no CUDA device is available')
    _, num_classes = get_classes(os.path.join(ROOT, classes_path))
    device = torch.device('cuda' if Cuda else 'cpu')
    model = MSFNet_Head(anchors_mask, num_classes, pretrained=False, phi=phi)
    checkpoint = os.path.join(ROOT, checkpoint_path)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    model.to(device).eval()

    result = benchmark_forward(
        model, input_shape, device, fps_warmup_iterations, fps_timed_iterations)
    output_path = os.path.join(ROOT, 'checkpoints', 'MASFNet-XWOD.fps.json')
    result.update({
        'model': 'MASFNet-XWOD',
        'checkpoint': checkpoint,
        'device': torch.cuda.get_device_name(device) if Cuda else 'cpu',
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as stream:
        json.dump(result, stream, indent=2)
    print('Model-only FPS: {:.2f} ({})'.format(result['fps'], result['device']))


if __name__ == '__main__':
    main()
