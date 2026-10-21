"""Semantically restrict COCO initialization to shared CSPDarknet stages."""

import torch


TRANSFER_PREFIXES = tuple(
    f'backbone.{module}.' for module in (
        'conv1', 'conv2', 'resblock_body1', 'resblock_body2',
        'resblock_body3', 'conv3',
    )
)


def is_transferable_key(key):
    return key.startswith(TRANSFER_PREFIXES)


def load_cspdarknet_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise TypeError(f'{checkpoint_path} must contain a raw state dict')

    model_state = model.state_dict()
    expected = {key for key in model_state if is_transferable_key(key)}
    if not expected:
        raise RuntimeError('The model has no tensors in the CSPDarknet transfer allowlist')

    copied = {}
    skipped = {}
    for key, value in checkpoint.items():
        if not is_transferable_key(key):
            skipped[key] = 'excluded by semantic allowlist'
        elif key not in model_state:
            skipped[key] = 'no matching model tensor'
        elif not isinstance(value, torch.Tensor):
            skipped[key] = 'checkpoint value is not a tensor'
        elif model_state[key].shape != value.shape:
            skipped[key] = f'shape mismatch: source={tuple(value.shape)} model={tuple(model_state[key].shape)}'
        else:
            copied[key] = value

    missing = sorted(expected - copied.keys())
    if missing:
        for key in missing:
            print(f'[pretrained] missing expected backbone tensor {key}')
        raise RuntimeError(
            f'Pretrained checkpoint missed {len(missing)} expected CSPDarknet tensors')

    print('[pretrained] loaded {}/{} CSPDarknet tensors; skipped {}'.format(
        len(copied), len(expected), len(skipped)))
    model_state.update(copied)
    model.load_state_dict(model_state, strict=True)
    return {
        'loaded_keys': sorted(copied),
        'skipped': skipped,
        'loaded_count': len(copied),
        'expected_count': len(expected),
    }
