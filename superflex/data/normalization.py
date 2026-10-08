import numpy as np
import torch


def normalize_points(points):
    translation = points.mean(0)
    points = points - translation
    scale = max(2 * np.max(np.abs(points)), 1e-4)
    return points / scale, translation, scale


def _normalization_tensors(reference, translation, scale):
    translation = torch.as_tensor(translation, dtype=reference.dtype, device=reference.device)
    scale = torch.as_tensor(scale, dtype=reference.dtype, device=reference.device)
    return translation[:, None, :], scale[:, None, None]


def _y_up_to_z_up(reference):
    return reference.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def denormalize_points(points, translation, scale, z_up=False):
    translation, scale = _normalization_tensors(points, translation, scale)
    if z_up:
        points = points @ _y_up_to_z_up(points).T
    # Undo rotation before restoring the original world-space translation.
    return points * scale + translation


def denormalize_outdict(outdict, translation, scale, z_up=False):
    translation, scale = _normalization_tensors(outdict['trans'], translation, scale)
    outdict['scale'] = outdict['scale'] * scale
    if z_up:
        rotation = _y_up_to_z_up(outdict['trans'])
        outdict['trans'] = outdict['trans'] @ rotation.T
        outdict['rotate'] = rotation @ outdict['rotate']
    outdict['trans'] = outdict['trans'] * scale + translation
    if 'bending_k' in outdict:
        outdict['bending_k'] = outdict['bending_k'] / scale
    return outdict
