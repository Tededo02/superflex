from itertools import product

from torch.autograd import Function

from superflex.functional.backend import get_backend

__all__ = ['trilinear_devoxelize']


class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """
        :param ctx:
        :param coords: the coordinates of points, FloatTensor[B, 3, N]
        :param features: FloatTensor[B, C, R, R, R]
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1)
        coords = coords[:,:3].contiguous()
        outs, inds, wgts = get_backend().trilinear_devoxelize_forward(
            resolution, is_training, coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: 
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = get_backend().trilinear_devoxelize_backward(
            grad_output.contiguous(), inds, wgts, ctx.r)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1),
                                ctx.r, ctx.r, ctx.r), None, None, None


def trilinear_devoxelize(features, coords, resolution, is_training=True):
    if features.is_cuda:
        return TrilinearDevoxelization.apply(features, coords, resolution, is_training)

    b, c = features.shape[:2]
    features = features.reshape(b, c, -1)
    # The original kernel differentiates features, but not point coordinates.
    coords = coords[:, :3].detach()
    lower_float = coords.floor()
    fraction = coords - lower_float
    lower = lower_float.long()
    # At an integer boundary, use the same voxel for both interpolation ends.
    upper = lower + (fraction > 0).long()
    out = features.new_zeros(b, c, coords.shape[-1])
    for x, y, z in product((0, 1), repeat=3):
        ix = upper[:, 0] if x else lower[:, 0]
        iy = upper[:, 1] if y else lower[:, 1]
        iz = upper[:, 2] if z else lower[:, 2]
        wx = fraction[:, 0] if x else 1 - fraction[:, 0]
        wy = fraction[:, 1] if y else 1 - fraction[:, 1]
        wz = fraction[:, 2] if z else 1 - fraction[:, 2]
        indices = (ix * resolution ** 2 + iy * resolution + iz).unsqueeze(1)
        sampled = features.gather(2, indices.expand(-1, c, -1))
        out = out + sampled * (wx * wy * wz).unsqueeze(1)
    return out
