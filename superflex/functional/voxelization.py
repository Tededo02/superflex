from torch.autograd import Function

from superflex.functional.backend import get_backend

__all__ = ['avg_voxelize']


class AvgVoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
        """
        features = features.contiguous()
        coords = coords.int().contiguous()
        b, c, _ = features.shape
        out, indices, counts = get_backend().avg_voxelize_forward(features, coords, resolution)
        ctx.save_for_backward(indices, counts)
        return out.view(b, c, resolution, resolution, resolution)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]
        indices, counts = ctx.saved_tensors
        grad_features = get_backend().avg_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, counts)
        return grad_features, None, None


def avg_voxelize(features, coords, resolution):
    if features.is_cuda:
        return AvgVoxelization.apply(features, coords, resolution)

    b, c, _ = features.shape
    coords = coords.long()
    indices = coords[:, 0] * resolution ** 2 + coords[:, 1] * resolution + coords[:, 2]
    indices = indices.unsqueeze(1)
    counts = features.new_zeros(b, 1, resolution ** 3)
    counts.scatter_add_(2, indices, features.new_ones(indices.shape))
    # Divide before accumulation to match the CUDA kernel's averaging order.
    normalized = features / counts.gather(2, indices).clamp_min(1)
    out = features.new_zeros(b, c, resolution ** 3)
    out.scatter_add_(2, indices.expand(-1, c, -1), normalized)
    return out.view(b, c, resolution, resolution, resolution)
