import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.autograd import gradcheck

from demo_viser import NUM_POINTS, load_points
from superflex.data.normalization import normalize_points, denormalize_points, denormalize_outdict
from superflex.functional import avg_voxelize, trilinear_devoxelize
from superflex.functional.backend import get_backend
from superflex.models.decoder import SinusoidalPositionalEncoding, LearnablePositionalEncoding


class PortableOperationsTests(unittest.TestCase):
    def test_voxel_averages_empty_cells_and_batches(self):
        features = torch.tensor([[[2., 6., 10.]], [[3., 9., 15.]]], requires_grad=True)
        coords = torch.tensor([[[0, 0, 1], [0, 0, 1], [0, 0, 1]]] * 2)
        out = avg_voxelize(features, coords, 2)
        expected = torch.zeros(2, 1, 2, 2, 2)
        expected[:, 0, 0, 0, 0] = torch.tensor([4., 6.])
        expected[:, 0, 1, 1, 1] = torch.tensor([10., 15.])
        torch.testing.assert_close(out, expected)
        out.sum().backward()
        torch.testing.assert_close(features.grad, torch.tensor([[[.5, .5, 1.]]] * 2))
        self.assertEqual(get_backend.cache_info().currsize, 0)

    def test_trilinear_polynomial_and_upper_boundary(self):
        axis = torch.arange(3, dtype=torch.float64)
        x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
        field = (x * y * z + 2 * x + 3 * y + 5 * z)[None, None]
        coords = torch.tensor([[[0., .25, 2., 2.], [0., 1.5, 2., .5], [0., .75, 2., 1.]]], dtype=torch.float64)
        out = trilinear_devoxelize(field, coords, 3, is_training=False)
        x, y, z = coords.unbind(dim=1)
        torch.testing.assert_close(out[:, 0], x * y * z + 2 * x + 3 * y + 5 * z)

    def test_feature_gradients(self):
        features = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
        coords = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, 1]]])
        self.assertTrue(gradcheck(lambda values: avg_voxelize(values, coords, 2), (features,)))
        voxels = torch.randn(1, 2, 2, 2, 2, dtype=torch.float64, requires_grad=True)
        positions = torch.tensor([[[.2, 1.], [.3, .4], [.7, 1.]]], dtype=torch.float64)
        self.assertTrue(gradcheck(lambda values: trilinear_devoxelize(values, positions, 2), (voxels,)))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'Apple MPS is unavailable in this process')
    def test_mps_matches_cpu(self):
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(2, 3, 32, generator=generator)
        coords = torch.randint(0, 4, (2, 3, 32), generator=generator)
        expected = avg_voxelize(features, coords, 4)
        actual = avg_voxelize(features.to('mps'), coords.to('mps'), 4)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)
        positions = torch.rand(2, 3, 32, generator=generator) * 3
        expected = trilinear_devoxelize(expected, positions, 4)
        actual = trilinear_devoxelize(actual, positions.to('mps'), 4)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)


class InputAndNormalizationTests(unittest.TestCase):
    def test_z_up_roundtrip_preserves_world_translation(self):
        points = np.array([[10., 20., 30.], [12., 24., 36.]], dtype=np.float32)
        normalized, translation, scale = normalize_points(points)
        rotation = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        y_up = torch.from_numpy(normalized) @ rotation
        restored = denormalize_points(y_up[None], translation[None], np.array([scale]), z_up=True)
        torch.testing.assert_close(restored[0], torch.from_numpy(points))
        outdict = {'trans': y_up[None], 'scale': torch.ones(1, 2, 3),
                   'rotate': rotation.T.expand(1, 2, 3, 3), 'bending_k': torch.ones(1, 2, 3)}
        restored_outdict = denormalize_outdict(outdict, translation[None], np.array([scale]), z_up=True)
        torch.testing.assert_close(restored_outdict['trans'][0], torch.from_numpy(points))
        torch.testing.assert_close(restored_outdict['rotate'], torch.eye(3).expand(1, 2, 3, 3))
        torch.testing.assert_close(restored_outdict['bending_k'], torch.full((1, 2, 3), 1 / float(scale)))

    def test_sampling_is_repeatable_and_accepts_small_clouds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small.ply'
            vertices = np.array([[1., 2., 3.], [4., 5., 6.]])
            trimesh.points.PointCloud(vertices).export(path)
            first = load_points(path, 42)
            second = load_points(path, 42)
            self.assertEqual(first.shape, (NUM_POINTS, 3))
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.isin(first[:, 0], [1., 4.]).all())

    def test_mesh_input_samples_surfaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'box.glb'
            trimesh.creation.box().export(path)
            points = load_points(path, 42)
            self.assertEqual(points.shape, (NUM_POINTS, 3))
            np.testing.assert_allclose(np.abs(points).max(axis=1), .5)

    def test_empty_cloud_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.ply'
            path.write_text('ply\nformat ascii 1.0\nelement vertex 0\n'
                            'property float x\nproperty float y\nproperty float z\nend_header\n')
            with self.assertRaises(ValueError):
                load_points(path, 42)

    def test_positional_encoding_starts_on_cpu(self):
        self.assertEqual(SinusoidalPositionalEncoding(8, 2).pe.device.type, 'cpu')
        encoding = LearnablePositionalEncoding(8, 2)
        self.assertIn('positional_encoding', dict(encoding.named_parameters()))
        self.assertEqual(encoding.positional_encoding.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
