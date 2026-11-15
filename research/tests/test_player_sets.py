"""Geometry invariants required by the experimental anonymous-player model."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from player_nodes import pack_player_nodes

try:
    import torch
except ImportError:
    torch = None


class PlayerGeometryTests(unittest.TestCase):
    def test_velocity_period_boundary_and_padding(self):
        xy = np.zeros((40, 2, 2), np.float32)
        xy[:, 0, 0] = 10 + np.arange(40)*.04
        xy[:, 1, 0] = 90
        xy[:, :, 1] = 34
        period = np.r_[np.ones(20), np.full(20, 2)]
        nodes = pack_player_nodes(xy, [0, 1], period)
        self.assertEqual(nodes.shape, (8, 2, 11, 8))
        self.assertAlmostEqual(float(nodes[2, 0, 0, 2]), .1, places=5)
        self.assertEqual(nodes[4, 0, 0, 5], 0)
        self.assertTrue(np.all(nodes[:, :, 1:] == 0))

    def test_empty_geometry_is_finite(self):
        nodes = pack_player_nodes(np.full((30, 2, 2), np.nan), [0, 1], np.ones(30))
        self.assertTrue(np.isfinite(nodes).all())
        self.assertFalse(nodes.any())

    @unittest.skipIf(torch is None, 'optional neural research dependencies absent')
    def test_residual_numpy_matches_team_swap_and_period_inference(self):
        from features import empty_feature_arrays
        from residual_set_temporal import ResidualSetTemporalNet, predict_residual_logits
        from numpy_residual_set import predict_logits
        torch.set_num_threads(1); torch.manual_seed(13)
        net = ResidualSetTemporalNet(width=8,layers=2).eval()
        n = 303; rng = np.random.RandomState(13)
        f = empty_feature_arrays(n)
        f['player_motion'] = np.zeros((n,12),np.float32)
        f['period'] = np.r_[np.ones(153),np.full(n-153,2)].astype(np.int8)
        f['ball_present'][::11] = True
        f['ball_x_norm'][::11] = rng.rand(len(f['ball_x_norm'][::11]))
        f['ball_y_norm'][::11] = rng.rand(len(f['ball_y_norm'][::11]))
        nodes = rng.rand((n+4)//5,2,11,8).astype(np.float32)
        nodes[...,6] = nodes[...,6] > .25
        reference = np.where(np.arange(n)%100 < 50,'H','A')
        reference[70:80] = 'D'
        graded = np.ones(n,bool); graded[150:153] = False
        mean,scale = np.zeros(57,np.float32),np.ones(57,np.float32)
        p = {key:value.numpy() for key,value in net.state_dict().items()}
        p['temporal.project.weight'] = p['temporal.project.weight'][:,np.r_[np.arange(57),157,np.arange(57,157)]]
        p.update(mean=mean,scale=scale,layers=np.array(2),baseline_prior=np.array(net.prior))
        expected = predict_residual_logits(f,graded,nodes,reference,(net,mean,scale))
        actual = predict_logits(f,graded,nodes,reference,p)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)

    @unittest.skipIf(torch is None, 'optional neural research dependencies absent')
    def test_numpy_export_matches_temporal_convolutions(self):
        from set_temporal_team_model import SetTemporalTeamNet
        from numpy_set_temporal import forward
        torch.set_num_threads(1)
        torch.manual_seed(9)
        net = SetTemporalTeamNet(width=8, layers=3).eval()
        p = {key: value.numpy() for key, value in net.state_dict().items()}
        p['layers'] = np.array(3)
        rng = np.random.RandomState(9)
        x = rng.randn(57, 151).astype(np.float32)
        nodes = rng.rand(151, 2, 11, 8).astype(np.float32)
        nodes[..., 6] = nodes[..., 6] > .5
        ctx = rng.rand(151, 8).astype(np.float32)
        with torch.inference_mode():
            expected = net(torch.from_numpy(x)[None], torch.from_numpy(nodes)[None],
                           torch.from_numpy(ctx)[None])[0].numpy()
        np.testing.assert_allclose(forward(x.T, nodes, ctx, p), expected, rtol=1e-5, atol=1e-6)

    @unittest.skipIf(torch is None, 'optional neural research dependencies absent')
    def test_set_pool_ignores_player_order_and_missing_values(self):
        from set_temporal_team_model import SetTemporalTeamNet
        torch.set_num_threads(1)
        torch.manual_seed(3)
        net = SetTemporalTeamNet(width=8, layers=2).eval()
        x, nodes, ctx = torch.randn(1, 57, 40), torch.randn(1, 40, 2, 11, 8), torch.rand(1, 40, 8)
        nodes[..., 6] = 1
        with torch.inference_mode():
            expected = net(x, nodes, ctx)
            actual = net(x, nodes[:, :, :, torch.randperm(11)], ctx)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            nodes[..., 6] = 0
            self.assertTrue(torch.isfinite(net(x, nodes, ctx)).all())


if __name__ == '__main__':
    unittest.main()
