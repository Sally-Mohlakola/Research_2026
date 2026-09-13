import unittest
import numpy as np
import torch
from neural.boundary_model import BoundaryModel


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2)
        self.model = BoundaryModel(np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32),
                                   np.array([[0,1,2]]), width=16, components=2)
        self.x = torch.zeros(128, 10)

    def test_surface_and_hemisphere(self):
        s = self.model.sample(self.x)
        self.assertTrue((s['barycentric'] >= 0).all())
        torch.testing.assert_close(s['barycentric'].sum(-1), torch.ones(128))
        torch.testing.assert_close(s['exit_position'][:,2], torch.zeros(128))
        torch.testing.assert_close(s['exit_direction'].norm(dim=-1), torch.ones(128))
        self.assertTrue((s['exit_direction'][:,2] > 0).all())

    def test_seeded_sampling(self):
        a = self.model.sample(self.x, torch.Generator().manual_seed(11))
        b = self.model.sample(self.x, torch.Generator().manual_seed(11))
        for key in a:
            torch.testing.assert_close(a[key], b[key])

    def test_truncated_records_have_no_exit_loss(self):
        losses = self.model.losses(self.x, torch.zeros(128, dtype=torch.long),
                                   torch.full((128,4), float('nan')), torch.zeros(128, dtype=torch.bool))
        self.assertTrue(torch.isfinite(losses[0]))
        self.assertEqual(float(losses[1].detach()), 0.)
        self.assertEqual(float(losses[2].detach()), 0.)

    def test_multimodal_learning(self):
        # Two distinct outcomes for identical entry conditions.
        y = torch.zeros(128, 4)
        y[:64, 0], y[64:, 0] = -3., 3.
        f, e = torch.zeros(128, dtype=torch.long), torch.ones(128, dtype=torch.bool)
        opt = torch.optim.Adam(self.model.parameters(), lr=.01)
        initial = float(self.model.losses(self.x, f, y, e)[2].detach())
        for _ in range(100):
            opt.zero_grad()
            loss = sum(self.model.losses(self.x, f, y, e))
            loss.backward()
            opt.step()
        final = float(self.model.losses(self.x, f, y, e)[2].detach())
        self.assertLess(final, initial-1)
        h = self.model.encoder(self.x[:1])
        logits, means, _ = self.model.distribution(h, f[:1])
        self.assertGreater(float((means[0,:,0].max()-means[0,:,0].min()).detach()), 2.)


if __name__ == '__main__':
    unittest.main()
