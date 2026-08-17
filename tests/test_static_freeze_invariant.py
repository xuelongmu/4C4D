"""Regression tests for the static-temporal freeze invariant.

Background: `--freeze_static_temporal` originally "froze" gaussians by zeroing
their gradients before `optimizer.step()`. That does not hold a parameter
still under Adam, which keeps stepping from its stored first/second moments
until they decay. The bug was invisible to the smoke test (training ran fine)
and to the metrics (the runs produced plausible numbers), and it invalidated
three full A/B runs before code review caught it.

These tests encode the invariant itself, so a future refactor of the freeze
cannot silently regress it:

* zeroing gradients is NOT sufficient — asserted, so nobody reintroduces it;
* snapshot-then-restore IS exact, regardless of optimizer internals;
* the mask must not be reused across a change in row identity.

Run: python -m unittest discover -s tests -v
"""
import unittest

import torch


def _adam_with_momentum(param, steps=5, lr=1e-2):
    """Give the optimizer non-trivial moment state, as mid-training would."""
    opt = torch.optim.Adam([param], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        param.grad = torch.ones_like(param)
        opt.step()
    return opt


class StaticFreezeInvariantTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n = 64
        self.mask = torch.zeros(self.n, dtype=torch.bool)
        self.mask[: self.n // 2] = True  # first half "static"

    def test_zeroing_gradients_does_not_freeze_under_adam(self):
        """The original implementation's assumption, asserted false.

        If this ever starts passing, Adam's semantics changed and the
        snapshot/restore in train.py could be simplified — but until then,
        zeroing gradients is not a freeze.
        """
        param = torch.nn.Parameter(torch.randn(self.n, 1))
        opt = _adam_with_momentum(param)
        before = param.data[self.mask].clone()

        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            param.grad = torch.ones_like(param)
            param.grad[self.mask] = 0.0  # the old "freeze"
            opt.step()

        drift = (param.data[self.mask] - before).abs().max().item()
        self.assertGreater(
            drift, 1e-4,
            "zeroing gradients unexpectedly held the rows still; if Adam's "
            "behaviour changed, revisit the freeze implementation")

    def test_snapshot_restore_holds_rows_exactly(self):
        """The current implementation's contract: frozen rows do not move."""
        param = torch.nn.Parameter(torch.randn(self.n, 1))
        opt = _adam_with_momentum(param)
        before = param.data[self.mask].clone()

        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            param.grad = torch.ones_like(param)
            param.grad[self.mask] = 0.0
            saved = param.data[self.mask].clone()   # snapshot, as train.py does
            opt.step()
            with torch.no_grad():
                param.data[self.mask] = saved       # restore, as train.py does

        self.assertTrue(
            torch.equal(param.data[self.mask], before),
            "frozen rows moved; the freeze must be exact, not approximate")

    def test_unfrozen_rows_still_train(self):
        """A freeze that also stops the free rows would be worse than useless."""
        param = torch.nn.Parameter(torch.randn(self.n, 1))
        opt = _adam_with_momentum(param)
        before_free = param.data[~self.mask].clone()

        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            param.grad = torch.ones_like(param)
            param.grad[self.mask] = 0.0
            saved = param.data[self.mask].clone()
            opt.step()
            with torch.no_grad():
                param.data[self.mask] = saved

        moved = (param.data[~self.mask] - before_free).abs().max().item()
        self.assertGreater(moved, 1e-3, "non-static rows stopped training")

    def test_mask_length_does_not_imply_row_identity(self):
        """Why the mask is invalidated on densify/prune rather than length-checked.

        Densification appends rows and pruning compacts them. A cloned-then-
        pruned population can return to its original length while every row
        maps to a different gaussian, so a length check cannot detect staleness
        and the stale mask would pin unrelated gaussians.
        """
        ids = torch.arange(self.n)
        mask = torch.zeros(self.n, dtype=torch.bool)
        mask[:8] = True

        grown = torch.cat([ids, ids[:8]])          # clone 8 rows
        keep = torch.ones(grown.shape[0], dtype=torch.bool)
        keep[:8] = False                            # prune 8 different rows
        compacted = grown[keep]

        self.assertEqual(compacted.shape[0], self.n,
                         "test setup should return to the original length")
        self.assertFalse(torch.equal(compacted, ids),
                         "row identities must differ despite equal length")
        # Applying the stale mask would now select gaussians it never classified.
        self.assertFalse(torch.equal(compacted[mask], ids[mask]))


if __name__ == "__main__":
    unittest.main()
