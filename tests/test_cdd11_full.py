import unittest
from pathlib import Path

from cdd11_full.protocol import DEGRADATIONS, OBJECTIVE_VARIANT, PROTOCOL_NAME, RUN_PROFILES


class TestProtocol(unittest.TestCase):
    def test_frozen_comparison_protocol_and_declared_loss_exception(self):
        self.assertEqual(PROTOCOL_NAME, "cdd11-v1")
        self.assertEqual(OBJECTIVE_VARIANT, "dacg-paper-rgb-fourier-l1")
        self.assertEqual(len(DEGRADATIONS), 11)
        self.assertEqual(RUN_PROFILES["smoke"]["max_steps"], 100)
        self.assertEqual(RUN_PROFILES["pilot"]["max_steps"], 5000)
        self.assertEqual(RUN_PROFILES["formal"]["max_steps"], 200000)
        self.assertEqual(RUN_PROFILES["formal"]["validation_interval"], 5000)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable in local static-check environment")
class TestTrainingCore(unittest.TestCase):
    def test_balanced_sampler_is_step_deterministic(self):
        from cdd11_full.data import BalancedDegradationBatchSampler, ManifestRecord

        class Dataset:
            split = "train"
            records = [ManifestRecord(f"{degradation}-{scene}", degradation, "train",
                       Path("/input"), Path("/target"), str(scene), degradation.count("_") + 1)
                       for degradation in DEGRADATIONS for scene in range(3)]

        first = list(BalancedDegradationBatchSampler(Dataset(), 20, 4, 3407))
        second = list(BalancedDegradationBatchSampler(Dataset(), 20, 4, 3407))
        resumed = list(BalancedDegradationBatchSampler(Dataset(), 22, 2, 3407))
        self.assertEqual(first, second)
        self.assertEqual(first[2:], resumed)
        for batch in first:
            categories = {Dataset.records[index].degradation for index, _ in batch}
            self.assertEqual(categories, set(DEGRADATIONS))

    def test_paper_loss_microbatch_equivalence(self):
        from cdd11_full.model import OriginalDACGLoss

        generator = torch.Generator().manual_seed(3407)
        prediction = torch.rand((11, 3, 16, 16), generator=generator, requires_grad=True)
        target = torch.rand((11, 3, 16, 16), generator=generator)
        loss = OriginalDACGLoss()
        full, _, _ = loss.per_sample(prediction, target)
        micro = []
        for start in range(0, 11, 3):
            values, _, _ = loss.per_sample(prediction[start:start + 3], target[start:start + 3])
            micro.extend(values)
        self.assertTrue(torch.allclose(full, torch.stack(micro), atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(full.mean(), torch.stack(micro).sum() / 11, atol=1e-7, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
