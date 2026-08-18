import unittest
from pathlib import Path
from unittest.mock import patch

from cdd11_full.data import validate_partition
from cdd11_full.protocol import DEGRADATIONS, PROTOCOL_NAME, protocol_metadata, sha256_payload


class TestCDD11FullProtocol(unittest.TestCase):
    def test_protocol_uses_every_official_train_scene_without_holdout(self):
        metadata = protocol_metadata()
        self.assertEqual(PROTOCOL_NAME, "cdd11-dacg-full-v1")
        self.assertEqual(metadata["training_scenes"], 1183)
        self.assertEqual(metadata["training_pairs"], 1183 * 11)
        self.assertIsNone(metadata["validation_partition"])
        self.assertEqual(metadata["checkpoint_selection"], "final_completed_training_checkpoint")
        self.assertFalse(metadata["oof"])
        self.assertEqual(len(DEGRADATIONS), 11)
        self.assertEqual(sha256_payload(metadata), sha256_payload(dict(reversed(list(metadata.items())))))

    def test_partition_pairing(self):
        paired = {"00001": Path("00001.png"), "00002": Path("00002.png")}
        with patch("cdd11_full.data.index_images", return_value=paired):
            self.assertEqual(validate_partition(Path("CDD11"), "train", expected_scenes=2), ["00001", "00002"])
        incomplete = dict(paired)
        incomplete.pop("00002")
        calls = [paired] + [incomplete] + [paired] * (len(DEGRADATIONS) - 1)
        with patch("cdd11_full.data.index_images", side_effect=calls):
            with self.assertRaises(ValueError):
                validate_partition(Path("CDD11"), "train", expected_scenes=2)


try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the local static-check environment")
class TestDeterministicSampler(unittest.TestCase):
    def test_resume_matches_uninterrupted_requests(self):
        from cdd11_full.data import DeterministicStepBatchSampler

        kwargs = dict(dataset_size=22, batch_size=2, accumulate=2, epochs=3, seed=3407)
        entire = list(DeterministicStepBatchSampler(**kwargs))
        resumed = list(DeterministicStepBatchSampler(**kwargs, start_step=3))
        self.assertEqual(resumed, entire[3 * kwargs["accumulate"]:])
        sampler = DeterministicStepBatchSampler(**kwargs)
        per_epoch = sampler.steps_per_epoch * kwargs["accumulate"]
        for epoch in range(kwargs["epochs"]):
            requests = entire[epoch * per_epoch:(epoch + 1) * per_epoch]
            indices = [index for batch in requests for index, _ in batch]
            self.assertEqual(len(indices), len(set(indices)))


if __name__ == "__main__":
    unittest.main()
