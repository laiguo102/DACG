import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from difix3d_selective.pixel_cfg_compare import run


class TestPixelCfgComparison(unittest.TestCase):
    def test_reuses_saved_cfg_endpoints_and_writes_comparison(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            cfg_dir = Path(directory) / "cfg"
            sample = cfg_dir / "gallery" / "sample_a"
            sample.mkdir(parents=True)
            (cfg_dir / "state.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "config": {
                            "protocol": (
                                "ccdd11-endpoint-correct-state-cfg-validation-v2"
                            ),
                            "betas": [0, 0.5, 1],
                        },
                    }
                ),
                encoding="utf-8",
            )
            for label, value in (("0", 51), ("0.5", 153), ("1", 204)):
                Image.new("RGB", (5, 4), (value, value, value)).save(
                    sample / f"beta_{label}.png"
                )
            output = Path(directory) / "comparison"
            result = run(
                argparse.Namespace(
                    cfg_dir=cfg_dir,
                    output_dir=output,
                    betas=None,
                    sample_ids=None,
                    max_samples=None,
                )
            )
            self.assertEqual(result, output.resolve())
            pixel = np.asarray(Image.open(output / "sample_a" / "pixel_beta_0.5.png"))
            self.assertTrue(np.all(pixel == 128))
            report = json.loads(
                (output / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["row_count"], 3)
            with Image.open(output / "sample_a" / "comparison_sheet.png") as sheet:
                self.assertEqual(sheet.size, (15, 84))


if __name__ == "__main__":
    unittest.main()
