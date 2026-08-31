"""Evaluate CCDD-11 selective Difix on half_test with old-style references."""

from difix3d_selective.evaluate import main


if __name__ == "__main__":
    main(default_dataset_format="ccdd11")
