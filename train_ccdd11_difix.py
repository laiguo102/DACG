"""Launch selective Difix training with CCDD-11 native selective supervision."""

from difix3d_selective.train import main


if __name__ == "__main__":
    main(default_dataset_format="ccdd11")
