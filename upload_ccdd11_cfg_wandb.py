"""Upload completed CCDD-11 CFG results to W&B without rerunning inference."""

import argparse
from pathlib import Path

from difix3d_selective.cfg_evaluate import upload_wandb_results


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--results-dir", type=Path, required=True)
    value.add_argument("--wandb-entity", default="c14150591-sjtu")
    value.add_argument("--wandb-project", default="difix-ccdd11-selective")
    value.add_argument(
        "--wandb-run-name",
        default="ccdd11-cfg-beta-axis-task-curves",
    )
    return value


def main() -> None:
    upload_wandb_results(parser().parse_args())


if __name__ == "__main__":
    main()
