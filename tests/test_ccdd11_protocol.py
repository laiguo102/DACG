from pathlib import Path

from difix3d_selective.ccdd_prepare import native_target_path
from difix3d_selective.protocol import directed_tasks, expected_counts


def test_ccdd_target_direction_uses_preserved_component_suffix():
    forward, reverse = directed_tasks(1)
    sub_root = Path("CCDD-11") / "half_train" / "sub_data"

    assert forward.remove == "low"
    assert forward.preserve == "haze"
    assert native_target_path(sub_root, "low_haze", "00001", forward.preserve) == (
        sub_root / "low_haze" / "00001" / "00001_haze_.png"
    )
    assert reverse.remove == "haze"
    assert reverse.preserve == "low"
    assert native_target_path(sub_root, "low_haze", "00001", reverse.preserve) == (
        sub_root / "low_haze" / "00001" / "00001_low_.png"
    )


def test_ccdd_all_five_expected_counts():
    assert expected_counts([1, 2, 3, 4, 5]) == {
        "coarse": 5915,
        "train": 10650,
        "validation": 1180,
    }
