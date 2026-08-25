# CDD-11 selective synthesis

This tool generates family-wise deterministic counterfactual states for the
four OneRestore/CDD-11 degradations. It keeps the released operator behavior,
but samples each family's latent variables once and reuses them for every
state.

The original upstream behavior is documented in the official OneRestore
[`syn_data.py`](https://github.com/gy65896/OneRestore/blob/main/syn_data/syn_data.py).
The local implementation deliberately keeps the official low-light noise
interpretation: the sampled value in `[0.03, 0.08]` is passed as NumPy
`normal(..., scale=...)`, so it is a standard deviation despite the paper's
variance wording.

## Input layout

```text
source/
  clear/       # one image per scene stem
  light_map/   # matching scene stems
  depth_map/   # matching scene stems
  rain_mask/   # arbitrary mask-pool stems
  snow_mask/   # arbitrary mask-pool stems
```

For the default configuration, mask pools are expected under
`rain_mask/{train,val,test}` and `snow_mask/{train,val,test}`. The smoke
configuration intentionally uses a shared pool.

## Commands

```bash
python -m venv .venv_cdd11_synthesis
.\.venv_cdd11_synthesis\Scripts\python.exe -m pip install -r requirements_synthesis.txt

\.venv_cdd11_synthesis\Scripts\python.exe tools\synthesize_cdd11_selective.py \
  --config configs/synthesize_cdd11_selective.yaml \
  --dry-run
```

The smoke configuration is intended for one local OneRestore example:

```bash
\.venv_cdd11_synthesis\Scripts\python.exe tools\synthesize_cdd11_selective.py \
  --config configs/synthesize_cdd11_selective_smoke.yaml \
  --split train \
  --max-scenes 1
```

Each family contains 12 PNG states, one `meta.json`, and contributes exactly
20 selective-removal records. Generated images and reports are local runtime
artifacts and are not source-controlled.
