# Upstream provenance

This package incorporates and adapts source code and architecture from
[`nv-tlabs/Difix3D`](https://github.com/nv-tlabs/Difix3D) at commit
[`c76edc595586e16732c91ddee82f3a6d83a8a9cc`](https://github.com/nv-tlabs/Difix3D/commit/c76edc595586e16732c91ddee82f3a6d83a8a9cc).
The adapted parts are the two-view UNet, fixed-timestep SD-Turbo restoration
path, VAE encoder/decoder skip path, LoRA setup, and VGG Gram training loss.

Local changes are substantial:

- DACG supplies the restored main view while the original degraded image is
  the reference view.
- Only the main-view VAE encoder skips are retained and decoded; reference-view
  skips are never added to the decoder.
- The frozen DACG DAM descriptor `P_global` conditions both UNet and VAE-decoder
  LoRA matrices through separate conditioners. The default mode supplies an
  explicit stage-ID embedding for each component (10 UNet stages and 6 VAE
  decoder stages).
- CDD-11 preparation, Accelerate training/resume, compact adapter checkpoints,
  native-resolution inference/evaluation, and grouped reporting are local.

The VGG Gram loss keeps Difix3D's released five layer weights and per-layer
`C*H*W` normalization. For reproducibility, the local trainer uses an aligned
center crop instead of the released random crop. Gram matrices are also computed
per image, avoiding the upstream implementation's cross-image coupling when a
training batch contains more than one sample (the two formulations coincide for
the upstream batch size of one).

The degradation-conditioned LoRA formulation follows the method described by
[`ArcticHare105/S3Diff`](https://github.com/ArcticHare105/S3Diff) at commit
[`6a2e0a47676f2f01fccdfd4e940077d3038d953e`](https://github.com/ArcticHare105/S3Diff/commit/6a2e0a47676f2f01fccdfd4e940077d3038d953e).
The implementation here is local and uses DACG's 96-dimensional DAM descriptor,
rather than S3Diff's degradation estimator.

The complete license distributed by Difix3D is retained verbatim in
`LICENSE_DIFIX3D.txt`. Difix3D and derivative code are limited to non-commercial
research or evaluation use under its NVIDIA license. SD-Turbo is additionally
subject to the bundled Stability AI Community License terms.
