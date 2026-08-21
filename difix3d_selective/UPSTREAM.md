# Upstream provenance

This package incorporates and adapts source code from
[`nv-tlabs/Difix3D`](https://github.com/nv-tlabs/Difix3D) at commit
`c76edc595586e16732c91ddee82f3a6d83a8a9cc`.

Imported components:

- `mv_unet.py`: the two-view UNet implementation.
- `loss.py`: the VGG Gram/style loss.
- the VAE encoder/decoder skip design and training structure used by `model.py`.

Local changes make the degraded reference view conditioning-only: both views enter
the VAE encoder and multi-view UNet, but only view 0's denoised latent and VAE encoder
skip activations are decoded and supervised. CDD11/DACG preparation and directed
`remove A, preserve B` manifests are local additions.

The complete upstream license is retained in `LICENSE_DIFIX3D.txt`. Difix3D and its
derivatives are limited to non-commercial use under that license.
