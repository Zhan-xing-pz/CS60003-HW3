# Topic 1 Model Artifacts

The Topic 1 artifacts are hosted in the Hugging Face model repository:

https://huggingface.co/zhanxing/CS60003-HW3/tree/main/topic1

## Weights

| Artifact | Training setting | Main contents |
|---|---|---|
| `background_flowers_3dgs_7000.zip` | Mip-NeRF 360 Flowers, 3DGS, 7000 iterations | Gaussian point cloud, camera metadata, evaluation JSON |
| `object_a_3dgs_7000.zip` | Captured object A, COLMAP + 3DGS, 7000 iterations | Gaussian point cloud, camera metadata, evaluation JSON |
| `object_b_dreamfusion_1200.zip` | Text-to-3D DreamFusion, 1200 steps | checkpoint and resolved threestudio configs |
| `object_c_zero123_freq_600.zip` | Stable Zero123, 600 steps | checkpoint and resolved threestudio configs |

## Validation Metrics

- Object A 3DGS: SSIM `0.9222`, PSNR `24.2401`, LPIPS `0.1431`.
- Flowers background 3DGS: SSIM `0.5455`, PSNR `20.8076`, LPIPS `0.4065`.
- Object B exported mesh: 1024 vertices and 2044 faces.
- Object C exported mesh: 11848 vertices and 23692 faces.

The repository also contains exported meshes, sampled point clouds, preview images, and the final fusion preview video under the `topic1/` directory.
