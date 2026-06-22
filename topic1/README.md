# CS60003 HW3 Topic 1: 3DGS 与 AIGC 多源资产融合

本仓库完成题目一的工程化流程：物体 A 多视角重建、物体 B 文本到 3D、物体 C 单图到 3D、Mip-NeRF 360 背景 3DGS，以及统一表示后的场景融合渲染。

项目代码：https://github.com/Zhan-xing-pz/CS60003-HW3/tree/main/topic1

模型与生成资产：https://huggingface.co/zhanxing/CS60003-HW3/tree/main/topic1

当前本机已完成的阶段：

- 从 `物体A.mp4` 抽取 72 张关键帧到 `data/object_a/images/`。
- 对 `物体C.png` 生成前景遮罩和 RGBA 输入到 `data/object_c/`。
- 使用 COLMAP CUDA 版完成物体 A 单相机 SfM，最大模型注册 49 张图、2086 个稀疏点、平均重投影误差 0.6865 px。
- 官方 3DGS 已在本机 `gaussian_splatting_py310` 环境中完成 CUDA 扩展编译、100 iteration 短训、render 和 metrics smoke test。
- 物体 A 已完成 7000 iteration 正式 3DGS 训练，测试指标为 SSIM 0.9222、PSNR 24.2401、LPIPS 0.1431。
- 背景 Flowers 已完成 7000 iteration 3DGS 训练，测试指标为 SSIM 0.5455、PSNR 20.8076、LPIPS 0.4065。
- 物体 B 已完成 DreamFusion / threestudio 1200 step 文本到 3D 训练，并通过 checkpoint 直采样密度场导出 OBJ mesh、点云、turntable 视频和 checkpoint 权重包。
- 物体 C 已完成 Stable Zero123 / threestudio 600 step 单图到 3D 训练，并导出 OBJ mesh、turntable 视频和 checkpoint 权重包。
- 生成统一点云融合原型：`outputs/videos/fusion_preview.mp4` 与 `outputs/assets/fused_preview.ply`，其中物体 B 使用 DreamFusion 导出 mesh 采样点云，物体 C 使用 Zero123 mesh 采样点云。
- 生成报告图表到 `outputs/figures/`。

## 环境

### 系统工具

已验证：

```powershell
ffmpeg -version
C:\Tools\COLMAP\COLMAP.bat -h
blender --version
conda --version
```

本机路径：

- COLMAP: `C:\Tools\COLMAP\COLMAP.bat`
- Blender: `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`
- CUDA Toolkit: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
- GPU: NVIDIA GeForce RTX 5070 Ti

### Python 预览环境

当前预览脚本使用 Windows Python 3.12：

```powershell
py -3.12 -m pip install -r requirements.txt
```

也可以创建独立环境：

```powershell
conda env create -f environment.preview.yml
conda activate hw3-topic1-preview
```

### 3DGS 重建环境

本机已验证的组合：

- Conda 环境: `gaussian_splatting_py310`
- Python: 3.10.20
- PyTorch: `2.7.1+cu128`
- CUDA Toolkit: 12.8
- GPU: NVIDIA GeForce RTX 5070 Ti，compute capability 12.0

可复现安装：

```powershell
conda env create -f environment.reconstruction.yml
conda activate gaussian_splatting_py310
powershell -ExecutionPolicy Bypass -File scripts\setup_external_repos.ps1
scripts\build_3dgs_extensions.bat
```

验证扩展导入：

```powershell
conda run -n gaussian_splatting_py310 python -c "import diff_gaussian_rasterization; import simple_knn._C; import fused_ssim; print('3DGS extensions OK')"
```

## 数据准备

原始素材放在仓库根目录：

```text
物体A.mp4
物体C.png
```

抽取物体 A 帧：

```powershell
py -3.12 scripts\prepare_object_a.py --count 72
```

处理物体 C 单图：

```powershell
py -3.12 scripts\prepare_object_c.py --method grabcut
```

如需使用 `rembg` 大模型抠图：

```powershell
py -3.12 scripts\prepare_object_c.py --method rembg
```

## 物体 A: COLMAP + 3DGS

运行 COLMAP：

```powershell
py -3.12 scripts\run_colmap_pipeline.py --workspace outputs\colmap\object_a_singlecam
```

注意：COLMAP 写 SQLite 数据库时在 Codex 沙箱内可能报 `disk I/O error`，需要在普通 PowerShell 或提权/非沙箱环境中运行。

当前可用的 3DGS 输入目录：

```text
outputs/colmap/object_a_singlecam/dense
```

下载官方 3DGS 和 threestudio：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_external_repos.ps1
```

使用已验证的 3DGS 环境训练 A：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_object_a_3dgs.ps1 -Iterations 7000 -Resolution 2
```

当前正式训练输出：

```text
outputs/gaussian_splatting/object_a
outputs/gaussian_splatting/object_a/point_cloud/iteration_7000/point_cloud.ply
outputs/videos/object_a_3dgs_render.mp4
outputs/figures/object_a_3dgs_eval_grid.jpg
outputs/figures/object_a_3dgs_training_metrics.png
outputs/model_weights/object_a_3dgs_7000.zip
```

正式训练指标：

| Method | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|
| `ours_7000` | 0.9222 | 24.2401 | 0.1431 |

注意：官方 `environment.yml` 使用 Python 3.7、PyTorch 1.12 和 CUDA 11.6，不适合本机 RTX 5070 Ti。实测 PyTorch `2.11.0+cu128` 在 Windows 编译官方 3DGS 扩展时会遇到头文件兼容问题；当前已验证可用版本为 PyTorch `2.7.1+cu128`。

已完成的 smoke test：

```powershell
cd third_party\gaussian-splatting
conda run -n gaussian_splatting_py310 python train.py -s ..\..\outputs\colmap\object_a_singlecam\dense -m ..\..\outputs\gaussian_splatting\object_a_smoke2 --iterations 100 --test_iterations 100 --save_iterations 100 --eval -r 4 --data_device cpu
conda run -n gaussian_splatting_py310 python render.py -m ..\..\outputs\gaussian_splatting\object_a_smoke2
conda run -n gaussian_splatting_py310 python metrics.py -m ..\..\outputs\gaussian_splatting\object_a_smoke2
```

smoke test 指标：SSIM 0.7295，PSNR 18.6779，LPIPS 0.4972。输出模型位于 `outputs/gaussian_splatting/object_a_smoke2/`。

## 背景场景: Mip-NeRF 360 Flowers

从 Mip-NeRF 360 页面下载官方 `360_extra_scenes.zip`，解压 `flowers` 场景到：

```text
data/mipnerf360/flowers
```

训练背景 3DGS：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_background_3dgs.ps1 -Iterations 7000 -Resolution 4
```

当前背景训练输出：

```text
outputs/gaussian_splatting/background_flowers
outputs/gaussian_splatting/background_flowers/point_cloud/iteration_7000/point_cloud.ply
outputs/videos/background_flowers_3dgs_render.mp4
outputs/figures/background_flowers_3dgs_eval_grid.jpg
outputs/figures/background_flowers_3dgs_training_metrics.png
outputs/model_weights/background_flowers_3dgs_7000.zip
```

背景训练指标：

| Method | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|
| `ours_7000` | 0.5455 | 20.8076 | 0.4065 |

## 物体 B: 文本到 3D

文本 Prompt：

```text
a small jade dragon statue with translucent green material, intricate carved scales, and a polished museum-object look
```

训练：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_threestudio_object_b_dreamfusion_sd.ps1 -MaxSteps 1200 -ValInterval 300 -CheckpointInterval 300
```

当前已完成的产物：

```text
outputs/checkpoints/object_b/last.ckpt
outputs/assets/object_b/object_b_dragon_1200.obj
outputs/assets/object_b/object_b_dreamfusion_1200_sampled.ply
outputs/model_weights/object_b_dreamfusion_1200.zip
outputs/videos/object_b_dreamfusion_1200_turntable.mp4
outputs/figures/object_b_dreamfusion_1200_turntable_frame.jpg
```

说明：threestudio 原生 `--export` 在本机 Windows 环境中会卡在隐式场转 OBJ 的 marching cubes 导出阶段。为避免重复卡住，本仓库提供 checkpoint 直导脚本，只读取 `geometry.*` 权重并采样密度场：

```powershell
conda run -n threestudio_py310 python scripts\export_object_b_from_checkpoint.py `
  --checkpoint outputs\checkpoints\object_b\last.ckpt `
  --obj outputs\assets\object_b\object_b_dragon_1200.obj `
  --ply outputs\assets\object_b\object_b_dreamfusion_1200_sampled.ply `
  --resolution 56 --sample-count 50000
```

## 物体 C: 单图到 3D / Zero123

训练：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_threestudio_object_c_zero123.ps1
```

当前已完成的产物：

```text
outputs/assets/object_c/object_c_freq_600.obj
outputs/assets/object_c/object_c_zero123_sampled.ply
outputs/model_weights/object_c_zero123_freq_600.zip
outputs/videos/object_c_zero123_turntable.mp4
outputs/videos/object_c_zero123_val.mp4
outputs/figures/object_c_zero123_turntable_frame.jpg
outputs/figures/object_c_zero123_training_grid.png
```

导出 mesh 后采样成融合点云：

```powershell
py -3.12 scripts\convert_mesh_to_pointcloud.py `
  --mesh outputs\assets\object_c\object_c_freq_600.obj `
  --output outputs\assets\object_c\object_c_zero123_sampled.ply `
  --points 50000 --scale 1.0 --yaw -26 --translate 1.3 0 0
```

## 本地融合预览

生成快速预览：

```powershell
conda run -n gaussian_splatting_py310 python scripts\generate_fusion_preview.py --quick
```

生成完整 6 秒预览：

```powershell
conda run -n gaussian_splatting_py310 python scripts\generate_fusion_preview.py
```

输出：

```text
outputs/videos/fusion_preview.mp4
outputs/preview/fusion_preview_first_frame.jpg
outputs/assets/fused_preview.ply
```

## 图表与报告

生成图表：

```powershell
py -3.12 scripts\make_figures.py
conda run -n gaussian_splatting_py310 python scripts\make_object_a_results.py
conda run -n gaussian_splatting_py310 python scripts\make_3dgs_results.py --model outputs\gaussian_splatting\background_flowers --stdout-log outputs\logs\background_flowers_3dgs_stdout.log --stderr-log outputs\logs\background_flowers_3dgs_stderr.log --prefix background_flowers_3dgs --title "Background Flowers 3DGS" --note "Mip-NeRF 360 Flowers" --note "Registered views: 173" --note "Initial sparse points: 38347" --note "Iterations: 7000"
```

报告源文件：

```text
reports/topic1_report.md
```

模型权重不进入 Git 历史，统一上传至 Hugging Face 的 `topic1/weights/`。导出的 OBJ/PLY、融合预览视频和关键预览图分别位于 `topic1/assets/` 与 `topic1/previews/`。每个权重包的训练设置、文件内容和 SHA256 校验值记录在模型仓库的 manifest 中。

## 参考实现

- 3D Gaussian Splatting 官方仓库: https://github.com/graphdeco-inria/gaussian-splatting
- threestudio 官方仓库: https://github.com/threestudio-project/threestudio
- COLMAP: https://github.com/colmap/colmap
- Mip-NeRF 360: https://jonbarron.info/mipnerf360/
