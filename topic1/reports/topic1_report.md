# 题目一：基于 3DGS 与 AIGC 的多源资产生成与真实场景融合

> GitHub 仓库：https://github.com/Zhan-xing-pz/CS60003-HW3  
> 模型权重：https://huggingface.co/zhanxing/CS60003-HW3/tree/main/topic1  
> 姓名与分工：彭卓（25210980159），完成数据准备、真实场景重建、AIGC 资产生成、融合渲染、实验分析与报告。

## 摘要

本实验围绕真实重建、文本生成与单图生成三种 3D 资产来源，构建一个多源资产融合流程。物体 A 由手机环绕视频经 COLMAP 提取相机位姿，并作为 3D Gaussian Splatting 的输入；物体 B 使用 threestudio 的文本到 3D 生成流程；物体 C 由单张真实照片经前景分割后输入 Stable Zero123；背景采用 Mip-NeRF 360 Flowers 场景重建。为统一表示形式，本文采用“高斯/点云显式表达”为融合中间层：真实 3DGS 结果保留高斯表示，AIGC mesh 资产通过表面采样转换为带颜色点云，进一步可初始化为高斯或在 Blender/自定义 renderer 中与背景组合。

## 数据与资产

物体 A 为手机拍摄的青绿色杯子环绕视频，原始视频共 1370 帧，30 FPS，分辨率 1280 x 720。实验抽取 72 张关键帧并缩放到最大宽度 1024，用于 COLMAP 特征提取与匹配；其中 49 张视角成功注册并进入 3DGS 训练。

物体 C 为同一杯子的单张照片，分辨率 1448 x 1086。实验使用 GrabCut 生成前景 alpha，并输出 `data/object_c/object_c_rgba.png`，作为 Stable Zero123 单图到 3D 的条件输入。当前已完成 600 step 训练、60 视角 turntable 渲染、OBJ mesh 导出和 50000 点表面采样。

物体 B 的文本 prompt 设定为：`a small jade dragon statue with translucent green material, intricate carved scales, and a polished museum-object look`。选择该 prompt 的原因是其几何形状与真实杯子明显不同，方便观察文本生成资产和真实重建资产在融合场景中的差异。

背景选用 Mip-NeRF 360 Flowers 场景。当前本地预览阶段使用程序化点云代替真实背景，用于验证融合坐标、尺度与渲染路径；背景正式训练已完成，输出可用于后续真实融合渲染。

![资产概览](../outputs/figures/asset_overview.jpg)

## 方法

### 物体 A：真实多视角重建

首先从视频中均匀抽帧，并略去首尾容易出现手持抖动的片段。随后使用 COLMAP CUDA 版本执行 SIFT 特征提取、穷举匹配、增量 SfM 和图像去畸变。由于所有帧来自同一手机视频，COLMAP 使用 `ImageReader.single_camera=1` 共享相机内参，避免每帧单独估计内参导致的漂移。

当前 COLMAP 最大模型统计如下：

| 指标 | 数值 |
|---|---:|
| 注册图像数 | 49 |
| 稀疏点数 | 2086 |
| 观测数 | 11424 |
| 平均 track length | 5.4765 |
| 平均每图观测数 | 233.1429 |
| 平均重投影误差 | 0.6865 px |

![COLMAP 质量统计](../outputs/figures/object_a_colmap_quality.png)

该结果已经生成 3DGS 可读取的 COLMAP 格式目录：`outputs/colmap/object_a_singlecam/dense`。随后在官方 3DGS 代码上训练 7000 iteration，使用 `--eval` 保留 7 个测试视角，图像数据放在 CPU 侧以降低显存占用。训练输出位于 `outputs/gaussian_splatting/object_a/`，最终高斯模型文件为 `point_cloud/iteration_7000/point_cloud.ply`，本地权重包为 `outputs/model_weights/object_a_3dgs_7000.zip`。

物体 A 的 3DGS 定量指标如下：

| 指标 | 数值 |
|---|---:|
| Test SSIM | 0.9222 |
| Test PSNR | 24.2401 |
| Test LPIPS | 0.1431 |
| Train PSNR @ 7000 | 34.4470 |
| Test PSNR @ 7000 | 24.5058 |

![物体 A 3DGS 训练与指标](../outputs/figures/object_a_3dgs_training_metrics.png)

下图给出保留测试视角的 GT、3DGS render 与差异图。整体几何轮廓、桌面透视和杯柄位置较稳定，误差主要集中在杯口内侧、高光反射和局部阴影边缘，这与低纹理陶瓷表面和反光材质较难由多视角照片完全约束有关。

![物体 A 3DGS 测试视角对比](../outputs/figures/object_a_3dgs_eval_grid.jpg)

### 物体 B：文本到 3D

物体 B 使用 threestudio 的 SDS 优化流程，由文本 prompt 通过 Stable Diffusion v1.5 提供多视角监督。由于本机 RTX 5070 Ti 对 `tiny-cuda-nn` HashGrid backward 兼容性不稳定，B 与 C 一样采用纯 PyTorch 的 `ProgressiveBandFrequency` 几何编码。训练完成 1200 step 后得到 `last.ckpt` 和 120 帧 turntable 渲染；threestudio 原生 `--export` 在 Windows 上容易卡在隐式场到 OBJ 的 marching cubes 阶段，因此本实验补充实现 `scripts/export_object_b_from_checkpoint.py`，使用 `weights_only=True` 安全读取 checkpoint 中的 `geometry.*` 权重，直接采样密度场并导出显式 mesh。

最终 B 的导出 OBJ 位于 `outputs/assets/object_b/object_b_dragon_1200.obj`，包含 1024 个顶点和 2044 个面；进一步表面采样得到 `outputs/assets/object_b/object_b_dreamfusion_1200_sampled.ply`，包含 50000 个带颜色点。该点云已接入融合 renderer，可作为显式资产，也可进一步初始化为一组 Gaussian primitives。B 的 checkpoint 与配置打包为 `outputs/model_weights/object_b_dreamfusion_1200.zip`，turntable 视频为 `outputs/videos/object_b_dreamfusion_1200_turntable.mp4`。

![物体 B DreamFusion turntable 抽帧](../outputs/figures/object_b_dreamfusion_1200_turntable_frame.jpg)

### 物体 C：单图到 3D

物体 C 首先由单张照片生成 alpha 前景图。随后使用 threestudio Stable Zero123 配置，以输入视角图像为条件优化 NeRF/mesh 表示。由于本机 RTX 5070 Ti 对 `tiny-cuda-nn` HashGrid backward 兼容性不稳定，本实验将几何编码切换为纯 PyTorch 的 `ProgressiveBandFrequency`，保证训练和导出流程可复现。

训练采用 600 step 快速配置，图像分辨率按 128、192、256 逐级提升，随机相机 batch size 按 4、2、1 逐级调整；最终导出 OBJ mesh，包含 11848 个顶点和 23692 个面。核心产物如下：

| 产物 | 路径 |
|---|---|
| Stable Zero123 checkpoint | `outputs/model_weights/object_c_zero123_freq_600.zip` |
| OBJ mesh | `outputs/assets/object_c/object_c_freq_600.obj` |
| Mesh sampled point cloud | `outputs/assets/object_c/object_c_zero123_sampled.ply` |
| 60-view turntable | `outputs/videos/object_c_zero123_turntable.mp4` |
| Validation render | `outputs/videos/object_c_zero123_val.mp4` |

下图为 Object C 导出的 turntable 视频抽帧，包含渲染外观、法向/深度等调试视图。单图生成能够较好保留输入视角的杯体颜色和轮廓，但背面形状仍主要依赖 Zero123 先验。

![物体 C Zero123 turntable 抽帧](../outputs/figures/object_c_zero123_turntable_frame.jpg)

### 背景 3DGS

背景选用 Mip-NeRF 360 Flowers。该数据集来自官方 `360_extra_scenes.zip`，解压后包含 173 张多视角图像、预计算 COLMAP sparse 模型和多级下采样图像，可直接作为 3DGS 训练输入。训练时启用 `--eval` 划分测试视角，输出 PSNR/SSIM/LPIPS 等指标，并导出多视角漫游视频。

背景 3DGS 训练设置为 7000 iteration、`-r 4`，初始稀疏点数为 38347，输出位于 `outputs/gaussian_splatting/background_flowers/`。最终高斯模型文件为 `point_cloud/iteration_7000/point_cloud.ply`，本地权重包为 `outputs/model_weights/background_flowers_3dgs_7000.zip`。

| 指标 | 数值 |
|---|---:|
| Test SSIM | 0.5455 |
| Test PSNR | 20.8076 |
| Test LPIPS | 0.4065 |
| Train PSNR @ 7000 | 22.0529 |
| Test PSNR @ 7000 | 20.8410 |

![背景 3DGS 训练与指标](../outputs/figures/background_flowers_3dgs_training_metrics.png)

![背景 3DGS 测试视角对比](../outputs/figures/background_flowers_3dgs_eval_grid.jpg)

由于 Flowers 是大范围非物体中心场景，且包含大量草地、叶片、花簇等高频重复纹理，7000 iteration 的快速训练结果能恢复整体花坛结构和颜色分布，但草地与叶片仍有明显平滑。最终质量版可进一步提高到 30000 iteration。

### 表示统一与融合

本实验采用显式点/高斯中间层完成融合：

1. A 和背景由 COLMAP 初始化，再经 3DGS 训练得到高斯集合。
2. B 和 C 由 threestudio/Zero123 输出 mesh；当前 B 已导出 `object_b_dragon_1200.obj`，C 已导出 `object_c_freq_600.obj`。
3. mesh 资产通过表面采样得到 `(x, y, z, r, g, b)` 点云，并通过尺度、旋转、平移矩阵放入背景坐标系。
4. 在预览阶段使用点云 splatting renderer 生成漫游视频；最终阶段可将采样点初始化为高斯，或在 Blender 中导入 mesh/point cloud 与背景渲染结果进行组合。

当前本地预览输出为 `outputs/videos/fusion_preview.mp4`，其中 B 使用 DreamFusion checkpoint 导出的 mesh 采样点云，C 使用 Zero123 mesh 采样点云，A 和背景仍为轻量点云预览表达。首帧如下：

![融合预览](../outputs/preview/fusion_preview_first_frame.jpg)

## 结果与分析

三种资产生成方式的相对比较如下图。多视角重建几何一致性最好，但对拍摄质量、纹理丰富度和 COLMAP 匹配质量敏感；文本到 3D 自由度最高，但容易出现过平滑、Janus、多面不一致等 SDS 典型问题；单图到 3D 能保留输入图像纹理和主体类别，但背面几何和纹理主要依赖先验补全。

![方法比较](../outputs/figures/method_comparison.png)

| 路线 | 几何准确度 | 纹理细节 | 耗时与资源 | 主要问题 |
|---|---|---|---|---|
| 多视角 3DGS | 高，受 COLMAP 位姿影响 | 高，真实图像监督 | 中等，需 COLMAP + 3DGS | 低纹理物体和反光表面难匹配 |
| 文本到 3D | 中等，受扩散先验影响 | 中等到高，语义强但细节可能不稳定 | 高，SDS 优化耗时 | 多视角一致性和几何漂移 |
| 单图到 3D | 中等，正面较好背面依赖补全 | 输入视角纹理好 | 高，需 Zero123 优化 | 背面幻觉、尺度不确定 |

## 训练日志

物体 A 与背景 Flowers 已完成真实 3DGS 训练。A 的日志保存在 `outputs/logs/object_a_3dgs_stdout.log` 和 `outputs/logs/object_a_3dgs_stderr.log`，背景日志保存在 `outputs/logs/background_flowers_3dgs_stdout.log` 和 `outputs/logs/background_flowers_3dgs_stderr.log`。物体 B 的 DreamFusion 最终达到 `max_steps=1200`，保留 `last.ckpt`、解析后的配置与 turntable 渲染。物体 C 的 Stable Zero123 日志保存在 `outputs/logs/object_c_zero123_train_stdout.log` 和 `outputs/logs/object_c_zero123_train_stderr.log`，最终训练约 7 分 22 秒，`max_steps=600` 正常结束，末尾训练 loss 约为 9.040。当前报告中的 A/背景曲线由本地训练日志解析生成；B/C 使用导出的 turntable 视频、mesh 和采样点云说明结果。

本作业要求的 WandB 训练与验证曲线统一在题目二 ACT 实验中提供；题目一保留原始训练日志、解析后的 3DGS 曲线、checkpoint、配置、渲染视频和导出资产，确保无需重新训练即可复核结果。

## 超参数

| 模块 | 设置 |
|---|---|
| A 抽帧 | 72 frames, max width 1024 |
| COLMAP camera model | OPENCV |
| COLMAP camera sharing | single camera |
| Feature extractor | SIFT GPU |
| Matcher | exhaustive matcher GPU |
| A 3DGS iterations | 7000 completed; 30000 optional for final quality |
| Background 3DGS iterations | 7000 completed; 30000 optional for final quality |
| B prompt | jade dragon statue prompt |
| B DreamFusion steps | 1200 completed |
| B mesh | 1024 vertices, 2044 faces |
| B sampled point cloud | 50000 points |
| C input | `data/object_c/object_c_rgba.png` |
| C Stable Zero123 steps | 600 completed |
| C mesh | 11848 vertices, 23692 faces |
| C sampled point cloud | 50000 points |
| Fusion preview | 1280 x 720, 30 FPS, 180 frames |

## 结论

当前已完成物体 A 的 COLMAP 位姿估计与 3DGS 训练、物体 B 的 DreamFusion 文本到 3D 训练与 checkpoint mesh 导出、物体 C 的 Stable Zero123 单图到 3D 训练和 mesh 导出，以及 Mip-NeRF 360 Flowers 背景 3DGS 训练。统一表示方案采用 mesh-to-point/Gaussian 的显式融合方式，已在本地预览中将 B 的 DreamFusion 采样点云与 C 的 Zero123 采样点云并入统一点云场景。代码、模型权重和生成资产分别整理到 GitHub 与 Hugging Face，便于复现实验和检查结果。
