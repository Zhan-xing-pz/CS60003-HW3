param(
  [string]$ThreeStudioRepo = "third_party\threestudio",
  [string]$Image = "data\object_c\object_c_rgba.png",
  [string]$EnvName = "threestudio_py310",
  [int]$Gpu = 0,
  [int]$MaxSteps = 600,
  [int]$ValInterval = 100,
  [int]$CheckpointInterval = 100,
  [int]$ValViews = 8,
  [int]$TestViews = 60
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path ".").Path
$threeStudioPath = (Resolve-Path $ThreeStudioRepo).Path
$imagePath = (Resolve-Path $Image).Path
$python = Join-Path $env:USERPROFILE "miniconda3\envs\$EnvName\python.exe"
$vsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

if (-not (Test-Path $python)) {
  throw "Python for env '$EnvName' not found: $python"
}
if (-not (Test-Path $vsDevCmd)) {
  throw "Visual Studio DevCmd not found: $vsDevCmd"
}

$cacheRoot = Join-Path $repoRoot ".cache"
$clipDir = Join-Path $cacheRoot "home\.cache\clip"
$hfDir = Join-Path $cacheRoot "huggingface"
$runTmp = Join-Path $env:TEMP "hw3_threestudio"
$torchExt = Join-Path $env:TEMP "hw3_torch_extensions"

New-Item -ItemType Directory -Force -Path `
  $clipDir, `
  $hfDir, `
  (Join-Path $runTmp "matplotlib"), `
  (Join-Path $runTmp "wandb"), `
  $torchExt | Out-Null

$targetImage = Join-Path $threeStudioPath "load\images\object_c_rgba.png"
Copy-Item -LiteralPath $imagePath -Destination $targetImage -Force

$overrides = @(
  "--config configs/stable-zero123.yaml",
  "--train",
  "--gpu $Gpu",
  "name=zero123-object-c",
  "tag=object_c_freq_${MaxSteps}",
  "data.image_path=./load/images/object_c_rgba.png",
  "trainer.max_steps=$MaxSteps",
  "trainer.val_check_interval=$ValInterval",
  "checkpoint.every_n_train_steps=$CheckpointInterval",
  "data.height=[128,192,256]",
  "data.width=[128,192,256]",
  "data.resolution_milestones=[200,400]",
  "data.random_camera.height=[64,96,128]",
  "data.random_camera.width=[64,96,128]",
  "data.random_camera.batch_size=[4,2,1]",
  "data.random_camera.resolution_milestones=[200,400]",
  "data.random_camera.eval_height=256",
  "data.random_camera.eval_width=256",
  "data.random_camera.n_val_views=$ValViews",
  "data.random_camera.n_test_views=$TestViews",
  "system.renderer.num_samples_per_ray=128",
  "system.loggers.wandb.enable=false",
  "system.optimizer.args.foreach=false",
  "system.geometry.pos_encoding_config.otype=ProgressiveBandFrequency",
  "system.geometry.pos_encoding_config.n_frequencies=12",
  "system.geometry.pos_encoding_config.include_xyz=true",
  "system.geometry.mlp_network_config.n_neurons=128",
  "system.geometry.mlp_network_config.n_hidden_layers=4"
)

$cmd = @(
  "`"$vsDevCmd`" -arch=x64",
  "set `"VSLANG=1033`"",
  "set `"TORCH_EXTENSIONS_DIR=$torchExt`"",
  "set `"TORCH_CUDA_ARCH_LIST=12.0`"",
  "set `"TCNN_CUDA_ARCHITECTURES=120`"",
  "set `"TORCH_DONT_CHECK_COMPILER_ABI=1`"",
  "set `"DISTUTILS_USE_SDK=1`"",
  "set `"MSSdk=1`"",
  "set `"CLIP_CACHE_DIR=$clipDir`"",
  "set `"HF_HOME=$hfDir`"",
  "set `"TRANSFORMERS_CACHE=$hfDir`"",
  "set `"HF_HUB_CACHE=$(Join-Path $hfDir 'hub')`"",
  "set `"MPLCONFIGDIR=$(Join-Path $runTmp 'matplotlib')`"",
  "set `"WANDB_DIR=$(Join-Path $runTmp 'wandb')`"",
  "set `"WANDB_CACHE_DIR=$(Join-Path $runTmp 'wandb')`"",
  "set `"WANDB_MODE=disabled`"",
  "set `"WANDB_SILENT=true`"",
  "set `"PYTHONIOENCODING=utf-8`"",
  "set `"PYTHONUTF8=1`"",
  "cd /d `"$threeStudioPath`"",
  "`"$python`" launch.py $($overrides -join ' ')"
) -join " && "

cmd.exe /d /s /c $cmd
