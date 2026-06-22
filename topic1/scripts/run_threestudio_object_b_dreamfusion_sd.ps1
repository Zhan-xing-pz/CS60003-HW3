param(
  [string]$ThreeStudioRepo = "third_party\threestudio",
  [string]$EnvName = "threestudio_py310",
  [int]$Gpu = 0,
  [int]$MaxSteps = 800,
  [int]$ValInterval = 100,
  [int]$CheckpointInterval = 100,
  [string]$PretrainedModel = "stable-diffusion-v1-5/stable-diffusion-v1-5",
  [string]$Prompt = "a small jade dragon statue with translucent green material, intricate carved scales, and a polished museum-object look",
  [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path ".").Path
$threeStudioPath = (Resolve-Path $ThreeStudioRepo).Path
$python = Join-Path $env:USERPROFILE "miniconda3\envs\$EnvName\python.exe"
$vsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

if (-not (Test-Path $python)) {
  throw "Python for env '$EnvName' not found: $python"
}
if (-not (Test-Path $vsDevCmd)) {
  throw "Visual Studio DevCmd not found: $vsDevCmd"
}

$cacheRoot = Join-Path $repoRoot ".cache"
$hfDir = Join-Path $cacheRoot "huggingface"
$runTmp = Join-Path $env:TEMP "hw3_threestudio"
$torchExt = Join-Path $env:TEMP "hw3_torch_extensions"

New-Item -ItemType Directory -Force -Path `
  $hfDir, `
  (Join-Path $runTmp "matplotlib"), `
  (Join-Path $runTmp "wandb"), `
  $torchExt | Out-Null

$tag = "object_b_dragon_${MaxSteps}"
$overrides = @(
  "--config configs/dreamfusion-sd.yaml",
  "--train",
  "--gpu $Gpu",
  "name=dreamfusion-object-b",
  "tag=$tag",
  "system.prompt_processor.pretrained_model_name_or_path=$PretrainedModel",
  "system.guidance.pretrained_model_name_or_path=$PretrainedModel",
  "system.prompt_processor.prompt=`"$Prompt`"",
  "trainer.max_steps=$MaxSteps",
  "trainer.val_check_interval=$ValInterval",
  "checkpoint.every_n_train_steps=$CheckpointInterval",
  "data.width=64",
  "data.height=64",
  "data.batch_size=1",
  "system.renderer.num_samples_per_ray=128",
  "system.loggers.wandb.enable=false",
  "system.optimizer.args.foreach=false",
  "system.geometry.pos_encoding_config.otype=ProgressiveBandFrequency",
  "system.geometry.pos_encoding_config.n_frequencies=12",
  "system.geometry.pos_encoding_config.include_xyz=true",
  "system.geometry.mlp_network_config.otype=VanillaMLP",
  "system.geometry.mlp_network_config.activation=ReLU",
  "system.geometry.mlp_network_config.output_activation=none",
  "system.geometry.mlp_network_config.n_neurons=64",
  "system.geometry.mlp_network_config.n_hidden_layers=2",
  "system.guidance.enable_attention_slicing=true",
  "system.background.dir_encoding_config.otype=ProgressiveBandFrequency",
  "system.background.dir_encoding_config.n_frequencies=4",
  "system.background.dir_encoding_config.include_xyz=true",
  "system.background.mlp_network_config.otype=VanillaMLP",
  "system.background.mlp_network_config.activation=ReLU",
  "system.background.mlp_network_config.output_activation=none",
  "system.background.mlp_network_config.n_neurons=16",
  "system.background.mlp_network_config.n_hidden_layers=2"
)

if ($ResumeCheckpoint -ne "") {
  $resolvedResume = (Resolve-Path $ResumeCheckpoint).Path
  $overrides += "resume=$resolvedResume"
}

$cmd = @(
  "`"$vsDevCmd`" -arch=x64",
  "set `"VSLANG=1033`"",
  "set `"TORCH_EXTENSIONS_DIR=$torchExt`"",
  "set `"TORCH_CUDA_ARCH_LIST=12.0`"",
  "set `"TCNN_CUDA_ARCHITECTURES=120`"",
  "set `"TORCH_DONT_CHECK_COMPILER_ABI=1`"",
  "set `"DISTUTILS_USE_SDK=1`"",
  "set `"MSSdk=1`"",
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
