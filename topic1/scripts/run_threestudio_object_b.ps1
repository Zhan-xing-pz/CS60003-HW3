param(
  [string]$ThreeStudioRepo = "third_party\threestudio",
  [string]$Prompt = "a small jade dragon statue with translucent green material, intricate carved scales, and a polished museum-object look",
  [int]$Gpu = 0,
  [int]$MaxSteps = 8000
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $ThreeStudioRepo)) {
  throw "threestudio repo not found: $ThreeStudioRepo. Run scripts\setup_external_repos.ps1 first."
}

Push-Location $ThreeStudioRepo
try {
  python launch.py --config configs/dreamfusion-sd.yaml --train --gpu $Gpu system.prompt_processor.prompt="$Prompt" trainer.max_steps=$MaxSteps
  Write-Host "After training, export the latest trial mesh with:"
  Write-Host "python launch.py --config path\to\trial\configs\parsed.yaml --export --gpu $Gpu resume=path\to\trial\ckpts\last.ckpt system.exporter_type=mesh-exporter system.exporter.fmt=obj"
}
finally {
  Pop-Location
}
