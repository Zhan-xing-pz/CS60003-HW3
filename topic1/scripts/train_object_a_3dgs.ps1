param(
  [string]$GaussianRepo = "third_party\gaussian-splatting",
  [string]$Source = "outputs\colmap\object_a_singlecam\dense",
  [string]$Model = "outputs\gaussian_splatting\object_a",
  [string]$CondaEnv = "gaussian_splatting_py310",
  [string]$DataDevice = "cpu",
  [int]$Iterations = 7000,
  [int]$Resolution = 2,
  [switch]$UseActivatedPython
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $GaussianRepo)) {
  throw "Gaussian Splatting repo not found: $GaussianRepo. Run scripts\setup_external_repos.ps1 first."
}
if (-not (Test-Path $Source)) {
  throw "Source COLMAP directory not found: $Source. Run scripts\prepare_object_a.py and scripts\run_colmap_pipeline.py first."
}

Push-Location $GaussianRepo
try {
  function Invoke-GaussianPython {
    param([string[]]$Arguments)
    if ($UseActivatedPython) {
      & python @Arguments
    }
    else {
      & conda run -n $CondaEnv python @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
      throw "3DGS command failed with exit code ${LASTEXITCODE}: python $($Arguments -join ' ')"
    }
  }

  Invoke-GaussianPython @("train.py", "-s", "..\..\$Source", "-m", "..\..\$Model", "--eval", "--iterations", "$Iterations", "-r", "$Resolution", "--data_device", "$DataDevice", "--test_iterations", "1000", "$Iterations", "--save_iterations", "$Iterations")
  Invoke-GaussianPython @("render.py", "-m", "..\..\$Model")
  Invoke-GaussianPython @("metrics.py", "-m", "..\..\$Model")
}
finally {
  Pop-Location
}
