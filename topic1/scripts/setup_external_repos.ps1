param(
  [string]$ThirdPartyDir = "third_party"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $ThirdPartyDir | Out-Null

if (-not (Test-Path "$ThirdPartyDir\gaussian-splatting")) {
  git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive "$ThirdPartyDir\gaussian-splatting"
} else {
  Write-Host "gaussian-splatting already exists"
}

if (-not (Test-Path "$ThirdPartyDir\threestudio")) {
  git clone https://github.com/threestudio-project/threestudio.git "$ThirdPartyDir\threestudio"
} else {
  Write-Host "threestudio already exists"
}

Write-Host "External repositories are under $ThirdPartyDir"
