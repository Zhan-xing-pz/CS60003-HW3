param(
  [string]$EnvName = "threestudio_py310",
  [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"

$envList = conda env list
$exists = $false
foreach ($line in $envList) {
  if ($line -match "^\s*$([regex]::Escape($EnvName))\s") {
    $exists = $true
    break
  }
}
if (-not $exists) {
  conda env create -f environment.threestudio.yml
}

conda run -n $EnvName python -m pip install --upgrade pip setuptools wheel
conda run -n $EnvName python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url $TorchIndex

conda run -n $EnvName python -m pip install `
  lightning==2.0.0 `
  "pydantic==1.10.26" `
  omegaconf==2.3.0 `
  jaxtyping `
  typeguard `
  diffusers==0.19.3 `
  transformers==4.28.1 `
  accelerate==0.21.0 `
  opencv-python `
  tensorboard `
  matplotlib `
  "imageio[ffmpeg]>=2.28.0" `
  xatlas `
  "trimesh[easy]" `
  networkx `
  PyMCubes `
  "wandb==0.15.12" `
  torchmetrics `
  IPython `
  ipywidgets `
  einops `
  kornia `
  taming-transformers-rom1504 `
  safetensors `
  huggingface_hub==0.16.4 `
  ninja

conda run -n $EnvName python -m pip install "setuptools<81"
conda run -n $EnvName python -m pip install --no-build-isolation git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
conda run -n $EnvName python -m pip install git+https://github.com/openai/CLIP.git

conda run -n $EnvName python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'arch', torch.cuda.get_device_capability(0))"
