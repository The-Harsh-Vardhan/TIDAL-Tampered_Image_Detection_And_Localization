param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",

    [int]$Port = 8000,

    [int]$MaxConcurrent = 4
)

$ErrorActionPreference = "Stop"

$bundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $bundleDir "..\\..\\..")).Path
$venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"

if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = "python"
}

$env:DEVICE = $Device
$env:MODEL_DIR = $bundleDir
$env:MODEL_FILENAME = "best_model.pt"
$env:MODEL_VERSION = "vR.P.30.1"
$env:MAX_CONCURRENT = "$MaxConcurrent"

Write-Host "Repo root: $repoRoot"
Write-Host "Model bundle: $bundleDir"
Write-Host "Device: $Device"
Write-Host "Port: $Port"

Set-Location $repoRoot
& $pythonExe -m uvicorn backend.app:app --host 0.0.0.0 --port $Port
