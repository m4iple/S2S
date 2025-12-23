# Script to activate conda environment and run .py

$condaBasePath = $null

$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if ($condaCmd -and $condaCmd.Source) {
    $condaBasePath = Split-Path (Split-Path $condaCmd.Source -Parent) -Parent
}

if (-not $condaBasePath) {
    $possiblePaths = @(
        "$env:USERPROFILE\anaconda3",
        "$env:USERPROFILE\miniconda3",
        "C:\ProgramData\Anaconda3",
        "C:\ProgramData\Miniconda3",
        "$env:LOCALAPPDATA\anaconda3",
        "$env:LOCALAPPDATA\miniconda3"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path "$path\Scripts\conda.exe") {
            $condaBasePath = $path
            Write-Host "Found conda at: $condaBasePath" -ForegroundColor Green
            break
        }
    }
}

if (-not $condaBasePath) {
    Write-Host "Error: conda installation not found" -ForegroundColor Red
    Write-Host "Searched in common locations but couldn't find conda" -ForegroundColor Yellow
    Write-Host "Please ensure Anaconda or Miniconda is installed" -ForegroundColor Yellow
    exit 1
}

$condaHook = "$condaBasePath\shell\condabin\conda-hook.ps1"
if (Test-Path $condaHook) {
    & $condaHook
} else {
    Write-Host "Using conda.bat directly..." -ForegroundColor Yellow
}

$envName = "s2s_env"

if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "Activating conda environment: $envName" -ForegroundColor Cyan
    conda activate $envName
} else {
    Write-Host "Activating conda environment: $envName (using conda.bat)" -ForegroundColor Cyan
    & "$condaBasePath\Scripts\activate.bat" $envName
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to activate conda environment '$envName'" -ForegroundColor Red
    Write-Host "Please create the environment first or update the environment name in this script" -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting main.py..." -ForegroundColor Green
python main.py
