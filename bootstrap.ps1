param(
    [ValidateSet('Setup', 'Ingest', 'Baseline', 'Xai', 'Siamese', 'Evaluate', 'Drift', 'All')]
    [string]$Action = 'Setup',
    [int]$Epochs = 3,
    [int]$BatchSize = 8
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Get-UvPath {
    $found = Get-Command uv -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    foreach ($candidate in @(
        (Join-Path $PSScriptRoot 'tools\uv.exe'),
        (Join-Path $env:USERPROFILE '.local\bin\uv.exe'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links\uv.exe'),
        (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe')
    )) { if (Test-Path -LiteralPath $candidate) { return $candidate } }
    return $null
}

$uv = Get-UvPath
if (-not $uv) {
    $env:UV_UNMANAGED_INSTALL = Join-Path $PSScriptRoot 'tools'
    $env:UV_NO_MODIFY_PATH = '1'
    Invoke-Expression (Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1')
    $uv = Get-UvPath
    if (-not $uv) { throw 'The project-local uv installation did not produce tools\uv.exe.' }
}

& $uv sync
if ($LASTEXITCODE -ne 0) {
    # Some Windows uv versions create Python 3.11 but fail to follow its minor-version junction once.
    $pythonRoot = Join-Path $env:APPDATA 'uv\python'
    $installed = Get-ChildItem -LiteralPath $pythonRoot -Directory -Filter 'cpython-3.11.*-windows-x86_64-none' -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    $pythonExe = if ($installed) { Join-Path $installed.FullName 'python.exe' } else { $null }
    if (-not $pythonExe -or -not (Test-Path -LiteralPath $pythonExe)) { throw "uv sync failed: $LASTEXITCODE" }
    & $uv sync --python $pythonExe
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed: $LASTEXITCODE" }
}

function Invoke-DataIngest {
    Write-Host 'Downloading CLEAR-100 image archives (about 4.6 GB total; resumable).'
    & $uv run fyp-ingest
    if ($LASTEXITCODE -ne 0) { throw "Dataset ingestion failed: $LASTEXITCODE" }
}
function Invoke-Train([string]$method) {
    & $uv run fyp-train --method $method --epochs $Epochs --batch-size $BatchSize
    if ($LASTEXITCODE -ne 0) { throw "Training failed for ${method}: $LASTEXITCODE" }
}
function Get-LatestRun([string]$method) {
    $folder = Join-Path $PSScriptRoot "runs\$method"
    if (-not (Test-Path -LiteralPath $folder)) { throw "No runs for $method" }
    $run = Get-ChildItem -LiteralPath $folder -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $run -or -not (Test-Path -LiteralPath (Join-Path $run.FullName 'models\best.pt'))) {
        throw "No best checkpoint for $method"
    }
    return $run.FullName
}
function Invoke-Eval([string]$method) {
    $run = Get-LatestRun $method
    $checkpoint = Join-Path $run 'models\best.pt'
    & $uv run fyp-evaluate --method $method --checkpoint $checkpoint --batch-size $BatchSize --output (Join-Path $run 'metrics\recheck')
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for ${method}: $LASTEXITCODE" }
}
function Invoke-Drift([string]$method) {
    $run = Get-LatestRun $method
    $checkpoint = Join-Path $run 'models\best.pt'
    & $uv run fyp-evaluate-drift --method $method --checkpoint $checkpoint --batch-size $BatchSize --output (Join-Path $run 'metrics\drift')
    if ($LASTEXITCODE -ne 0) { throw "Drift evaluation failed for ${method}: $LASTEXITCODE" }
}

switch ($Action) {
    'Setup' { Write-Host 'Environment ready. Run -Action Ingest to download the benchmark.' }
    'Ingest' { Invoke-DataIngest }
    'Baseline' { Invoke-Train 'baseline' }
    'Xai' { Invoke-Train 'xai' }
    'Siamese' {
        $baseRun = Get-LatestRun 'xai'
        & $uv run fyp-train-router --base-checkpoint (Join-Path $baseRun 'models\best.pt') --epochs $Epochs --batch-size $BatchSize
        if ($LASTEXITCODE -ne 0) { throw "Siamese adaptation failed: $LASTEXITCODE" }
    }
    'Evaluate' { Invoke-Eval 'baseline'; Invoke-Eval 'xai' }
    'Drift' { Invoke-Drift 'baseline'; Invoke-Drift 'xai' }
    'All' {
        Invoke-Train 'baseline'
        Invoke-Train 'xai'
        Invoke-Eval 'baseline'
        Invoke-Eval 'xai'
        $baseRun = Get-LatestRun 'xai'
        & $uv run fyp-train-router --base-checkpoint (Join-Path $baseRun 'models\best.pt') --epochs $Epochs --batch-size $BatchSize
        if ($LASTEXITCODE -ne 0) { throw "Siamese adaptation failed: $LASTEXITCODE" }
    }
}
