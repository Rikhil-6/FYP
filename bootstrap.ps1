param(
    [ValidateSet('Setup', 'Ingest', 'Baseline', 'Xai', 'Siamese', 'Evaluate', 'All')]
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
        (Join-Path $env:USERPROFILE '.local\bin\uv.exe'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links\uv.exe'),
        (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe')
    )) { if (Test-Path -LiteralPath $candidate) { return $candidate } }
    return $null
}

$uv = Get-UvPath
if (-not $uv) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        & $winget.Source install --id astral-sh.uv --exact --source winget --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) { throw "winget install failed: $LASTEXITCODE" }
    } else {
        Invoke-Expression (Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1')
    }
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
    $uv = Get-UvPath
    if (-not $uv) { throw 'uv installed but is not on PATH. Open a new PowerShell window and rerun.' }
}

& $uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync failed: $LASTEXITCODE" }

function Invoke-DataIngest {
    Write-Host 'Downloading FMoW-WILDS (about 54 GB compressed; allow substantial disk space).'
    & $uv run ingest.py
    if ($LASTEXITCODE -ne 0) { throw "Dataset ingestion failed: $LASTEXITCODE" }
}
function Invoke-Train([string]$method) {
    & $uv run train.py --method $method --epochs $Epochs --batch-size $BatchSize
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
    & $uv run evaluate.py --method $method --checkpoint $checkpoint --batch-size $BatchSize --output (Join-Path $run 'metrics\recheck')
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for ${method}: $LASTEXITCODE" }
}

switch ($Action) {
    'Setup' { Write-Host 'Environment ready. Run -Action Ingest to download the benchmark.' }
    'Ingest' { Invoke-DataIngest }
    'Baseline' { Invoke-Train 'baseline' }
    'Xai' { Invoke-Train 'xai' }
    'Siamese' {
        $baseRun = Get-LatestRun 'xai'
        & $uv run train_siamese.py --base-checkpoint (Join-Path $baseRun 'models\best.pt') --epochs $Epochs --batch-size $BatchSize
        if ($LASTEXITCODE -ne 0) { throw "Siamese adaptation failed: $LASTEXITCODE" }
    }
    'Evaluate' { Invoke-Eval 'baseline'; Invoke-Eval 'xai' }
    'All' {
        Invoke-Train 'baseline'
        Invoke-Train 'xai'
        Invoke-Eval 'baseline'
        Invoke-Eval 'xai'
        $baseRun = Get-LatestRun 'xai'
        & $uv run train_siamese.py --base-checkpoint (Join-Path $baseRun 'models\best.pt') --epochs $Epochs --batch-size $BatchSize
        if ($LASTEXITCODE -ne 0) { throw "Siamese adaptation failed: $LASTEXITCODE" }
    }
}
