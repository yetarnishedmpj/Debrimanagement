param(
    [int]$Port = 8501
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$dashboardPath = Join-Path $projectRoot 'dashboard.py'

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment not found at $venvPython. Run setup_and_test.ps1 first."
    exit 1
}

& $venvPython -m streamlit run $dashboardPath --server.port $Port
