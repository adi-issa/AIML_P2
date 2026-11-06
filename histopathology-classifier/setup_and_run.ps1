$ErrorActionPreference = "Stop"

$PortalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualEnvironment = Join-Path $PortalRoot ".venv"
$VirtualPython = Join-Path $VirtualEnvironment "Scripts\python.exe"
$Requirements = Join-Path $PortalRoot "requirements.txt"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The Python launcher (py.exe) was not found. Install Python 3.11 first."
}

try {
    & py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" | Out-Null
}
catch {
    throw "Python 3.11 is required. Install it, then run this script again."
}

if (-not (Test-Path -LiteralPath $VirtualPython)) {
    Write-Host "Creating the Python 3.11 environment..." -ForegroundColor Cyan
    & py -3.11 -m venv $VirtualEnvironment
}

& $VirtualPython -c "import importlib.util as u, sys; sys.exit(int(any(u.find_spec(p) is None for p in ('tensorflow','keras','streamlit','numpy','PIL'))))"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing portal dependencies (first run can take several minutes)..." -ForegroundColor Cyan
    & $VirtualPython -m pip install --upgrade pip
    & $VirtualPython -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed. Review the pip error above."
    }
}

Write-Host "Starting Histopathology Cancer Classifier..." -ForegroundColor Green
Set-Location $PortalRoot
& $VirtualPython -m streamlit run "app.py"
