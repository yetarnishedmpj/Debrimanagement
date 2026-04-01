[CmdletBinding()]
param(
    [string]$PythonBin = $env:PYTHON_BIN,
    [ValidateSet('phased', 'one-shot')]
    [string]$InstallMode = $(if ($env:INSTALL_MODE) { $env:INSTALL_MODE } else { 'phased' }),
    [string]$VenvDir = $(if ($env:VENV_DIR) { $env:VENV_DIR } else { (Join-Path $PSScriptRoot '.venv') }),
    [string]$RequirementsFile = $(if ($env:REQUIREMENTS_FILE) { $env:REQUIREMENTS_FILE } else { (Join-Path $PSScriptRoot 'requirements.txt') }),
    [string]$LogDir = $(if ($env:LOG_DIR) { $env:LOG_DIR } else { (Join-Path $PSScriptRoot 'logs') }),
    [string]$TorchIndexUrl = $(if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { '' }),
    [string]$TestTarget = $(if ($env:TEST_TARGET) { $env:TEST_TARGET } else { (Join-Path $PSScriptRoot 'tests') }),
    [string]$PytestFlags = $(if ($env:PYTEST_FLAGS) { $env:PYTEST_FLAGS } else { '-q' }),
    [string]$ExtraPipArgs = $(if ($env:EXTRA_PIP_ARGS) { $env:EXTRA_PIP_ARGS } else { '' }),
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$RecreateVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("bootstrap_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

function Write-Log {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Split-Args {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return @()
    }

    return @($Text -split '\s+' | Where-Object { $_ -and $_.Trim().Length -gt 0 })
}

function Resolve-PythonCommand {
    param([string]$Override)

    if (-not [string]::IsNullOrWhiteSpace($Override)) {
        $parts = Split-Args $Override
        if ($parts.Count -eq 0) {
            throw 'PYTHON_BIN was provided but could not be parsed.'
        }

        return @{
            FilePath = $parts[0]
            Arguments = if ($parts.Count -gt 1) { @($parts[1..($parts.Count - 1)]) } else { @() }
        }
    }

    $candidates = @(
        @{ FilePath = 'python'; Arguments = @() },
        @{ FilePath = 'python3'; Arguments = @() },
        @{ FilePath = 'py'; Arguments = @('-3') }
    )

    foreach ($candidate in $candidates) {
        if (Get-Command $candidate.FilePath -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }

    throw 'No Python interpreter found. Pass -PythonBin "py -3" or set $env:PYTHON_BIN.'
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )

    $commandText = @($FilePath) + $Arguments
    Write-Log ("+ {0}" -f ($commandText -join ' '))

    & $FilePath @Arguments 2>&1 | Tee-Object -FilePath $LogFile -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw ("Command failed with exit code {0}: {1}" -f $exitCode, ($commandText -join ' '))
    }
}

function Resolve-VenvPython {
    param([string]$Directory)

    $windowsPython = Join-Path $Directory 'Scripts\python.exe'
    if (Test-Path $windowsPython) {
        return $windowsPython
    }

    $posixPython = Join-Path $Directory 'bin\python'
    if (Test-Path $posixPython) {
        return $posixPython
    }

    throw 'Virtual environment created, but its Python executable was not found.'
}

function Install-RequirementsPhased {
    param(
        [string]$VenvPython,
        [string]$TorchIndex,
        [string]$ExtraArgs
    )

    $commonArgs = Split-Args $ExtraArgs

    Invoke-LoggedCommand -FilePath $VenvPython -Arguments (@(
        '-m', 'pip', 'install',
        'numpy==1.26.4',
        'dm-tree==0.1.8',
        'requests==2.32.5',
        'matplotlib==3.10.8',
        'plotly==5.24.1',
        'streamlit==1.39.0',
        'pytest==8.4.2'
    ) + $commonArgs)

    Invoke-LoggedCommand -FilePath $VenvPython -Arguments (@(
        '-m', 'pip', 'install',
        'gymnasium==0.28.1'
    ) + $commonArgs)

    $torchArgs = @('-m', 'pip', 'install')
    if (-not [string]::IsNullOrWhiteSpace($TorchIndex)) {
        $torchArgs += @('--index-url', $TorchIndex)
    }
    $torchArgs += @('torch==2.5.1') + $commonArgs
    Invoke-LoggedCommand -FilePath $VenvPython -Arguments $torchArgs

    Invoke-LoggedCommand -FilePath $VenvPython -Arguments (@(
        '-m', 'pip', 'install',
        'ray[rllib]==2.40.0'
    ) + $commonArgs)
}

function Install-RequirementsOneShot {
    param(
        [string]$VenvPython,
        [string]$Requirements,
        [string]$ExtraArgs
    )

    Invoke-LoggedCommand -FilePath $VenvPython -Arguments (@(
        '-m', 'pip', 'install',
        '-r', $Requirements
    ) + (Split-Args $ExtraArgs))
}

$pythonCommand = Resolve-PythonCommand -Override $PythonBin
Write-Log ("Using bootstrap log {0}" -f $LogFile)
Write-Log ("Resolved host Python command: {0}" -f ((@($pythonCommand.FilePath) + $pythonCommand.Arguments) -join ' '))
Write-Log ("Install mode: {0}" -f $InstallMode)

if ($RecreateVenv -and (Test-Path $VenvDir)) {
    Write-Log ("Removing existing virtual environment at {0}" -f $VenvDir)
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvDir)) {
    Write-Log ("Creating virtual environment at {0}" -f $VenvDir)
    Invoke-LoggedCommand -FilePath $pythonCommand.FilePath -Arguments ($pythonCommand.Arguments + @('-m', 'venv', $VenvDir))
} else {
    Write-Log ("Reusing existing virtual environment at {0}" -f $VenvDir)
}

$venvPython = Resolve-VenvPython -Directory $VenvDir
Write-Log ("Resolved venv Python: {0}" -f $venvPython)

if (-not $SkipInstall) {
    Invoke-LoggedCommand -FilePath $venvPython -Arguments (@(
        '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'
    ) + (Split-Args $ExtraPipArgs))

    switch ($InstallMode) {
        'phased' {
            Install-RequirementsPhased -VenvPython $venvPython -TorchIndex $TorchIndexUrl -ExtraArgs $ExtraPipArgs
        }
        'one-shot' {
            Install-RequirementsOneShot -VenvPython $venvPython -Requirements $RequirementsFile -ExtraArgs $ExtraPipArgs
        }
    }
} else {
    Write-Log 'SKIP_INSTALL is set. Skipping dependency installation.'
}

Invoke-LoggedCommand -FilePath $venvPython -Arguments @('-m', 'compileall', $PSScriptRoot)
Invoke-LoggedCommand -FilePath $venvPython -Arguments @('-m', 'pip', 'check')

if (-not $SkipTests) {
    Invoke-LoggedCommand -FilePath $venvPython -Arguments (@('-m', 'pytest') + (Split-Args $PytestFlags) + @($TestTarget))
} else {
    Write-Log 'SKIP_TESTS is set. Skipping pytest.'
}

Write-Log 'Bootstrap finished successfully.'

