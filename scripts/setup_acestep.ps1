# Install ACE-Step 1.5 for Songly on Windows.
# Follows ace-step/docs/en/INSTALL.md (uv sync) and pins a known commit.
# Songly then runs generation through .venv-acestep (see SONGCRAFT_ACESTEP_PYTHON).

$ErrorActionPreference = "Stop"

# Pinned revision of https://github.com/ACE-Step/ACE-Step-1.5
# Update this when you intentionally upgrade ACE-Step.
$AceStepCommit = "ca1e85fe9430179831e6bc6be790c332190a3866"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$AceStepDir = Join-Path $RepoRoot "ace-step"
$VenvDir = Join-Path $RepoRoot ".venv-acestep"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Find-Python311Or312 {
    foreach ($cmd in @(
        @{ Exe = "py"; Args = @("-3.12", "-c", "import sys; print(sys.executable)") },
        @{ Exe = "py"; Args = @("-3.11", "-c", "import sys; print(sys.executable)") },
        @{ Exe = "python3.12"; Args = @("-c", "import sys; print(sys.executable)") },
        @{ Exe = "python3.11"; Args = @("-c", "import sys; print(sys.executable)") }
    )) {
        try {
            $out = & $cmd.Exe @($cmd.Args) 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                return $out.Trim()
            }
        } catch {
            # try next candidate
        }
    }
    return $null
}

Push-Location $RepoRoot
try {
    $python = Find-Python311Or312
    if (-not $python) {
        Write-Host @"
ACE-Step needs Python 3.11 or 3.12 (stable, not a pre-release).
Install one of those versions, then re-run:
  make setup
  # or: powershell -File scripts/setup_acestep.ps1
Current 'python' on PATH is not used if it is outside 3.11-3.12.
"@
        exit 1
    }

    $ver = & $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($ver -notin @("3.11", "3.12")) {
        Write-Host "Refusing to continue: $python reports Python $ver (need 3.11 or 3.12)."
        exit 1
    }
    Write-Host "Using Python $ver at $python"

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Installing uv (ACE-Step's package manager)..."
        # From ace-step/docs/en/INSTALL.md
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Host "uv was installed but is not on PATH in this shell. Open a new terminal and re-run this script."
            exit 1
        }
    }

    if (-not (Test-Path (Join-Path $AceStepDir "pyproject.toml"))) {
        Write-Host "Cloning ACE-Step 1.5 into ace-step/ ..."
        git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git $AceStepDir
        # depth-1 may not include the pinned commit; fetch that commit explicitly.
        Push-Location $AceStepDir
        try {
            git fetch --depth 1 origin $AceStepCommit
            git checkout $AceStepCommit
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "ace-step/ already present; checking out pinned commit $AceStepCommit"
        Push-Location $AceStepDir
        try {
            git fetch --depth 1 origin $AceStepCommit 2>$null
            git checkout $AceStepCommit
        } catch {
            Write-Host "WARNING: could not check out $AceStepCommit. Using the existing checkout."
            Write-Host $_.Exception.Message
        } finally {
            Pop-Location
        }
    }

    if (-not (Test-Path $VenvPython)) {
        Write-Host "Creating .venv-acestep ..."
        & $python -m venv $VenvDir
    }

    Write-Host "Running uv sync into .venv-acestep (CUDA wheels come from ace-step/pyproject.toml) ..."
    # UV_PROJECT_ENVIRONMENT makes uv use Songly's .venv-acestep instead of ace-step/.venv.
    # Not fully re-verified in this session on a clean machine; if sync fails, see ace-step/docs/en/INSTALL.md.
    $env:UV_PROJECT_ENVIRONMENT = $VenvDir
    Push-Location $AceStepDir
    try {
        uv sync
    } finally {
        Pop-Location
        Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
    }

    & $VenvPython -c "import acestep; print('ace-step import ok:', acestep.__file__)"
    Write-Host @"

Setup finished.
- ACE-Step checkout: $AceStepDir
- Python for Songly: $VenvPython
First real song will download weights into ace-step/checkpoints (gitignored).
"@
} finally {
    Pop-Location
}
