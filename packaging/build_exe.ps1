param(
    [switch]$SkipInstaller,
    [switch]$SkipDependencyInstall,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$SpecPath = Join-Path $PSScriptRoot "AutoPDFTranslator.spec"
$DistDir = Join-Path $ProjectRoot "dist"
$BuildDir = Join-Path $ProjectRoot "build"
$InstallerScript = Join-Path $PSScriptRoot "installer.iss"

Set-Location $ProjectRoot

$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

if ($Clean) {
    if (Test-Path $DistDir) {
        Remove-Item -LiteralPath $DistDir -Recurse -Force
    }
    if (Test-Path $BuildDir) {
        Remove-Item -LiteralPath $BuildDir -Recurse -Force
    }
}

if (!$SkipDependencyInstall) {
    python -m pip install -r requirements.txt
    python -m pip install pyinstaller
}

pyinstaller --noconfirm --clean $SpecPath

$ExePath = Join-Path $DistDir "AutoPDFTranslator\AutoPDFTranslator.exe"
if (!(Test-Path $ExePath)) {
    throw "Build failed: $ExePath was not created."
}

Write-Host "EXE build complete:" -ForegroundColor Green
Write-Host $ExePath

if ($SkipInstaller) {
    Write-Host "Installer step skipped." -ForegroundColor Yellow
    exit 0
}

$InnoCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { $_ -and (Test-Path $_) }

if (!$InnoCandidates) {
    Write-Host "Inno Setup 6 was not found. Install it, then rerun this script to create the installer." -ForegroundColor Yellow
    Write-Host "Download: https://jrsoftware.org/isinfo.php"
    exit 0
}

$InnoCompiler = @($InnoCandidates)[0]
& $InnoCompiler $InstallerScript

$InstallerOutput = Join-Path $ProjectRoot "installer"
Write-Host "Installer build complete. Output folder:" -ForegroundColor Green
Write-Host $InstallerOutput
