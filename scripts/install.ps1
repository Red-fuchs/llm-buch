# Windows: build release and install buch.exe to %LOCALAPPDATA%\Programs\buch, add that folder to the user PATH.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
cargo build --release
if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
$dest = Join-Path $env:LOCALAPPDATA 'Programs\buch'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item -Force 'target\release\buch.exe' (Join-Path $dest 'buch.exe')
$user = [Environment]::GetEnvironmentVariable('Path', 'User')
if (($user -split ';') -notcontains $dest) {
    [Environment]::SetEnvironmentVariable('Path', ($user.TrimEnd(';') + ';' + $dest), 'User')
    Write-Host "Added $dest to your user PATH. Open a NEW terminal for it to take effect."
}
Write-Host "Installed $dest\buch.exe"
