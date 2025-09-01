# start.ps1 — bootstrap venv, install deps, run bot
param(
  [string]$Port = "COM4"
)

$ErrorActionPreference = "Stop"
$base = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $base

if (!(Test-Path .env) -and (Test-Path .env.example)) {
  Copy-Item .env.example .env
}

if (!(Test-Path .venv)) {
  py -3 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install -r requirements.txt

Write-Host "[INFO] starting winsammi.py on $Port"
& .\.venv\Scripts\python.exe .\winsammi.py --port $Port
