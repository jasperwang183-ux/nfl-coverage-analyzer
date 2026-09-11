# PostToolUse hook (Write|Edit): auto-format Python files with ruff.
# Uses the project venv's ruff if present, else ruff on PATH, else no-op.
try {
    $payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
    $file = $payload.tool_input.file_path
    if (-not $file) { $file = $payload.tool_response.filePath }
    if ($file -and $file -match '\.py$' -and (Test-Path $file)) {
        $root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
        $venvRuff = Join-Path $root '.venv\Scripts\ruff.exe'
        if (Test-Path $venvRuff) {
            & $venvRuff format --line-length 100 $file 2>$null | Out-Null
        } elseif (Get-Command ruff -ErrorAction SilentlyContinue) {
            ruff format --line-length 100 $file 2>$null | Out-Null
        }
    }
} catch {}
exit 0
