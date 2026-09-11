# PreToolUse hook (Bash): deny destructive commands.
try {
    $payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
    $cmd = $payload.tool_input.command
} catch { exit 0 }
if (-not $cmd) { exit 0 }

$patterns = @(
    'rm\s+(-\w*\s+)*-\w*[rf]\w*[rf]',        # rm -rf and variants
    'git\s+push\s+(\S+\s+)*--force(\s|$)',   # force push (--force-with-lease allowed)
    'git\s+reset\s+--hard',
    'git\s+clean\s+(-\w*\s+)*-\w*f',
    'Remove-Item\s+.*-Recurse.*-Force',
    'rmdir\s+/s',
    'del\s+/[sf]',
    'git\s+checkout\s+\.',
    '>\s*data[/\\]raw'                        # clobbering raw data
)

foreach ($p in $patterns) {
    if ($cmd -match $p) {
        $out = @{
            hookSpecificOutput = @{
                hookEventName            = 'PreToolUse'
                permissionDecision       = 'deny'
                permissionDecisionReason = "Blocked by block-dangerous-bash hook (matched: $p). Use a safer alternative, or run it yourself if truly intended."
            }
        } | ConvertTo-Json -Depth 5 -Compress
        Write-Output $out
        exit 0
    }
}
exit 0
