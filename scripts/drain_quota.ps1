# Re-run a generation condition until it completes, draining the API quota as it refills.
#
#   powershell -NoProfile -File scripts\drain_quota.ps1
#   powershell -NoProfile -File scripts\drain_quota.ps1 -Conditions "rag:bm25-light" -SleepSeconds 900
#
# Groq's free tier is a rolling window that refills continuously at about 134 tokens per minute
# (docs/design.md, section 10), so a run stops at the limit (exit code 3) with most of its work still to
# do. Every answer is cached, so re-running costs nothing for what is already done and picks up exactly
# where it stopped. This loop does that on a timer until the condition finishes.
#
# It writes to data/drain_quota.log and exits when the run reports success. Stop it with:
#   Get-CimInstance Win32_Process -Filter "CommandLine LIKE '%drain_quota%'" | ForEach-Object { Stop-Process -Id $_.ProcessId }

param(
    [string]$Conditions = "rag:rerank_hybrid_bm25-light__bge-m3",
    [string]$Corpus = "unpc",
    [int]$Attempts = 120,
    [int]$SleepSeconds = 1800
)

Set-Location (Join-Path $PSScriptRoot "..")
$log = "data/drain_quota.log"
"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm') starting: $Conditions, up to $Attempts attempts every $SleepSeconds s ===" |
    Out-File -Append -Encoding utf8 $log

for ($i = 1; $i -le $Attempts; $i++) {
    $output = & .\.venv\Scripts\python.exe scripts/run_generation.py --corpus $Corpus --conditions $Conditions 2>&1
    $code = $LASTEXITCODE
    $tail = ($output | Select-Object -Last 2) -join " | "
    "$(Get-Date -Format 'HH:mm') attempt $i exit $code : $tail" | Out-File -Append -Encoding utf8 $log
    if ($code -eq 0) {
        "$(Get-Date -Format 'HH:mm') COMPLETE after $i attempts" | Out-File -Append -Encoding utf8 $log
        break
    }
    if ($code -ne 3) {
        "$(Get-Date -Format 'HH:mm') stopping: exit $code is not the daily-limit code" | Out-File -Append -Encoding utf8 $log
        break
    }
    Start-Sleep -Seconds $SleepSeconds
}
