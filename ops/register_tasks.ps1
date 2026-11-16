# Registers DCFG-Runner and DCFG-Watchdog for the current user (no admin). Re-run to update.
$ErrorActionPreference = 'Stop'
$distro = 'Ubuntu-24.04'
$linuxUser = 'harmeet'
$user = "$env:USERDOMAIN\$env:USERNAME"

function New-WslAction([string]$script) {
    New-ScheduledTaskAction -Execute 'wsl.exe' `
        -Argument "-d $distro -u $linuxUser -- bash -lc `"~/dcfg/ops/$script`""
}
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

# (a) Runner: at logon, unlimited runtime, restart on failure.
$runnerSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'DCFG-Runner' -Force -Principal $principal `
    -Action (New-WslAction 'run_queue.sh') `
    -Trigger (New-ScheduledTaskTrigger -AtLogOn -User $user) `
    -Settings $runnerSettings | Out-Null

# (b) Watchdog: every 30 min, indefinitely, 10 min limit.
$wdTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 30)
$wdSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'DCFG-Watchdog' -Force -Principal $principal `
    -Action (New-WslAction 'watchdog.sh') -Trigger $wdTrigger `
    -Settings $wdSettings | Out-Null

Get-ScheduledTask -TaskName 'DCFG-*' | Format-Table TaskName, State -AutoSize
