# Registers DCFG-Runner and DCFG-Watchdog for the current user (no admin). Re-run to update.
$ErrorActionPreference = 'Stop'
$distro = 'Ubuntu-24.04'
$linuxUser = 'harmeet'
$user = "$env:USERDOMAIN\$env:USERNAME"

# Hidden launcher: wscript runs wsl.exe with window style 0 (hidden), so there is no console
# window to close by accident. The script name (run_queue.sh / watchdog.sh) is argument 0.
$dir = Join-Path $env:LOCALAPPDATA 'dcfg'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$vbsPath = Join-Path $dir 'run_wsl.vbs'
$vbs = @"
Dim sh, rc
Set sh = CreateObject("WScript.Shell")
rc = sh.Run("wsl.exe -d $distro -u $linuxUser -- bash -lc ""~/dcfg/ops/" & WScript.Arguments(0) & """", 0, True)
WScript.Quit rc
"@
Set-Content -Path $vbsPath -Value $vbs -Encoding ASCII

function New-WslAction([string]$script) {
    New-ScheduledTaskAction -Execute 'wscript.exe' `
        -Argument "//B //Nologo `"$vbsPath`" $script"
}
# LogonType Interactive is deliberate: the user logs in after a reboot (PLAN Section 3), and
# Interactive needs no stored password. The tasks therefore run only while logged on.
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
# Empty Duration means "repeat indefinitely" (works on Windows PowerShell 5.1).
$wdTrigger.Repetition.Duration = ''
$wdSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'DCFG-Watchdog' -Force -Principal $principal `
    -Action (New-WslAction 'watchdog.sh') -Trigger $wdTrigger `
    -Settings $wdSettings | Out-Null

Get-ScheduledTask -TaskName 'DCFG-*' | Format-Table TaskName, State -AutoSize
$rep = (Get-ScheduledTask -TaskName 'DCFG-Watchdog').Triggers[0].Repetition
"Watchdog repetition: Interval='$($rep.Interval)' Duration='$($rep.Duration)' (empty = indefinite)"
