param([ValidateSet('hud', 'terminal')][string]$Interface = 'hud')

# Run from the PowerShell session whose aliases/functions you want to use.
# Snapshots are local, temporary, never correction-training data.
$projectRoot = Microsoft.PowerShell.Management\Split-Path -Parent $PSScriptRoot
$shellName = if ($PSVersionTable.PSEdition -eq 'Core') { 'pwsh' } else { 'powershell' }
$snapshotPath = Microsoft.PowerShell.Management\Join-Path ([IO.Path]::GetTempPath()) ('intuition-shell-' + [guid]::NewGuid().ToString('N') + '.json')
$priorShell = $env:INTUITION_SHELL
$priorCatalog = $env:INTUITION_SHELL_CATALOG
$catalogEntries = @(Microsoft.PowerShell.Core\Get-Command -CommandType Alias,Function -All -ListImported | Microsoft.PowerShell.Core\ForEach-Object {
    [pscustomobject]@{name=$_.Name; kind=[string]$_.CommandType; definition=$_.Definition}
})
try {
    @{shell=$shellName; commands=$catalogEntries} | Microsoft.PowerShell.Utility\ConvertTo-Json -Depth 5 | Microsoft.PowerShell.Management\Set-Content -LiteralPath $snapshotPath -Encoding UTF8
    $env:INTUITION_SHELL = $shellName
    $env:INTUITION_SHELL_CATALOG = $snapshotPath
    Microsoft.PowerShell.Management\Push-Location -LiteralPath $projectRoot
    try {
        $entry = if ($Interface -eq 'hud') { 'start_ui.py' } else { 'intuitionos.py' }
        & (Microsoft.PowerShell.Management\Join-Path $projectRoot '.venv\Scripts\python.exe') $entry
    } finally { Microsoft.PowerShell.Management\Pop-Location }
} finally {
    $env:INTUITION_SHELL = $priorShell
    $env:INTUITION_SHELL_CATALOG = $priorCatalog
    Microsoft.PowerShell.Management\Remove-Item -LiteralPath $snapshotPath -ErrorAction SilentlyContinue
}
