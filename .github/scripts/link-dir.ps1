param(
    [Parameter(Mandatory)][string]$Source,
    [Parameter(Mandatory)][string]$Target,
    [string[]]$Exclude = @("node_modules")
)

$source = Resolve-Path $Source
New-Item -ItemType Directory -Force -Path $Target | Out-Null

Get-ChildItem -Path $source -Directory | Where-Object { $_.Name -notin $Exclude } | ForEach-Object {
    $link = Join-Path $Target $_.Name
    if (Test-Path $link) {
        Write-Host "SKIP (exists): $($_.Name)"
    } else {
        cmd /c mklink /J "$link" "$($_.FullName)" | Out-Null
        Write-Host "LINKED: $($_.Name) -> $($_.FullName)"
    }
}
