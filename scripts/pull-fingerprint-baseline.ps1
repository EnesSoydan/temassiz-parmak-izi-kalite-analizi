param(
  [string]$PackageName = 'com.anonymous.temassizparmakizikaliteanalizi',
  [string]$Serial = '',
  [ValidateSet('enrollment', 'probe')]
  [string]$Report = 'enrollment',
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\evaluation_reports')
)

$ErrorActionPreference = 'Stop'
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$reportFileName = if ($Report -eq 'probe') {
  'fingerprint-probe-report.json'
} else {
  'fingerprint-baseline-report.json'
}
$remotePath = "files/fingerprint-captures/$reportFileName"

if (-not (Test-Path -LiteralPath $adb)) {
  throw "ADB bulunamadı: $adb"
}

$devices = & $adb devices
$deviceSerials = @(
  $devices |
    Select-Object -Skip 1 |
    Where-Object { $_ -match "\sdevice$" } |
    ForEach-Object {
      # mDNS cihaz adları boşluk içerebilir; yalnızca ilk kelimeyi almak seri numarasını bozar.
      if ($_ -match '^(?<serial>.+?)\s+device(?:\s|$)') {
        $matches['serial'].Trim()
      }
    }
)
if ($Serial) {
  if ($Serial -notin $deviceSerials) {
    throw "İstenen ADB cihazı bağlı değil: $Serial"
  }
  $selectedSerial = $Serial
} elseif ($deviceSerials.Count -eq 1) {
  $selectedSerial = $deviceSerials[0]
} else {
  throw "Birden fazla veya sıfır ADB cihazı var. -Serial ile birini seç: $($deviceSerials -join ', ')"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outputPath = Join-Path $OutputDirectory "fingerprint-$Report-$timestamp.json"
$content = & $adb -s $selectedSerial exec-out run-as $PackageName cat $remotePath

if ($LASTEXITCODE -ne 0 -or -not $content) {
  throw "Cihazda $Report raporu okunamadı. Önce uygulamadaki ilgili rapor düğmesini çalıştır."
}
$json = $content -join "`n"
$null = $json | ConvertFrom-Json

[System.IO.File]::WriteAllText(
  [System.IO.Path]::GetFullPath($outputPath),
  $json + "`n",
  [System.Text.UTF8Encoding]::new($false)
)
Write-Output "$Report raporu kopyalandı: $([System.IO.Path]::GetFullPath($outputPath))"
