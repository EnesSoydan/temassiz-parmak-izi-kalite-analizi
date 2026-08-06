param(
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')) 'YOLO\segmentation_data\raw\phone-validation'),
  [string]$Package = 'com.anonymous.temassizparmakizikaliteanalizi',
  [string]$Serial = '',
  [string]$AdbPath = '',
  [string]$RemoteDirectory = 'files/fingerprint-captures/aligned-canonical-roi',
  [string]$NamePattern = '*'
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($AdbPath)) {
  $AdbPath = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
}
if (-not (Test-Path -LiteralPath $AdbPath)) {
  throw "adb bulunamadı: $AdbPath"
}

if ([string]::IsNullOrWhiteSpace($Serial)) {
  $deviceLines = @(& $AdbPath devices | Select-Object -Skip 1 | Where-Object { $_ -match '\sdevice\s*$' })
  if ($deviceLines.Count -ne 1) {
    throw "Tek bir bağlı adb cihazı bekleniyor. -Serial ile cihazı belirtin. Bulunan: $($deviceLines.Count)"
  }
  $Serial = ($deviceLines[0] -split '\s+')[0]
}

$output = [System.IO.Path]::GetFullPath($OutputRoot)
$phoneRoot = Join-Path $output 'raw\phone-validation'
$imageOutput = Join-Path $phoneRoot 'images'
$annotationOutput = Join-Path $output 'annotations\phone-validation-template.json'
New-Item -ItemType Directory -Force -Path $imageOutput | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $annotationOutput) | Out-Null

$files = @(& $AdbPath -s $Serial shell run-as $Package find $remoteDirectory -type f)
$files = @($files | Where-Object { [System.IO.Path]::GetFileName($_.Trim()) -like $NamePattern })
if ($LASTEXITCODE -ne 0 -or $files.Count -eq 0) {
  throw "Canonical ROI bulunamadı veya run-as başarısız oldu. Paket: $Package"
}

function Copy-RemoteFile([string]$remotePath, [string]$localPath) {
  $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
  $startInfo.FileName = $AdbPath
  $startInfo.Arguments = "-s `"$Serial`" exec-out run-as $Package cat $remotePath"
  $startInfo.UseShellExecute = $false
  $startInfo.RedirectStandardOutput = $true
  $startInfo.CreateNoWindow = $true
  $process = [System.Diagnostics.Process]::new()
  $process.StartInfo = $startInfo
  [void]$process.Start()
  $targetStream = [System.IO.File]::Open($localPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
  try {
    $process.StandardOutput.BaseStream.CopyTo($targetStream)
  } finally {
    $targetStream.Dispose()
  }
  $process.WaitForExit()
  if ($process.ExitCode -ne 0) {
    throw "Telefon dosyası alınamadı: $remotePath"
  }
}

$entries = [System.Collections.Generic.List[object]]::new()
$copied = 0
foreach ($remotePath in $files) {
  $name = [System.IO.Path]::GetFileName($remotePath.Trim())
  if ([string]::IsNullOrWhiteSpace($name) -or [System.IO.Path]::GetExtension($name).ToLowerInvariant() -notin @('.jpg', '.jpeg', '.png')) {
    continue
  }
  $localPath = Join-Path $imageOutput $name
  Copy-RemoteFile $remotePath.Trim() $localPath
  $match = [regex]::Match($name, '^(?<capture>.+)-(?<finger>index|middle|ring|pinky)\.(?<ext>jpg|jpeg|png)$', 'IgnoreCase')
  $finger = if ($match.Success) { $match.Groups['finger'].Value.ToLowerInvariant() } else { 'unknown' }
  $captureId = if ($match.Success) { $match.Groups['capture'].Value } else { [System.IO.Path]::GetFileNameWithoutExtension($name) }
  $entries.Add([ordered]@{
      image = "raw/phone-validation/images/$name"
      mask = "masks/phone-validation/$([System.IO.Path]::GetFileNameWithoutExtension($name)).png"
      polygons = @()
      coordinate_space = 'pixel'
      person_id = 'phone-validation'
      split = 'val'
      source = 'phone-aligned-canonical'
      label_status = 'manual-polygon-required'
      capture_id = $captureId
      finger_position = $finger
      remote_path = $remotePath.Trim()
    })
  $copied += 1
  if ($copied % 25 -eq 0) { Write-Output "Telefon ROI: $copied/$($files.Count)" }
}

$json = $entries.ToArray() | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText($annotationOutput, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
Write-Output "Telefon canonical ROI kopyalandı: $copied"
Write-Output "Görseller: $imageOutput"
Write-Output "Etiket şablonu: $annotationOutput"
