param(
    [string]$LlamaRoot = "E:\llama.cpp",
    [string]$ModelPath = "",
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8090,
    [int]$ContextSize = 8192,
    [int]$GpuLayers = 99,
    [int]$Parallel = 1,
    [string]$ApiKey = ""
)

$ErrorActionPreference = "Stop"
$server = Join-Path $LlamaRoot "llama-server.exe"
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $LlamaRoot "models\Qwen3-4B-Q4_K_M.gguf"
}
if (!(Test-Path -LiteralPath $server)) {
    throw "llama-server.exe not found: $server"
}
if (!(Test-Path -LiteralPath $ModelPath)) {
    throw "GGUF model not found: $ModelPath"
}

$serverArgs = @(
    "-m", $ModelPath,
    "--host", $ListenHost,
    "--port", $Port,
    "-c", $ContextSize,
    "-ngl", $GpuLayers,
    "--parallel", $Parallel
)
if (![string]::IsNullOrWhiteSpace($ApiKey)) {
    $serverArgs += @("--api-key", $ApiKey)
}

Write-Host "Starting llama.cpp: http://${ListenHost}:$Port"
Write-Host "Model: $ModelPath"
& $server @serverArgs
exit $LASTEXITCODE
