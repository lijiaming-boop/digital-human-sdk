param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 18080,
    [string]$Voice = "Microsoft Huihui Desktop"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

function Write-JsonResponse {
    param(
        [System.Net.HttpListenerContext]$Context,
        [int]$StatusCode,
        [hashtable]$Body
    )

    $json = $Body | ConvertTo-Json -Depth 8 -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $Context.Response.StatusCode = $StatusCode
    $Context.Response.ContentType = "application/json; charset=utf-8"
    $Context.Response.ContentLength64 = $bytes.Length
    $Context.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    $Context.Response.OutputStream.Close()
}

$listener = [System.Net.HttpListener]::new()
$prefix = "http://${ListenHost}:${Port}/"
$listener.Prefixes.Add($prefix)

$synthesizer = [System.Speech.Synthesis.SpeechSynthesizer]::new()
$installedVoices = @($synthesizer.GetInstalledVoices() | ForEach-Object {
    $_.VoiceInfo.Name
})
if ($installedVoices.Count -eq 0) {
    throw "No Windows speech synthesis voice is installed"
}
if ($Voice -and $installedVoices -contains $Voice) {
    $synthesizer.SelectVoice($Voice)
} elseif ($Voice) {
    throw "Requested voice '$Voice' is not installed. Available: $($installedVoices -join ', ')"
}

$audioFormat = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

try {
    $listener.Start()
    Write-Host "Windows TTS service listening on $prefix"
    Write-Host "Voice: $($synthesizer.Voice.Name)"

    while ($listener.IsListening) {
        $context = $listener.GetContext()
        try {
            $path = $context.Request.Url.AbsolutePath
            if ($context.Request.HttpMethod -eq "GET" -and $path -eq "/health") {
                Write-JsonResponse -Context $context -StatusCode 200 -Body @{
                    status = "ok"
                    voice = $synthesizer.Voice.Name
                    sample_rate = 16000
                    channels = 1
                    format = "pcm_s16le"
                }
                continue
            }
            if ($context.Request.HttpMethod -ne "POST" -or $path -ne "/tts") {
                Write-JsonResponse -Context $context -StatusCode 404 -Body @{
                    error = "not found"
                }
                continue
            }

            if ($context.Request.ContentLength64 -le 0 -or
                $context.Request.ContentLength64 -gt 64KB) {
                Write-JsonResponse -Context $context -StatusCode 400 -Body @{
                    error = "invalid request body size"
                }
                continue
            }

            $reader = [System.IO.StreamReader]::new(
                $context.Request.InputStream,
                $context.Request.ContentEncoding,
                $true,
                4096,
                $false)
            try {
                $requestText = $reader.ReadToEnd()
            } finally {
                $reader.Dispose()
            }
            $request = $requestText | ConvertFrom-Json
            $text = [string]$request.text
            $sampleRate = if ($null -eq $request.sample_rate) { 16000 } else { [int]$request.sample_rate }
            $channels = if ($null -eq $request.channels) { 1 } else { [int]$request.channels }
            $format = if ($null -eq $request.format) { "pcm_s16le" } else { [string]$request.format }
            if ([string]::IsNullOrWhiteSpace($text)) {
                Write-JsonResponse -Context $context -StatusCode 400 -Body @{
                    error = "text is required"
                }
                continue
            }
            if ($text.Length -gt 2000) {
                Write-JsonResponse -Context $context -StatusCode 400 -Body @{
                    error = "text is too long"
                }
                continue
            }
            if ($sampleRate -ne 16000 -or $channels -ne 1 -or $format -ne "pcm_s16le") {
                Write-JsonResponse -Context $context -StatusCode 400 -Body @{
                    error = "only 16000 Hz mono pcm_s16le is supported"
                }
                continue
            }

            $waveStream = [System.IO.MemoryStream]::new()
            try {
                $synthesizer.SetOutputToAudioStream($waveStream, $audioFormat)
                $synthesizer.Speak($text)
                $pcm = $waveStream.ToArray()
                if ($pcm.Length -eq 0) {
                    throw "TTS generated an empty PCM stream"
                }
            } finally {
                $synthesizer.SetOutputToNull()
                $waveStream.Dispose()
            }

            $context.Response.StatusCode = 200
            $context.Response.ContentType = "application/octet-stream"
            $context.Response.Headers["X-Audio-Sample-Rate"] = "16000"
            $context.Response.Headers["X-Audio-Channels"] = "1"
            $context.Response.Headers["X-Audio-Format"] = "pcm_s16le"
            $context.Response.ContentLength64 = $pcm.Length
            $context.Response.OutputStream.Write($pcm, 0, $pcm.Length)
            $context.Response.OutputStream.Close()
        } catch {
            if ($context.Response.OutputStream.CanWrite) {
                Write-JsonResponse -Context $context -StatusCode 500 -Body @{
                    error = $_.Exception.Message
                }
            }
        }
    }
} finally {
    $synthesizer.Dispose()
    if ($listener.IsListening) { $listener.Stop() }
    $listener.Close()
}
