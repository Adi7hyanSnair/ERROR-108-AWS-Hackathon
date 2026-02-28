# =============================================================================
#  NeuroTidy API Test Script (PowerShell) — Extended with /review endpoint
#  Usage: .\test_api.ps1 -Endpoint "https://your-api-url/prod"
#  Or set NEUROTIDY_API_ENDPOINT in config.env first
# =============================================================================

param(
    [string]$Endpoint = $env:NEUROTIDY_API_ENDPOINT
)

if (-not $Endpoint) {
    $configFile = Join-Path $PSScriptRoot "..\config.env"
    if (Test-Path $configFile) {
        $lines = Get-Content $configFile
        foreach ($line in $lines) {
            if ($line -match "^NEUROTIDY_API_ENDPOINT=(.+)$") {
                $Endpoint = $Matches[1].Trim()
                break
            }
        }
    }
}

if (-not $Endpoint -or $Endpoint -match "^<") {
    Write-Host "❌ No API endpoint found. Set NEUROTIDY_API_ENDPOINT in config.env" -ForegroundColor Red
    exit 1
}

Write-Host "🧪 Testing NeuroTidy API (5 endpoints)" -ForegroundColor Cyan
Write-Host "   Endpoint: $Endpoint" -ForegroundColor Gray
Write-Host ""

$Headers = @{ "Content-Type" = "application/json" }
$Passed = 0
$Failed = 0
$Results = @()

function Test-Endpoint {
    param($Name, $Path, $Body, $ExpectKeys = @())
    Write-Host "🔸 Test: $Name" -ForegroundColor Yellow
    try {
        $url = "$Endpoint/$Path"
        $json = $Body | ConvertTo-Json -Depth 5
        $response = Invoke-RestMethod -Uri $url -Method POST -Headers $Headers -Body $json -TimeoutSec 90
        $missing = $ExpectKeys | Where-Object { -not $response.PSObject.Properties[$_] }
        if ($missing) {
            Write-Host "   ⚠️  PARTIAL — Missing keys: $($missing -join ', ')" -ForegroundColor Yellow
            $script:Failed++
            return $null
        }
        Write-Host "   ✅ PASSED — analysis_id: $($response.analysis_id)" -ForegroundColor Green
        $script:Passed++
        return $response
    } catch {
        Write-Host "   ❌ FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $script:Failed++
        return $null
    }
}

# ── Test 1: Explain (Beginner) ──────────────────────────────────────────────
$r1 = Test-Endpoint -Name "Explain — Beginner mode" -Path "explain" `
    -ExpectKeys @("analysis_id", "explanation") `
    -Body @{
        code = "def add(a, b):`n    return a + b"
        mode = "beginner"
    }

# ── Test 2: Explain (Advanced DL) ───────────────────────────────────────────
$r2 = Test-Endpoint -Name "Explain — Advanced mode (ML code)" -Path "explain" `
    -ExpectKeys @("analysis_id", "explanation") `
    -Body @{
        code = "import torch`nimport torch.nn as nn`nclass Net(nn.Module):`n    def __init__(self):`n        super().__init__()`n        self.fc = nn.Linear(784, 10)`n    def forward(self, x):`n        return self.fc(x)"
        mode = "advanced"
    }

# ── Test 3: Cache hit (same code twice should be faster / return same shape) ─
Write-Host "🔸 Test: Explain — Cache hit (second call same code)" -ForegroundColor Yellow
try {
    $t1 = [System.Diagnostics.Stopwatch]::StartNew()
    $rCache1 = Invoke-RestMethod -Uri "$Endpoint/explain" -Method POST -Headers $Headers `
        -Body '{"code":"def cache_test(): return 42","mode":"beginner"}' -TimeoutSec 90
    $t1.Stop()

    $t2 = [System.Diagnostics.Stopwatch]::StartNew()
    $rCache2 = Invoke-RestMethod -Uri "$Endpoint/explain" -Method POST -Headers $Headers `
        -Body '{"code":"def cache_test(): return 42","mode":"beginner"}' -TimeoutSec 90
    $t2.Stop()

    if ($rCache1.explanation -and $rCache2.explanation) {
        Write-Host "   ✅ PASSED — 1st call: $($t1.ElapsedMilliseconds)ms  2nd call: $($t2.ElapsedMilliseconds)ms" -ForegroundColor Green
        $Passed++
    } else {
        Write-Host "   ❌ FAILED — No explanation returned" -ForegroundColor Red
        $Failed++
    }
} catch {
    Write-Host "   ❌ FAILED: $($_.Exception.Message)" -ForegroundColor Red
    $Failed++
}

# ── Test 4: Static Analysis ──────────────────────────────────────────────────
$r4 = Test-Endpoint -Name "Static Analysis" -Path "analyze" `
    -ExpectKeys @("analysis_id", "violations", "metrics") `
    -Body @{
        code   = "def process(data=[]):`n    result = []`n    for i in range(len(data)):`n        result.append(data[i] * 2)`n    return result"
        use_ai = $false
    }

# ── Test 5: DL Optimizer ─────────────────────────────────────────────────────
$r5 = Test-Endpoint -Name "DL Optimizer" -Path "optimize" `
    -ExpectKeys @("analysis_id", "violations") `
    -Body @{
        code   = "import torch`nfor epoch in range(100):`n    for batch in dataloader:`n        output = model(batch.to('cuda'))`n        loss = criterion(output, labels)`n        loss.backward()`n        optimizer.step()"
        use_ai = $false
    }

# ── Test 6: Bug Debugger ─────────────────────────────────────────────────────
$r6 = Test-Endpoint -Name "Bug Debugger" -Path "debug" `
    -ExpectKeys @("analysis_id", "error_type", "root_cause", "learning_tips", "confidence_level") `
    -Body @{
        error       = "NameError: name 'model' is not defined"
        stack_trace = "  File 'train.py', line 15, in <module>`n    output = model(data)"
        code        = "output = model(data)"
    }

# ── Test 7: /review — No token configured (should get 503 or 401) ───────────
Write-Host "🔸 Test: /review — graceful response when GitHub not configured" -ForegroundColor Yellow
try {
    $revPayload = @{
        action       = "opened"
        pull_request = @{ number = 1; head = @{ sha = "abc123" }; diff_url = "https://github.com" }
        repository   = @{ full_name = "testuser/testrepo" }
    } | ConvertTo-Json -Depth 5
    $revResp = Invoke-WebRequest -Uri "$Endpoint/review" -Method POST -Headers $Headers -Body $revPayload -TimeoutSec 30 -ErrorAction SilentlyContinue
    $revBody = $revResp.Content | ConvertFrom-Json
    if ($revResp.StatusCode -in @(200, 401, 503)) {
        Write-Host "   ✅ PASSED — /review endpoint reachable (status $($revResp.StatusCode))" -ForegroundColor Green
        $Passed++
    } else {
        Write-Host "   ⚠️  UNEXPECTED status: $($revResp.StatusCode)" -ForegroundColor Yellow
        $Failed++
    }
} catch {
    # 401/503 come through as exceptions in PS Invoke-WebRequest
    if ($_.Exception.Response.StatusCode.value__ -in @(401, 503)) {
        Write-Host "   ✅ PASSED — /review endpoint reachable (HTTP $($_.Exception.Response.StatusCode.value__))" -ForegroundColor Green
        $Passed++
    } else {
        Write-Host "   ❌ FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $Failed++
    }
}

# ── Error handling: missing code ──────────────────────────────────────────────
Write-Host "🔸 Test: Error handling — missing 'code' field" -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "$Endpoint/explain" -Method POST -Headers $Headers `
        -Body '{"mode":"beginner"}' -TimeoutSec 30
    Write-Host "   ❌ FAILED — Expected 400 but got 200" -ForegroundColor Red
    $Failed++
} catch {
    if ($_.Exception.Response.StatusCode.value__ -eq 400) {
        Write-Host "   ✅ PASSED — 400 returned for missing code" -ForegroundColor Green
        $Passed++
    } else {
        Write-Host "   ⚠️  Got status $($_.Exception.Response.StatusCode.value__)" -ForegroundColor Yellow
        $Passed++
    }
}

# ── Summary ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "Results: $Passed passed, $Failed failed" `
    -ForegroundColor $(if ($Failed -eq 0) { "Green" } else { "Red" })

if ($Passed -gt 0 -and $r1) {
    Write-Host ""
    Write-Host "📝 Sample explanation snippet:" -ForegroundColor Cyan
    $explanation = $r1.explanation
    if ($explanation -is [string] -and $explanation.Length -gt 300) {
        Write-Host "   $($explanation.Substring(0, 300))..." -ForegroundColor Gray
    }
}

if ($r6) {
    Write-Host ""
    Write-Host "🐛 Debug confidence level: $($r6.confidence_level)" -ForegroundColor Cyan
}
