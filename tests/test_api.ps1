# =============================================================================
#  NeuroTidy API Test Script (PowerShell)
#  Usage: .\test_api.ps1 -Endpoint "https://your-api-url/prod"
#  Or set NEUROTIDY_API_ENDPOINT in config.env first
# =============================================================================

param(
    [string]$Endpoint = $env:NEUROTIDY_API_ENDPOINT
)

if (-not $Endpoint) {
    # Try to read from config.env
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

Write-Host "🧪 Testing NeuroTidy API" -ForegroundColor Cyan
Write-Host "   Endpoint: $Endpoint" -ForegroundColor Gray
Write-Host ""

$Headers = @{ "Content-Type" = "application/json" }
$Passed = 0
$Failed = 0

function Test-Endpoint {
    param($Name, $Path, $Body)
    Write-Host "🔸 Test: $Name" -ForegroundColor Yellow
    try {
        $url = "$Endpoint/$Path"
        $json = $Body | ConvertTo-Json -Depth 5
        $response = Invoke-RestMethod -Uri $url -Method POST -Headers $Headers -Body $json -TimeoutSec 60
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
$r1 = Test-Endpoint -Name "Explain - Beginner mode" -Path "explain" -Body @{
    code = "def add(a, b):\n    return a + b"
    mode = "beginner"
}

# ── Test 2: Explain (Advanced) ──────────────────────────────────────────────
$r2 = Test-Endpoint -Name "Explain - Advanced mode (ML code)" -Path "explain" -Body @{
    code = "import torch`nimport torch.nn as nn`nclass Net(nn.Module):`n    def __init__(self):`n        super().__init__()`n        self.fc = nn.Linear(784, 10)`n    def forward(self, x):`n        return self.fc(x)"
    mode = "advanced"
}

# ── Test 3: Static Analysis ─────────────────────────────────────────────────
$r3 = Test-Endpoint -Name "Static Analysis" -Path "analyze" -Body @{
    code = "def process(data=[]):`n    result = []`n    for i in range(len(data)):`n        result.append(data[i] * 2)`n    return result"
    use_ai = $false
}

# ── Test 4: DL Optimizer ────────────────────────────────────────────────────
$r4 = Test-Endpoint -Name "DL Optimizer" -Path "optimize" -Body @{
    code = "import torch`nfor epoch in range(100):`n    for batch in dataloader:`n        output = model(batch.to('cuda'))`n        loss = criterion(output, labels)`n        loss.backward()`n        optimizer.step()"
    use_ai = $false
}

# ── Test 5: Bug Debugger ────────────────────────────────────────────────────
$r5 = Test-Endpoint -Name "Bug Debugger" -Path "debug" -Body @{
    error = "NameError: name 'model' is not defined"
    stack_trace = "  File 'train.py', line 15, in <module>`n    output = model(data)"
    code = "output = model(data)"
}

# ── Summary ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "Results: $Passed passed, $Failed failed" -ForegroundColor $(if ($Failed -eq 0) { "Green" } else { "Red" })

if ($Passed -gt 0 -and $r1) {
    Write-Host ""
    Write-Host "📝 Sample explanation snippet:" -ForegroundColor Cyan
    $explanation = $r1.explanation
    if ($explanation -is [string] -and $explanation.Length -gt 200) {
        Write-Host "   $($explanation.Substring(0, 200))..." -ForegroundColor Gray
    }
}

