# NeuroTidy AWS Deployment Script (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "Deploying NeuroTidy AWS Infrastructure..." -ForegroundColor Green

# Configuration
$STACK_NAME = "neurotidy-stack"
$REGION = "us-east-1"
$TEMPLATE_FILE = "$PSScriptRoot/template.yaml"
$CONFIG_FILE = "$PSScriptRoot/../config.eng"

# 1. Parse config.eng for parameters
$PARAM_OVERRIDES = ""
if (Test-Path $CONFIG_FILE) {
    Write-Host "Reading configuration from config.eng..." -ForegroundColor Cyan
    $config_lines = Get-Content $CONFIG_FILE | Where-Object { $_ -match "^[A-Z_]+=" }
    foreach ($line in $config_lines) {
        $parts = $line -split "=", 2
        $key = $parts[0].Trim()
        $val = $parts[1].Trim()
        
        if ($key -eq "GITHUB_APP_ID" -and $val) { $PARAM_OVERRIDES += " GithubAppId=$val" }
        if ($key -eq "GITHUB_PRIVATE_KEY_PATH" -and $val) { $PARAM_OVERRIDES += " GithubPrivateKeyPath=$val" }
        if ($key -eq "GITHUB_WEBHOOK_SECRET" -and $val) { $PARAM_OVERRIDES += " GithubWebhookSecret=$val" }
        if ($key -eq "BEDROCK_MODEL_ID" -and $val) { $PARAM_OVERRIDES += " BedrockModelId=$val" }
    }
}

# 2. Check AWS CLI
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host "AWS CLI not found. Please install it first." -ForegroundColor Red
    exit 1
}

# 3. Check if SAM CLI is available
if (Get-Command sam -ErrorAction SilentlyContinue) {
    Write-Host "Using SAM CLI for deployment" -ForegroundColor Green
    
    # Build
    Write-Host "Building Lambda function..." -ForegroundColor Cyan
    sam build --template-file $TEMPLATE_FILE
    
    # Deploy
    Write-Host "Deploying stack..." -ForegroundColor Cyan
    $deploy_cmd = "sam deploy --stack-name $STACK_NAME --region $REGION --capabilities CAPABILITY_IAM --resolve-s3 --no-confirm-changeset"
    if ($PARAM_OVERRIDES) {
        $deploy_cmd += " --parameter-overrides" + $PARAM_OVERRIDES
    }
    
    Invoke-Expression $deploy_cmd
} else {
    Write-Host "SAM CLI not found. Falling back to AWS CLI (ignoring config.eng params for now)..." -ForegroundColor Yellow
    
    # Build/Package
    Write-Host "Packaging Lambda function..." -ForegroundColor Cyan
    Set-Location ../lambda
    if (Test-Path ../infrastructure/lambda.zip) { Remove-Item ../infrastructure/lambda.zip }
    Compress-Archive -Path * -DestinationPath ../infrastructure/lambda.zip -Force
    Set-Location ../infrastructure
    
    # Deploy
    Write-Host "Deploying stack..." -ForegroundColor Cyan
    aws cloudformation deploy `
        --template-file $TEMPLATE_FILE `
        --stack-name $STACK_NAME `
        --region $REGION `
        --capabilities CAPABILITY_IAM
}

# 4. Get outputs
Write-Host ""
Write-Host "Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Stack Outputs:" -ForegroundColor Cyan
aws cloudformation describe-stacks `
    --stack-name $STACK_NAME `
    --region $REGION `
    --query "Stacks[0].Outputs[*].[OutputKey,OutputValue]" `
    --output table

Write-Host ""
Write-Host "NeuroTidy is ready to use!" -ForegroundColor Green
Write-Host ""
Write-Host "To test, run:" -ForegroundColor Cyan
Write-Host "neurotidy explain sample_code.py --mode beginner" -ForegroundColor Yellow
