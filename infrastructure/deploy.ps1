# NeuroTidy AWS Deployment Script (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Deploying NeuroTidy AWS Infrastructure..." -ForegroundColor Green

# Configuration
$STACK_NAME = "neurotidy-stack"
$REGION = "us-east-1"
$TEMPLATE_FILE = "template.yaml"

# Check AWS CLI
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host "❌ AWS CLI not found. Please install it first." -ForegroundColor Red
    exit 1
}

# Check if SAM CLI is available
if (Get-Command sam -ErrorAction SilentlyContinue) {
    Write-Host "✅ Using SAM CLI for deployment" -ForegroundColor Green
    
    # Build
    Write-Host "📦 Building Lambda function..." -ForegroundColor Cyan
    sam build --template-file $TEMPLATE_FILE
    
    # Deploy
    Write-Host "🚢 Deploying stack..." -ForegroundColor Cyan
    sam deploy `
        --stack-name $STACK_NAME `
        --region $REGION `
        --capabilities CAPABILITY_IAM `
        --resolve-s3 `
        --no-confirm-changeset
} else {
    Write-Host "⚠️  SAM CLI not found. Using CloudFormation directly..." -ForegroundColor Yellow
    
    # Package Lambda code
    Write-Host "📦 Packaging Lambda function..." -ForegroundColor Cyan
    Set-Location ../lambda
    Compress-Archive -Path * -DestinationPath ../infrastructure/lambda.zip -Force
    Set-Location ../infrastructure
    
    # Create S3 bucket for deployment artifacts
    $DEPLOY_BUCKET = "neurotidy-deploy-$(Get-Date -Format 'yyyyMMddHHmmss')"
    aws s3 mb "s3://$DEPLOY_BUCKET" --region $REGION
    
    # Upload Lambda code
    aws s3 cp lambda.zip "s3://$DEPLOY_BUCKET/lambda.zip"
    
    # Deploy CloudFormation stack
    Write-Host "🚢 Deploying stack..." -ForegroundColor Cyan
    aws cloudformation deploy `
        --template-file $TEMPLATE_FILE `
        --stack-name $STACK_NAME `
        --region $REGION `
        --capabilities CAPABILITY_IAM
    
    # Cleanup
    Remove-Item lambda.zip
}

# Get outputs
Write-Host ""
Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Stack Outputs:" -ForegroundColor Cyan
aws cloudformation describe-stacks `
    --stack-name $STACK_NAME `
    --region $REGION `
    --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' `
    --output table

Write-Host ""
Write-Host "🎉 NeuroTidy is ready to use!" -ForegroundColor Green
Write-Host ""
Write-Host "Test with:" -ForegroundColor Cyan
Write-Host 'curl -X POST <API_ENDPOINT> -H "Content-Type: application/json" -d ''{"code":"def hello():\n    print(\"Hello\")", "mode":"intermediate"}''' -ForegroundColor Yellow
