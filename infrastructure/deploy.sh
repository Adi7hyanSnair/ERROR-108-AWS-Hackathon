#!/bin/bash

# NeuroTidy AWS Deployment Script

set -e

echo "🚀 Deploying NeuroTidy AWS Infrastructure..."

# Configuration
STACK_NAME="neurotidy-stack"
REGION="us-east-1"
TEMPLATE_FILE="template.yaml"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first."
    exit 1
fi

# Check if SAM CLI is available
if command -v sam &> /dev/null; then
    echo "✅ Using SAM CLI for deployment"
    
    # Build
    echo "📦 Building Lambda function..."
    sam build --template-file $TEMPLATE_FILE
    
    # Deploy
    echo "🚢 Deploying stack..."
    sam deploy \
        --stack-name $STACK_NAME \
        --region $REGION \
        --capabilities CAPABILITY_IAM \
        --resolve-s3 \
        --no-confirm-changeset
else
    echo "⚠️  SAM CLI not found. Using CloudFormation directly..."
    
    # Package Lambda code
    echo "📦 Packaging Lambda function..."
    cd ../lambda
    zip -r ../infrastructure/lambda.zip . -x "*.pyc" -x "__pycache__/*"
    cd ../infrastructure
    
    # Create S3 bucket for deployment artifacts
    DEPLOY_BUCKET="neurotidy-deploy-$(date +%s)"
    aws s3 mb s3://$DEPLOY_BUCKET --region $REGION
    
    # Upload Lambda code
    aws s3 cp lambda.zip s3://$DEPLOY_BUCKET/lambda.zip
    
    # Deploy CloudFormation stack
    echo "🚢 Deploying stack..."
    aws cloudformation deploy \
        --template-file $TEMPLATE_FILE \
        --stack-name $STACK_NAME \
        --region $REGION \
        --capabilities CAPABILITY_IAM \
        --parameter-overrides \
            LambdaCodeBucket=$DEPLOY_BUCKET \
            LambdaCodeKey=lambda.zip
    
    # Cleanup
    rm lambda.zip
fi

# Get outputs
echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Stack Outputs:"
aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
    --output table

echo ""
echo "🎉 NeuroTidy is ready to use!"
echo ""
echo "Test with:"
echo 'curl -X POST <API_ENDPOINT> -H "Content-Type: application/json" -d '"'"'{"code":"def hello():\n    print(\"Hello\")", "mode":"intermediate"}'"'"''
