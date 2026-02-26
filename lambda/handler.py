import json
import os
import uuid
from datetime import datetime
import boto3
from code_explainer import CodeExplainer

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

# Environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', 'neurotidy-results')
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'neurotidy-cache')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')

table = dynamodb.Table(DYNAMODB_TABLE)


def lambda_handler(event, context):
    """
    Main Lambda handler for code analysis requests.
    """
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))
        code = body.get('code', '')
        mode = body.get('mode', 'intermediate')
        
        if not code:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Code is required'})
            }
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Initialize code explainer
        explainer = CodeExplainer(bedrock_client, BEDROCK_MODEL_ID)
        
        # Perform code explanation
        explanation = explainer.explain_code(code, mode)
        
        # Prepare result
        result = {
            'analysis_id': analysis_id,
            'code': code,
            'mode': mode,
            'explanation': explanation,
            'timestamp': timestamp
        }
        
        # Store in S3
        s3_key = f"analyses/{datetime.utcnow().strftime('%Y/%m/%d')}/{analysis_id}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(result, indent=2),
            ContentType='application/json'
        )
        
        # Cache metadata in DynamoDB
        table.put_item(
            Item={
                'analysis_id': analysis_id,
                'timestamp': timestamp,
                's3_location': f"s3://{S3_BUCKET}/{s3_key}",
                'mode': mode,
                'code_length': len(code),
                'ttl': int(datetime.utcnow().timestamp()) + 86400  # 24 hour TTL
            }
        )
        
        # Return response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'analysis_id': analysis_id,
                'explanation': explanation,
                's3_location': f"s3://{S3_BUCKET}/{s3_key}",
                'timestamp': timestamp
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
