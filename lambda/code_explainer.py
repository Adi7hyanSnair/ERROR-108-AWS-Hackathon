import json
import ast


class CodeExplainer:
    """
    Code explanation component using Amazon Bedrock.
    """
    
    def __init__(self, bedrock_client, model_id):
        self.bedrock_client = bedrock_client
        self.model_id = model_id
    
    def explain_code(self, code: str, mode: str = 'intermediate') -> str:
        """
        Explain Python code using AI with different complexity levels.
        
        Args:
            code: Python source code to explain
            mode: Explanation mode (beginner, intermediate, advanced)
        
        Returns:
            Human-friendly code explanation
        """
        # Parse code to extract structure
        code_structure = self._analyze_structure(code)
        
        # Build prompt based on mode
        prompt = self._build_prompt(code, mode, code_structure)
        
        # Call Bedrock
        explanation = self._call_bedrock(prompt)
        
        return explanation
    
    def _analyze_structure(self, code: str) -> dict:
        """
        Analyze code structure using AST.
        """
        try:
            tree = ast.parse(code)
            structure = {
                'functions': [],
                'classes': [],
                'imports': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    structure['functions'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    structure['classes'].append(node.name)
                elif isinstance(node, ast.Import):
                    structure['imports'].extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    structure['imports'].append(node.module)
            
            return structure
        except:
            return {'functions': [], 'classes': [], 'imports': []}
    
    def _build_prompt(self, code: str, mode: str, structure: dict) -> str:
        """
        Build explanation prompt based on mode.
        """
        mode_instructions = {
            'beginner': 'Explain this code in simple terms suitable for beginners. Include basic programming concepts and step-by-step logic.',
            'intermediate': 'Explain this code focusing on algorithmic logic, design patterns, and code organization.',
            'advanced': 'Explain this code emphasizing performance considerations, optimization opportunities, and advanced techniques.'
        }
        
        instruction = mode_instructions.get(mode, mode_instructions['intermediate'])
        
        prompt = f"""{instruction}

Code to explain:
```python
{code}
```

Code structure detected:
- Functions: {', '.join(structure['functions']) if structure['functions'] else 'None'}
- Classes: {', '.join(structure['classes']) if structure['classes'] else 'None'}
- Imports: {', '.join(structure['imports']) if structure['imports'] else 'None'}

Provide a clear, structured explanation including:
1. Purpose and overview
2. Key components and their roles
3. Logic flow and execution
4. Important considerations or best practices
"""
        return prompt
    
    def _call_bedrock(self, prompt: str) -> str:
        """
        Call Amazon Bedrock API for code explanation.
        """
        try:
            # Prepare request for Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Invoke Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            explanation = response_body['content'][0]['text']
            
            return explanation
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
