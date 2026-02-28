"""
RAG-based DL Optimizer using Amazon Bedrock Knowledge Base
Combines static analysis with intelligent rule retrieval and LLM reasoning
"""

import ast
import json
import re
from typing import List, Dict, Optional


class RAGOptimizer:
    """
    RAG-based optimizer that:
    1. Analyzes code structure (AST)
    2. Retrieves relevant rules from knowledge base
    3. Uses LLM to apply rules with skill-level awareness
    """

    def __init__(self, bedrock_client, bedrock_agent_client, model_id: str, knowledge_base_id: str = None):
        """
        Args:
            bedrock_client: Bedrock runtime client for LLM
            bedrock_agent_client: Bedrock agent client for knowledge base
            model_id: Bedrock model ID
            knowledge_base_id: Knowledge base ID (optional, falls back to local)
        """
        self.bedrock_client = bedrock_client
        self.bedrock_agent_client = bedrock_agent_client
        self.model_id = model_id
        self.knowledge_base_id = knowledge_base_id
        
        # Load local rules as fallback
        self.local_rules = self._load_local_rules()

    def analyze(self, code: str, mode: str = "intermediate", use_ai: bool = True) -> dict:
        """
        Analyze code with RAG approach
        
        Args:
            code: Python source code
            mode: Skill level (beginner, intermediate, expert)
            use_ai: Whether to use LLM (if False, only static analysis)
        
        Returns:
            dict with violations, optimizations, and insights
        """
        # Step 1: Extract code features using AST
        features = self._extract_code_features(code)
        
        # Step 2: Run quick static checks for critical errors
        critical_errors = self._detect_critical_errors(code, features)
        
        if not use_ai:
            # Fast mode: only return critical errors
            return {
                "violations": critical_errors,
                "mode": mode,
                "analysis_type": "static_only",
                "features_detected": features,
                "summary": f"Found {len(critical_errors)} critical issues (static analysis only)"
            }
        
        # Step 3: Retrieve relevant rules from knowledge base
        relevant_rules = self._retrieve_relevant_rules(features, mode)
        
        # Step 4: Use LLM to analyze with retrieved rules
        llm_analysis = self._analyze_with_llm(code, relevant_rules, features, mode)
        
        # Step 5: Combine results
        all_violations = critical_errors + llm_analysis.get('violations', [])
        
        # Remove duplicates
        all_violations = self._deduplicate_violations(all_violations)
        
        return {
            "violations": all_violations,
            "mode": mode,
            "analysis_type": "rag_enhanced",
            "features_detected": features,
            "rules_consulted": [r['rule_id'] for r in relevant_rules],
            "summary": self._build_summary(all_violations, mode),
            "skill_level_insights": llm_analysis.get('skill_insights', {})
        }

    def _extract_code_features(self, code: str) -> Dict:
        """Extract features from code using AST"""
        features = {
            "has_training_loop": False,
            "has_backward": False,
            "has_optimizer_step": False,
            "has_zero_grad": False,
            "has_dataloader": False,
            "has_cuda": False,
            "has_lstm_gru_rnn": False,
            "has_gradient_clipping": False,
            "has_mixed_precision": False,
            "has_model_train": False,
            "has_validation": False,
            "has_seed": False,
            "model_type": None,
            "complexity": "simple"
        }
        
        # Text-based feature detection
        features['has_backward'] = 'backward()' in code
        features['has_optimizer_step'] = 'optimizer.step()' in code or '.step()' in code
        features['has_zero_grad'] = 'zero_grad()' in code
        features['has_dataloader'] = 'DataLoader' in code
        features['has_cuda'] = 'cuda' in code or '.to(device)' in code
        features['has_lstm_gru_rnn'] = any(x in code for x in ['LSTM', 'GRU', 'RNN'])
        features['has_gradient_clipping'] = 'clip_grad' in code
        features['has_mixed_precision'] = 'autocast' in code or 'GradScaler' in code
        features['has_model_train'] = 'model.train()' in code or '.train()' in code
        features['has_validation'] = any(x in code for x in ['valid', 'val_', 'validation'])
        features['has_seed'] = any(x in code for x in ['manual_seed', 'random.seed', 'np.random.seed'])
        
        # Detect training loop
        features['has_training_loop'] = ('for epoch' in code or 'for batch' in code) and features['has_backward']
        
        # Determine complexity
        if features['has_lstm_gru_rnn'] or code.count('nn.') > 5:
            features['complexity'] = 'complex'
        elif code.count('nn.') > 2:
            features['complexity'] = 'moderate'
        
        return features

    def _detect_critical_errors(self, code: str, features: Dict) -> List[dict]:
        """Fast static detection of critical errors"""
        errors = []
        
        # NT007: Missing zero_grad
        if features['has_training_loop'] and features['has_backward'] and not features['has_zero_grad']:
            errors.append({
                "rule_id": "NT007",
                "severity": "HIGH",
                "category": "training-loop",
                "description": "Missing optimizer.zero_grad() in training loop",
                "line_number": 0,
                "source": "static_analysis"
            })
        
        # NT013: Missing gradient clipping for RNN
        if features['has_lstm_gru_rnn'] and features['has_training_loop'] and not features['has_gradient_clipping']:
            errors.append({
                "rule_id": "NT013",
                "severity": "MEDIUM",
                "category": "training-stability",
                "description": "Missing gradient clipping for RNN/LSTM",
                "line_number": 0,
                "source": "static_analysis"
            })
        
        # NT015: Missing model.train()
        if features['has_training_loop'] and not features['has_model_train']:
            errors.append({
                "rule_id": "NT015",
                "severity": "MEDIUM",
                "category": "training-correctness",
                "description": "Model not set to training mode (missing model.train())",
                "line_number": 0,
                "source": "static_analysis"
            })
        
        return errors

    def _retrieve_relevant_rules(self, features: Dict, mode: str) -> List[dict]:
        """Retrieve relevant rules based on code features"""
        
        # Build query based on features
        query_terms = []
        if features['has_training_loop']:
            query_terms.append("training loop")
        if features['has_lstm_gru_rnn']:
            query_terms.append("LSTM RNN gradient")
        if features['has_cuda']:
            query_terms.append("GPU CUDA performance")
        if features['has_dataloader']:
            query_terms.append("DataLoader optimization")
        
        query = " ".join(query_terms) if query_terms else "deep learning optimization"
        
        # Try to retrieve from Bedrock Knowledge Base
        if self.knowledge_base_id and self.bedrock_agent_client:
            try:
                relevant_rules = self._query_knowledge_base(query, mode)
                if relevant_rules:
                    return relevant_rules
            except Exception as e:
                print(f"Knowledge base query failed: {e}, falling back to local rules")
        
        # Fallback: Filter local rules based on features
        return self._filter_local_rules(features, mode)

    def _query_knowledge_base(self, query: str, mode: str) -> List[dict]:
        """Query Bedrock Knowledge Base for relevant rules"""
        try:
            response = self.bedrock_agent_client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': query},
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': 5
                    }
                }
            )
            
            # Extract rules from retrieval results
            rules = []
            for result in response.get('retrievalResults', []):
                content = result.get('content', {}).get('text', '')
                try:
                    rule = json.loads(content)
                    rules.append(rule)
                except:
                    pass
            
            return rules
        except Exception as e:
            print(f"Knowledge base retrieval error: {e}")
            return []

    def _filter_local_rules(self, features: Dict, mode: str) -> List[dict]:
        """Filter local rules based on code features"""
        relevant = []
        
        for rule in self.local_rules:
            # Check if rule's code features match detected features
            rule_features = rule.get('code_features', [])
            
            # Simple matching logic
            if 'training_loop' in rule_features and features['has_training_loop']:
                relevant.append(rule)
            elif 'lstm' in rule_features and features['has_lstm_gru_rnn']:
                relevant.append(rule)
            elif 'cuda' in rule_features and features['has_cuda']:
                relevant.append(rule)
            elif 'dataloader' in rule_features and features['has_dataloader']:
                relevant.append(rule)
        
        # Limit to top 5 most relevant
        return relevant[:5]

    def _load_local_rules(self) -> List[dict]:
        """Load rules from local JSON file"""
        try:
            import os
            rules_path = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base', 'rules_database.json')
            with open(rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('rules', [])
        except Exception as e:
            print(f"Failed to load local rules: {e}")
            return []

    def _analyze_with_llm(self, code: str, rules: List[dict], features: Dict, mode: str) -> dict:
        """Use LLM to analyze code with retrieved rules"""
        
        # Build context from retrieved rules
        rules_context = self._build_rules_context(rules, mode)
        
        # Build skill-appropriate prompt
        prompt = self._build_analysis_prompt(code, rules_context, features, mode)
        
        # Call Bedrock
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            text = json.loads(response['body'].read())['content'][0]['text']
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            print(f"LLM analysis error: {e}")
        
        return {"violations": [], "skill_insights": {}}

    def _build_rules_context(self, rules: List[dict], mode: str) -> str:
        """Build context string from retrieved rules"""
        context_parts = []
        
        for rule in rules:
            skill_advice = rule.get('skill_level_advice', {}).get(mode, rule.get('description', ''))
            
            context_parts.append(f"""
[{rule['rule_id']}] {rule['name']}
Severity: {rule['severity']}
Description: {rule['description']}
Why it matters: {rule.get('why_it_matters', '')}
{mode.capitalize()} advice: {skill_advice}
Good example:
{rule.get('good_example', '')}
""")
        
        return "\n".join(context_parts)

    def _build_analysis_prompt(self, code: str, rules_context: str, features: Dict, mode: str) -> str:
        """Build skill-appropriate analysis prompt"""
        
        mode_instructions = {
            "beginner": """You are a friendly ML teacher helping a beginner learn PyTorch.
Focus on:
- Simple, easy-to-implement suggestions
- Educational explanations
- Safe, low-risk optimizations
- Hyperparameter tuning basics
Avoid: Complex architectures, advanced techniques, risky changes""",
            
            "intermediate": """You are an ML engineer mentor helping an intermediate developer.
Focus on:
- Architectural improvements
- Training best practices
- Learning rate scheduling
- Data augmentation
- Regularization techniques
Provide: Practical suggestions with moderate complexity""",
            
            "expert": """You are a performance optimization expert helping an experienced ML engineer.
Focus on:
- Maximum performance gains
- Low-level optimizations
- Distributed training
- Memory efficiency
- Profiling and benchmarking
- Cutting-edge techniques
Provide: Advanced optimizations with performance metrics"""
        }
        
        instruction = mode_instructions.get(mode, mode_instructions['intermediate'])
        
        prompt = f"""{instruction}

Analyze this PyTorch code for a {mode.upper()} level user.

Code features detected:
{json.dumps(features, indent=2)}

Relevant optimization rules to check:
{rules_context}

Code to analyze:
```python
{code}
```

Return JSON with this structure:
{{
  "violations": [
    {{
      "rule_id": "NT0XX",
      "severity": "HIGH|MEDIUM|LOW",
      "description": "Clear description",
      "explanation": "Why this matters for {mode} level",
      "suggested_fix": "Specific code or approach",
      "skill_level_note": "Additional context for {mode} users",
      "performance_impact": "Expected improvement"
    }}
  ],
  "skill_insights": {{
    "top_priority": "Most important thing to fix",
    "quick_wins": ["Easy improvements"],
    "learning_opportunities": ["Concepts to study"]
  }}
}}

Only report violations that are actually present in the code. Use the provided rules as reference."""
        
        return prompt

    def _deduplicate_violations(self, violations: List[dict]) -> List[dict]:
        """Remove duplicate violations"""
        seen = set()
        unique = []
        
        for v in violations:
            key = v.get('rule_id', '') + v.get('description', '')
            if key not in seen:
                seen.add(key)
                unique.append(v)
        
        return unique

    def _build_summary(self, violations: List[dict], mode: str) -> str:
        """Build human-readable summary"""
        if not violations:
            return f"✅ Code looks good for {mode} level! No major issues found."
        
        high = sum(1 for v in violations if v.get('severity') == 'HIGH')
        medium = sum(1 for v in violations if v.get('severity') == 'MEDIUM')
        low = sum(1 for v in violations if v.get('severity') == 'LOW')
        
        return f"Found {len(violations)} optimization opportunities for {mode} level: {high} high priority, {medium} medium, {low} low."
