import requests
import json
import sys
import time
from eval_dataset import EVAL_DATASET

import os

API_BASE = os.environ.get("API_BASE_URL", "https://1d21iee6x0.execute-api.us-east-1.amazonaws.com/prod")

def run_eval_case(case):
    endpoint = case["endpoint"]
    url = f"{API_BASE}{endpoint}"
    
    payload = {"code": case["code"], "mode": "intermediate"}
    if "error_message" in case:
        payload["error"] = case["error_message"]
        
    print(f"\\nRunning Eval: [{case['id']}] -> {endpoint}")
    print(f"Description: {case['description']}")
    
    try:
        start_time = time.time()
        r = requests.post(url, json=payload, timeout=45)
        duration = time.time() - start_time
        
        if r.status_code != 200:
            print(f"  ❌ API Error: HTTP {r.status_code} - {r.text[:100]}")
            return False, "API Error"
            
        data = r.json()
        
        # Combine text responses from different endpoints to look for expected findings
        response_text = ""
        if "explanation" in data:
            response_text += str(data["explanation"])
        if "summary" in data:
            response_text += str(data["summary"])
        if "violations" in data:
            response_text += json.dumps(data["violations"])
            
        response_text = response_text.lower()
        
        # Check if any of the expected findings are in the response
        findings = [f.lower() for f in case["expected_findings"]]
        passed = any(f in response_text for f in findings)
        
        if passed:
            print(f"  ✅ PASS ({duration:.2f}s) - Found expected pattern")
            return True, None
        else:
            print(f"  ❌ FAIL ({duration:.2f}s) - Did not find any of: {findings}")
            return False, f"Missing findings"
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ FAIL: Request exception -> {e}")
        return False, str(e)

def main():
    print("=" * 60)
    print("NeuroTidy Model Evaluation Suite")
    print("=" * 60)
    
    total = len(EVAL_DATASET)
    passed = 0
    failed = []
    
    for case in EVAL_DATASET:
        success, reason = run_eval_case(case)
        if success:
            passed += 1
        else:
            failed.append({
                "id": case["id"],
                "reason": reason
            })
            
    score = (passed / total) * 100
    
    print("\\n" + "=" * 60)
    print(f"EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Test Cases: {total}")
    print(f"Passed:           {passed}")
    print(f"Failed:           {total - passed}")
    print(f"Accuracy Score:   {score:.1f}%")
    
    if failed:
        print("\\nFailed Cases:")
        for f in failed:
            print(f"  - {f['id']}: {f['reason']}")
            
    if score == 100:
        print("\\n🏆 PERFECT SCORE! The model is highly accurate on this dataset.")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
