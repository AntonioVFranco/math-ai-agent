#!/usr/bin/env python3
"""
Test script for the benchmark system without making API calls.

This script tests the benchmark infrastructure using mock responses
to verify all components work together correctly.
"""

import sys
import os
import json
import tempfile
from unittest.mock import Mock, patch

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

def test_answer_evaluator():
    """Test the answer evaluator functionality."""
    print("Testing Answer Evaluator...")
    
    from evaluators.answer_evaluator import evaluate_answer
    
    # Test cases
    test_cases = [
        ("The answer is 72", "#### 72", True),
        ("Final answer: 10", "#### 10", True),
        ("I got 42", "#### 42", True),
        ("No clear answer", "#### 5", False)
    ]
    
    all_passed = True
    for i, (predicted, ground_truth, expected) in enumerate(test_cases):
        is_correct, details = evaluate_answer(predicted, ground_truth)
        if is_correct == expected:
            print(f"  ✓ Test {i+1}: {predicted[:20]}... -> {is_correct}")
        else:
            print(f"  ✗ Test {i+1}: {predicted[:20]}... -> {is_correct} (expected {expected})")
            all_passed = False
    
    return all_passed

def test_dataset_loader():
    """Test the dataset loader with a temporary file."""
    print("Testing Dataset Loader...")
    
    from benchmark_runner import DatasetLoader
    
    # Create a temporary JSONL file
    test_data = [
        {"question": "What is 2 + 2?", "answer": "#### 4"},
        {"question": "What is 3 * 5?", "answer": "#### 15"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
        temp_file = f.name
    
    try:
        # Load the data
        problems = DatasetLoader.load_gsm8k_jsonl(temp_file)
        
        if len(problems) == 2:
            print("  ✓ Loaded correct number of problems")
        else:
            print(f"  ✗ Expected 2 problems, got {len(problems)}")
            return False
        
        if problems[0]['question'] == "What is 2 + 2?":
            print("  ✓ Problem content loaded correctly")
        else:
            print("  ✗ Problem content incorrect")
            return False
        
        return True
    
    finally:
        # Clean up
        os.unlink(temp_file)

def test_mock_agent_runner():
    """Test the agent runner with mocked engine."""
    print("Testing Agent Runner (mocked)...")
    
    # Mock the engine response
    mock_result = Mock()
    mock_result.success = True
    mock_result.final_answer = "The answer is 42"
    mock_result.verification_passed = True
    
    with patch('benchmark_runner.execute_solution_pipeline', return_value=mock_result):
        from benchmark_runner import AgentRunner
        
        runner = AgentRunner("fake-api-key")
        answer, latency, verified, error = runner.run_problem("What is 6 * 7?")
        
        if answer == "The answer is 42":
            print("  ✓ Agent runner returned correct answer")
        else:
            print(f"  ✗ Expected 'The answer is 42', got '{answer}'")
            return False
        
        if verified:
            print("  ✓ Verification status correct")
        else:
            print("  ✗ Verification status incorrect")
            return False
        
        if error is None:
            print("  ✓ No error reported")
        else:
            print(f"  ✗ Unexpected error: {error}")
            return False
    
    return True

def test_mock_baseline_runner():
    """Test the baseline runner with mocked OpenAI."""
    print("Testing Baseline Runner (mocked)...")
    
    # Mock OpenAI response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "The answer is 42"
    
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch('openai.OpenAI', return_value=mock_client):
        from benchmark_runner import BaselineLLMRunner
        
        runner = BaselineLLMRunner("fake-api-key")
        answer, latency, error = runner.run_problem("What is 6 * 7?")
        
        if answer == "The answer is 42":
            print("  ✓ Baseline runner returned correct answer")
        else:
            print(f"  ✗ Expected 'The answer is 42', got '{answer}'")
            return False
        
        if error is None:
            print("  ✓ No error reported")
        else:
            print(f"  ✗ Unexpected error: {error}")
            return False
    
    return True

def test_benchmark_orchestrator():
    """Test the benchmark orchestrator with mocked components."""
    print("Testing Benchmark Orchestrator (mocked)...")
    
    # Create test dataset file
    test_data = [
        {"question": "What is 2 + 2?", "answer": "#### 4"},
        {"question": "What is 3 * 5?", "answer": "#### 15"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
        dataset_file = f.name
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        output_file = f.name
    
    try:
        # Mock engine response
        mock_result = Mock()
        mock_result.success = True
        mock_result.final_answer = "The answer is 4"
        mock_result.verification_passed = True
        
        # Mock OpenAI response
        mock_openai_response = Mock()
        mock_openai_response.choices = [Mock()]
        mock_openai_response.choices[0].message.content = "The answer is 4"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        
        with patch('benchmark_runner.execute_solution_pipeline', return_value=mock_result), \
             patch('openai.OpenAI', return_value=mock_client):
            
            from benchmark_runner import BenchmarkOrchestrator
            
            orchestrator = BenchmarkOrchestrator("fake-api-key")
            df = orchestrator.run_benchmark(dataset_file, max_problems=1, output_path=output_file)
            
            if not df.empty:
                print("  ✓ DataFrame generated")
            else:
                print("  ✗ DataFrame is empty")
                return False
            
            if os.path.exists(output_file):
                print("  ✓ Output CSV file created")
                
                # Check file content
                with open(output_file, 'r') as f:
                    content = f.read()
                    if 'problem_id' in content and 'agent_answer' in content:
                        print("  ✓ CSV contains expected columns")
                    else:
                        print("  ✗ CSV missing expected columns")
                        return False
            else:
                print("  ✗ Output CSV file not created")
                return False
        
        return True
    
    finally:
        # Clean up
        for temp_file in [dataset_file, output_file]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def main():
    """Run all tests."""
    print("Testing Benchmark System Components")
    print("=" * 50)
    
    tests = [
        test_answer_evaluator,
        test_dataset_loader,
        test_mock_agent_runner,
        test_mock_baseline_runner,
        test_benchmark_orchestrator
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_func.__name__} PASSED\n")
            else:
                print(f"✗ {test_func.__name__} FAILED\n")
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Benchmark system is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())