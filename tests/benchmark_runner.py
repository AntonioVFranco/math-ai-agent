"""
Comparative LLM Performance Benchmark Runner

This script evaluates the performance of our math-ai-agent against baseline LLMs
on standard mathematical datasets like GSM8K and MATH.

Measures and compares:
- Accuracy
- Latency 
- Quality of explanations

Author: MathBoardAI Agent Team
Task ID: TEST-002
"""

import json
import time
import logging
import os
import sys
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import asyncio
import traceback

# Add the src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our evaluator
from evaluators.answer_evaluator import evaluate_answer, calculate_accuracy_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Structure for storing benchmark results for a single problem."""
    problem_id: str
    problem_text: str
    ground_truth: str
    
    # Agent results
    agent_answer: str
    agent_correct: bool
    agent_latency_ms: float
    agent_verified: bool
    agent_error: Optional[str]
    
    # Baseline results
    baseline_answer: str
    baseline_correct: bool
    baseline_latency_ms: float
    baseline_error: Optional[str]
    
    # Evaluation details
    evaluation_details: Dict[str, Any]


class DatasetLoader:
    """Loads and manages benchmark datasets."""
    
    @staticmethod
    def load_gsm8k_jsonl(file_path: str, max_problems: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load GSM8K dataset from JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            max_problems: Maximum number of problems to load (None for all)
            
        Returns:
            List of problem dictionaries
        """
        problems = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_problems and i >= max_problems:
                        break
                    
                    line = line.strip()
                    if line:
                        try:
                            problem = json.loads(line)
                            # Validate required fields
                            if 'question' in problem and 'answer' in problem:
                                problems.append({
                                    'id': f"gsm8k_{i}",
                                    'question': problem['question'],
                                    'answer': problem['answer']
                                })
                            else:
                                logger.warning(f"Invalid problem format at line {i+1}")
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error at line {i+1}: {e}")
            
            logger.info(f"Loaded {len(problems)} problems from {file_path}")
            return problems
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []


class AgentRunner:
    """Runs problems through our local math-ai-agent."""
    
    def __init__(self, api_key: str):
        """
        Initialize the agent runner.
        
        Args:
            api_key: OpenAI API key for the agent
        """
        self.api_key = api_key
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the math-ai-agent engine."""
        try:
            from core.engine import execute_solution_pipeline
            self.execute_pipeline = execute_solution_pipeline
            logger.info("MathBoardAI Agent engine initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import engine: {e}")
            self.execute_pipeline = None
    
    def run_problem(self, problem_text: str) -> Tuple[str, float, bool, Optional[str]]:
        """
        Run a problem through the math-ai-agent.
        
        Args:
            problem_text: The mathematical problem to solve
            
        Returns:
            Tuple of (answer_text, latency_ms, verified, error_message)
        """
        if not self.execute_pipeline:
            return "", 0.0, False, "Engine not available"
        
        start_time = time.time()
        
        try:
            # Execute the solution pipeline
            result = self.execute_pipeline(problem_text, self.api_key)
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            if result.success:
                # Extract the answer text
                answer_text = result.final_answer if result.final_answer else ""
                
                # Check if result was verified
                verified = (result.verification_passed if hasattr(result, 'verification_passed') 
                           else False)
                
                return answer_text, latency_ms, verified, None
            else:
                error_msg = result.error_message if hasattr(result, 'error_message') else "Unknown error"
                return "", latency_ms, False, error_msg
                
        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            error_msg = f"Agent execution error: {str(e)}"
            logger.error(error_msg)
            return "", latency_ms, False, error_msg


class BaselineLLMRunner:
    """Runs problems through baseline LLMs (e.g., GPT-4)."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the baseline LLM runner.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use for baseline
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Baseline LLM client initialized with model: {self.model}")
        except ImportError:
            logger.error("OpenAI library not available")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def run_problem(self, problem_text: str) -> Tuple[str, float, Optional[str]]:
        """
        Run a problem through the baseline LLM.
        
        Args:
            problem_text: The mathematical problem to solve
            
        Returns:
            Tuple of (answer_text, latency_ms, error_message)
        """
        if not self.client:
            return "", 0.0, "OpenAI client not available"
        
        # Create a simple prompt for the baseline
        prompt = f"""Solve this mathematical problem step by step:

{problem_text}

Please show your work and provide the final numerical answer clearly."""
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves mathematical problems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            answer_text = response.choices[0].message.content
            return answer_text, latency_ms, None
            
        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            error_msg = f"Baseline LLM error: {str(e)}"
            logger.error(error_msg)
            return "", latency_ms, error_msg


class BenchmarkOrchestrator:
    """Orchestrates the complete benchmarking process."""
    
    def __init__(self, api_key: str, baseline_model: str = "gpt-4o"):
        """
        Initialize the benchmark orchestrator.
        
        Args:
            api_key: OpenAI API key
            baseline_model: Baseline model to compare against
        """
        self.api_key = api_key
        self.baseline_model = baseline_model
        self.agent_runner = AgentRunner(api_key)
        self.baseline_runner = BaselineLLMRunner(api_key, baseline_model)
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self, dataset_path: str, max_problems: Optional[int] = None,
                     output_path: str = "benchmark_report.csv") -> pd.DataFrame:
        """
        Run the complete benchmark.
        
        Args:
            dataset_path: Path to the dataset file
            max_problems: Maximum number of problems to process
            output_path: Path for the output CSV report
            
        Returns:
            DataFrame with benchmark results
        """
        logger.info("Starting benchmark run...")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Max problems: {max_problems}")
        logger.info(f"Baseline model: {self.baseline_model}")
        
        # Load dataset
        problems = DatasetLoader.load_gsm8k_jsonl(dataset_path, max_problems)
        if not problems:
            logger.error("No problems loaded. Exiting.")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(problems)} problems...")
        
        # Process each problem
        for i, problem in enumerate(problems):
            logger.info(f"Processing problem {i+1}/{len(problems)}: {problem['id']}")
            
            try:
                result = self._process_single_problem(problem)
                self.results.append(result)
                
                # Log progress
                if (i + 1) % 5 == 0:
                    logger.info(f"Completed {i+1}/{len(problems)} problems")
                    
            except Exception as e:
                logger.error(f"Error processing problem {problem['id']}: {e}")
                traceback.print_exc()
                continue
        
        # Generate report
        if self.results:
            df = self._generate_report(output_path)
            self._print_summary()
            return df
        else:
            logger.error("No results generated")
            return pd.DataFrame()
    
    def _process_single_problem(self, problem: Dict[str, Any]) -> BenchmarkResult:
        """Process a single problem through both systems."""
        problem_id = problem['id']
        problem_text = problem['question']
        ground_truth = problem['answer']
        
        logger.debug(f"Processing: {problem_text[:50]}...")
        
        # Run through agent
        agent_answer, agent_latency, agent_verified, agent_error = self.agent_runner.run_problem(problem_text)
        
        # Run through baseline
        baseline_answer, baseline_latency, baseline_error = self.baseline_runner.run_problem(problem_text)
        
        # Evaluate both answers
        agent_correct, agent_eval_details = evaluate_answer(agent_answer, ground_truth)
        baseline_correct, baseline_eval_details = evaluate_answer(baseline_answer, ground_truth)
        
        # Combine evaluation details
        evaluation_details = {
            'agent_evaluation': agent_eval_details,
            'baseline_evaluation': baseline_eval_details
        }
        
        return BenchmarkResult(
            problem_id=problem_id,
            problem_text=problem_text,
            ground_truth=ground_truth,
            agent_answer=agent_answer,
            agent_correct=agent_correct,
            agent_latency_ms=agent_latency,
            agent_verified=agent_verified,
            agent_error=agent_error,
            baseline_answer=baseline_answer,
            baseline_correct=baseline_correct,
            baseline_latency_ms=baseline_latency,
            baseline_error=baseline_error,
            evaluation_details=evaluation_details
        )
    
    def _generate_report(self, output_path: str) -> pd.DataFrame:
        """Generate CSV report from results."""
        logger.info(f"Generating report: {output_path}")
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            data.append({
                'problem_id': result.problem_id,
                'problem_text': result.problem_text,
                'ground_truth': result.ground_truth,
                'agent_answer': result.agent_answer,
                'agent_correct': result.agent_correct,
                'agent_latency_ms': result.agent_latency_ms,
                'agent_verified': result.agent_verified,
                'agent_error': result.agent_error or "",
                'baseline_answer': result.baseline_answer,
                'baseline_correct': result.baseline_correct,
                'baseline_latency_ms': result.baseline_latency_ms,
                'baseline_error': result.baseline_error or ""
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Report saved to: {output_path}")
        
        return df
    
    def _print_summary(self):
        """Print benchmark summary statistics."""
        if not self.results:
            return
        
        total_problems = len(self.results)
        
        # Calculate metrics for agent
        agent_correct_count = sum(1 for r in self.results if r.agent_correct)
        agent_accuracy = agent_correct_count / total_problems
        agent_avg_latency = sum(r.agent_latency_ms for r in self.results) / total_problems
        agent_error_count = sum(1 for r in self.results if r.agent_error)
        
        # Calculate metrics for baseline
        baseline_correct_count = sum(1 for r in self.results if r.baseline_correct)
        baseline_accuracy = baseline_correct_count / total_problems
        baseline_avg_latency = sum(r.baseline_latency_ms for r in self.results) / total_problems
        baseline_error_count = sum(1 for r in self.results if r.baseline_error)
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total Problems: {total_problems}")
        print(f"Baseline Model: {self.baseline_model}")
        print("\nACCURACY:")
        print(f"  MathBoardAI Agent: {agent_accuracy:.3f} ({agent_correct_count}/{total_problems})")
        print(f"  Baseline LLM:  {baseline_accuracy:.3f} ({baseline_correct_count}/{total_problems})")
        print("\nLATENCY (Average):")
        print(f"  MathBoardAI Agent: {agent_avg_latency:.1f} ms")
        print(f"  Baseline LLM:  {baseline_avg_latency:.1f} ms")
        print("\nERRORS:")
        print(f"  MathBoardAI Agent: {agent_error_count}")
        print(f"  Baseline LLM:  {baseline_error_count}")
        
        # Performance comparison
        accuracy_diff = agent_accuracy - baseline_accuracy
        latency_diff = agent_avg_latency - baseline_avg_latency
        
        print("\nPERFORMANCE COMPARISON:")
        accuracy_status = "BETTER" if accuracy_diff > 0.01 else "WORSE" if accuracy_diff < -0.01 else "SIMILAR"
        latency_status = "FASTER" if latency_diff < -100 else "SLOWER" if latency_diff > 100 else "SIMILAR"
        
        print(f"  Accuracy: {accuracy_status} ({accuracy_diff:+.3f})")
        print(f"  Speed: {latency_status} ({latency_diff:+.1f} ms)")
        print("="*60)


def main():
    """Main entry point for the benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comparative LLM performance benchmark")
    parser.add_argument("--dataset", type=str, default="data/benchmarks/gsm8k_sample.jsonl",
                       help="Path to dataset JSONL file")
    parser.add_argument("--api-key", type=str, required=True,
                       help="OpenAI API key")
    parser.add_argument("--max-problems", type=int, default=None,
                       help="Maximum number of problems to process")
    parser.add_argument("--baseline-model", type=str, default="gpt-4o",
                       help="Baseline model to compare against")
    parser.add_argument("--output", type=str, default="benchmark_report.csv",
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key or not args.api_key.startswith('sk-'):
        logger.error("Valid OpenAI API key is required")
        return 1
    
    # Initialize orchestrator
    orchestrator = BenchmarkOrchestrator(args.api_key, args.baseline_model)
    
    # Run benchmark
    try:
        df = orchestrator.run_benchmark(
            dataset_path=args.dataset,
            max_problems=args.max_problems,
            output_path=args.output
        )
        
        if not df.empty:
            logger.info("Benchmark completed successfully!")
            return 0
        else:
            logger.error("Benchmark failed to generate results")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())