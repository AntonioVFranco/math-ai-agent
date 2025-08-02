#!/bin/bash

# Comparative LLM Performance Benchmark Runner
# 
# This script runs the complete benchmark suite to evaluate our math-ai-agent
# against baseline LLMs on mathematical problem datasets.
#
# Author: Math AI Agent Team
# Task ID: TEST-002

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATASET_PATH="data/benchmarks/gsm8k_sample.jsonl"
MAX_PROBLEMS=""
BASELINE_MODEL="gpt-4o"
OUTPUT_PATH="benchmark_report.csv"
API_KEY=""

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -k, --api-key KEY        OpenAI API key (required)"
    echo "  -d, --dataset PATH       Path to dataset JSONL file (default: $DATASET_PATH)"
    echo "  -n, --max-problems NUM   Maximum number of problems to process (default: all)"
    echo "  -m, --model MODEL        Baseline model to compare against (default: $BASELINE_MODEL)"
    echo "  -o, --output PATH        Output CSV file path (default: $OUTPUT_PATH)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY          OpenAI API key (alternative to -k)"
    echo ""
    echo "Examples:"
    echo "  $0 -k sk-your-api-key-here"
    echo "  $0 -k sk-your-key -n 10 -m gpt-4"
    echo "  OPENAI_API_KEY=sk-your-key $0 -n 5"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -n|--max-problems)
            MAX_PROBLEMS="$2"
            shift 2
            ;;
        -m|--model)
            BASELINE_MODEL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check for API key
if [[ -z "$API_KEY" ]]; then
    if [[ -n "$OPENAI_API_KEY" ]]; then
        API_KEY="$OPENAI_API_KEY"
    else
        print_error "OpenAI API key is required. Use -k option or set OPENAI_API_KEY environment variable."
        show_usage
        exit 1
    fi
fi

# Validate API key format
if [[ ! "$API_KEY" =~ ^sk- ]]; then
    print_error "Invalid API key format. OpenAI API keys should start with 'sk-'"
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_info "Starting Comparative LLM Performance Benchmark"
print_info "=============================================="
print_info "Project Root: $PROJECT_ROOT"
print_info "Dataset: $DATASET_PATH"
print_info "Baseline Model: $BASELINE_MODEL"
print_info "Output: $OUTPUT_PATH"

if [[ -n "$MAX_PROBLEMS" ]]; then
    print_info "Max Problems: $MAX_PROBLEMS"
else
    print_info "Max Problems: All"
fi

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if dataset exists
if [[ ! -f "$DATASET_PATH" ]]; then
    print_error "Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Check if required Python packages are available
print_info "Checking dependencies..."
python3 -c "import pandas, openai" 2>/dev/null || {
    print_error "Required Python packages not available. Please install: pandas openai"
    print_info "Run: pip install pandas openai"
    exit 1
}

# Check if our modules are importable
python3 -c "
import sys
import os
sys.path.insert(0, os.path.join('src'))
try:
    from core.engine import execute_solution_pipeline
    print('✓ Math AI Agent engine available')
except ImportError as e:
    print(f'✗ Math AI Agent engine not available: {e}')
    sys.exit(1)
" || {
    print_error "Math AI Agent engine not available"
    exit 1
}

# Build the Python command
PYTHON_CMD="python3 tests/benchmark_runner.py --api-key '$API_KEY' --dataset '$DATASET_PATH' --baseline-model '$BASELINE_MODEL' --output '$OUTPUT_PATH'"

if [[ -n "$MAX_PROBLEMS" ]]; then
    PYTHON_CMD="$PYTHON_CMD --max-problems $MAX_PROBLEMS"
fi

print_info "Running benchmark..."
print_info "Command: $PYTHON_CMD"

# Create output directory if it doesn't exist
OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
if [[ "$OUTPUT_DIR" != "." && ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    print_info "Created output directory: $OUTPUT_DIR"
fi

# Run the benchmark
if eval "$PYTHON_CMD"; then
    print_success "Benchmark completed successfully!"
    
    # Check if output file was created
    if [[ -f "$OUTPUT_PATH" ]]; then
        print_success "Report generated: $OUTPUT_PATH"
        
        # Show some basic stats about the report
        if command -v wc &> /dev/null; then
            LINES=$(wc -l < "$OUTPUT_PATH")
            print_info "Report contains $((LINES - 1)) problem results"
        fi
        
        # Show the first few lines of the report
        print_info "Report preview:"
        head -3 "$OUTPUT_PATH" 2>/dev/null || true
        
    else
        print_warning "Report file was not generated: $OUTPUT_PATH"
    fi
    
    exit 0
else
    print_error "Benchmark failed!"
    exit 1
fi