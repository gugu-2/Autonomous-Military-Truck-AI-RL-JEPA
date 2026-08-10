#!/bin/bash

# Default values
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_SIMULATION=false
RUN_BENCHMARK=false

# Parse arguments
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [--unit] [--integration] [--simulation] [--benchmark] [--all]"
    exit 1
fi

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --unit) RUN_UNIT=true ;;
        --integration) RUN_INTEGRATION=true ;;
        --simulation) RUN_SIMULATION=true ;;
        --benchmark) RUN_BENCHMARK=true ;;
        --all)
            RUN_UNIT=true
            RUN_INTEGRATION=true
            RUN_SIMULATION=true
            RUN_BENCHMARK=true
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Setup test environment
echo "Setting up test environment..."
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export ROS_DOMAIN_ID=42

# Ensure pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Please install dependencies in the venv."
    exit 1
fi

EXIT_CODE=0
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Run requested tests
if [ "$RUN_UNIT" = true ]; then
    echo "==================================="
    echo "Running Unit Tests..."
    echo "==================================="
    pytest tests/unit/ --cov=src --cov-report=term-missing
    if [ $? -ne 0 ]; then EXIT_CODE=1; fi
fi

if [ "$RUN_INTEGRATION" = true ]; then
    echo "==================================="
    echo "Running Integration Tests..."
    echo "==================================="
    pytest tests/integration/
    if [ $? -ne 0 ]; then EXIT_CODE=1; fi
fi

if [ "$RUN_SIMULATION" = true ]; then
    echo "==================================="
    echo "Running Simulation Tests..."
    echo "==================================="
    pytest tests/simulation/
    if [ $? -ne 0 ]; then EXIT_CODE=1; fi
fi

if [ "$RUN_BENCHMARK" = true ]; then
    echo "==================================="
    echo "Running Benchmark Tests..."
    echo "==================================="
    pytest tests/benchmark/
    if [ $? -ne 0 ]; then EXIT_CODE=1; fi
fi

echo "==================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}All requested tests passed successfully!${NC}"
else
    echo -e "${RED}Some tests failed. Check the output above for details.${NC}"
fi
echo "==================================="

exit $EXIT_CODE
