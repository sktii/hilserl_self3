#!/bin/bash

# Navigate to the repository root (assuming this script is run from its location or repo root)
# We want to ensure we are in the repo root for python imports to work
# Script Location: examples_UR5e/experiments/stack_cube_sim/sim_real_humanControll.sh
# Repo Root: ../../../

# Get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$DIR/../../.."

cd "$REPO_ROOT" || exit 1

echo "================================================"
echo "   UR5e Sim-to-Real Human Control Verification"
echo "================================================"

read -p "Connect to Real Robot? (y/N): " connect_real

REAL_FLAG=""
IP_FLAG=""

if [[ "$connect_real" =~ ^[Yy]$ ]]; then
    read -p "Enter Robot Server IP [192.168.0.10]: " robot_ip
    robot_ip=${robot_ip:-192.168.0.10}

    REAL_FLAG="--real"
    IP_FLAG="--ip $robot_ip"

    echo "Starting in REAL ROBOT mode (IP: $robot_ip)..."
    echo "WARNING: Ensure the UR5e Server is running on the target machine!"
else
    echo "Starting in SIMULATION ONLY mode..."
fi

echo "Launching Python Controller..."
echo "------------------------------------------------"

# Check if PYTHONPATH needs setting
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/UR5e_sim

python3 examples_UR5e/run_human_control.py $REAL_FLAG $IP_FLAG
