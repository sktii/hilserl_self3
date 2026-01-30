#!/bin/bash
# Check if an IP address is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <Robot_IP> [Gripper_IP]"
    echo "Using default IP: 192.168.0.10"
    ROBOT_IP="192.168.0.10"
else
    ROBOT_IP="$1"
fi

if [ -z "$2" ]; then
    GRIPPER_IP="$ROBOT_IP"
else
    GRIPPER_IP="$2"
fi

echo "Launching UR5e Server for Robot at $ROBOT_IP (Gripper at $GRIPPER_IP)..."

# Assuming we are in the repo root or can find the python module
python3 serl_robot_infra/robot_servers/ur5e_server.py --robot_ip="$ROBOT_IP" --gripper_ip="$GRIPPER_IP"
