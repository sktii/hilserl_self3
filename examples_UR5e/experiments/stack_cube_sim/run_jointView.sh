#!/bin/bash

# 假設 Actor 跑在 5001 (因為 Learner 佔用了 5000)
# 您可以根據實際情況修改這個數字
TARGET_PORT=5001

echo "Starting Robot Monitor Dashboard on Port $TARGET_PORT..."
echo "Waiting for Robot Server at http://127.0.0.1:$TARGET_PORT..."

# 執行 Python 腳本並帶入埠號參數
python monitor_dashboard.py --port $TARGET_PORT