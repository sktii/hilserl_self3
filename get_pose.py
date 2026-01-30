import sys
import numpy as np
import time

# 嘗試匯入 ur_rtde
try:
    import rtde_receive
except ImportError:
    print("找不到 ur_rtde，請確認您在正確的 conda 環境中 (hilserl_self2_nocamera)")
    sys.exit(1)

ROBOT_IP = "192.168.0.3"  # 您的機械臂 IP

def main():
    print(f"正在連線到機械臂 {ROBOT_IP}...")
    try:
        r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

        print("連線成功！正在讀取數據...")
        time.sleep(0.5) # 稍微等一下確保數據穩定

        # 1. 讀取關節角度 (Joint Positions) -> 用於 Gym 環境 Home 點
        q = r.getActualQ()

        # 2. 讀取笛卡爾座標 (Cartesian TCP Pose) -> 用於 Driver 的 Center 點
        tcp = r.getActualTCPPose()

        # 格式化數據
        q_str = ", ".join([f"{x:.4f}" for x in q])
        tcp_str = ", ".join([f"{x:.4f}" for x in tcp])

        print("\n" + "="*60)
        print("🎉 讀取成功！請分別複製以下兩段程式碼：")
        print("="*60)

        print("\n👇 [第一部分] 給 Client 端 (human_control.py) 使用：")
        print("-" * 50)
        print(f"UR5eStackCubeGymEnv._UR5E_HOME = np.asarray([{q_str}])")
        print("-" * 50)

        print("\n👇 [第二部分] 給 Server 端 (driver.py) 使用：")
        print("請找到 CONFIG 字典裡的 'center' 並替換成這行：")
        print("-" * 50)
        print(f'"center": [{tcp_str}],')
        print("-" * 50)

        print("\n✅ 完成後，兩邊的起始點將完美同步！")
        print("="*60 + "\n")

    except Exception as e:
        print(f"連線失敗: {e}")
        print("請確認機械臂已開機且為 Remote Control 模式")

if __name__ == "__main__":
    main()