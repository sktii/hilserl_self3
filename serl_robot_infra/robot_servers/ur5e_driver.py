import time
import threading
import numpy as np
import rtde_receive
import rtde_control
from threading import Lock

class UR_controller():
    def __init__(self, ip="192.168.0.10"):
        self.ip = ip
        self.rtde_c = None
        self.rtde_r = None

        # 初始安全變數
        self.target_q = None
        self.lock = Lock()

        # 啟動背景控制線程
        self.running = True
        self.thread = threading.Thread(target=self._control_loop)
        self.thread.start()

        print("[Driver] Initializing...")
        while self.rtde_c is None:
            time.sleep(0.2)
        print("[Driver] Ready.")

    def servo_q(self, q):
        """ 外部呼叫此函數，只更新目標，不阻塞 """
        with self.lock:
            self.target_q = np.array(q)

    def get_joint_q(self):
        if self.rtde_r:
            return self.rtde_r.getActualQ()
        return [0]*6

    def _control_loop(self):
        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
            self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
            print(f"[Driver] Connected to {self.ip}")

            # 初始化目標為當前位置 (防止暴衝)
            current_q = self.rtde_r.getActualQ()
            with self.lock:
                self.target_q = np.array(current_q)

            dt = 0.002 # 500Hz

            while self.running:
                start_t = time.time()

                # 1. 獲取最新目標
                with self.lock:
                    cmd_q = self.target_q.copy()

                # 2. 執行 servoJ (這是不阻塞的，或者非常快)
                # lookahead_time=0.1, gain=300 是經驗參數
                self.rtde_c.servoJ(cmd_q, 0.0, 0.0, dt, 0.1, 300)

                # 3. 維持 500Hz 頻率
                diff = time.time() - start_t
                if diff < dt:
                    time.sleep(dt - diff)

        except Exception as e:
            print(f"[Driver] Error: {e}")
        finally:
            if self.rtde_c: self.rtde_c.stopScript()