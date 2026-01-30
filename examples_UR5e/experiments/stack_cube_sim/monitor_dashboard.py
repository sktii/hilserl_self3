import tkinter as tk
from tkinter import ttk
import requests
import time
import threading
import argparse
import queue

# 解析命令列參數
parser = argparse.ArgumentParser(description="Robot Monitor Dashboard")
parser.add_argument("--port", type=int, default=5000, help="Port of the robot server (default: 5000)")
args = parser.parse_args()

# 預設嘗試的埠號列表。優先嘗試 5001 (Actor)，然後是 5000 (Learner)
TARGET_PORTS = [args.port, 5001, 5000]
TARGET_PORTS = list(dict.fromkeys(TARGET_PORTS))

class RobotMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hil-Serl Robot Monitor (Threaded)")
        self.root.geometry("400x380")
        self.root.configure(bg="#f0f0f0")

        self.session = requests.Session()
        
        # 用於執行緒間溝通的佇列 (只存最新的數據)
        self.data_queue = queue.Queue(maxsize=1)
        
        # 控制變數
        self.running = True
        self.connected_port = None
        self.current_port_index = 0
        self.server_status = "Disconnected"
        self.server_color = "gray"

        # --- 介面佈局 ---
        title_label = tk.Label(root, text="Robot State Monitor", font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)

        self.frame = tk.Frame(root, bg="#f0f0f0")
        self.frame.pack(padx=20, pady=5)

        self.labels = {}
        self.vars_to_monitor = [
            ("X", "0.000"), ("Y", "0.000"), ("Z", "0.000"),
            ("RX (qx)", "0.000"), ("RY (qy)", "0.000"), ("RZ (qz)", "0.000"), ("RW (qw)", "0.000"),
            ("Gripper", "0.000")
        ]

        for name, init_val in self.vars_to_monitor:
            row = tk.Frame(self.frame, bg="#f0f0f0")
            row.pack(fill="x", pady=2)
            lbl_name = tk.Label(row, text=f"{name}:", width=10, anchor="w", font=("Consolas", 12), bg="#f0f0f0")
            lbl_name.pack(side="left")
            lbl_val = tk.Label(row, text=init_val, width=15, anchor="e", font=("Consolas", 12, "bold"), bg="white", fg="blue")
            lbl_val.pack(side="right")
            self.labels[name] = lbl_val

        self.status_label = tk.Label(root, text="Starting...", fg="gray", bg="#f0f0f0", font=("Arial", 10))
        self.status_label.pack(side="bottom", pady=5)

        # --- 啟動背景執行緒 ---
        self.monitor_thread = threading.Thread(target=self.background_poller, daemon=True)
        self.monitor_thread.start()

        # --- 啟動介面更新迴圈 ---
        self.update_ui()

    def background_poller(self):
        """這是背景執行緒：專門負責網路請求，即使卡住也不會影響介面"""
        while self.running:
            # 決定要連哪一個 Port
            target_port = self.connected_port if self.connected_port else TARGET_PORTS[self.current_port_index]
            server_url = f"http://127.0.0.1:{target_port}"

            try:
                # 發送請求 (這裡可以稍微久一點，沒關係)
                start_ts = time.time()
                response = self.session.post(f"{server_url}/getstate", json={}, timeout=0.5)
                
                if response.status_code == 200:
                    self.connected_port = target_port # 鎖定 Port
                    data = response.json()
                    
                    # 計算延遲 (Latency)
                    latency = (time.time() - start_ts) * 1000 
                    
                    # 將數據放入佇列供主執行緒讀取
                    package = {
                        "pose": data.get("pose", [0]*7),
                        "gripper": data.get("gripper_pos", 0),
                        "status": f"● Connected Port {target_port} ({latency:.0f}ms)",
                        "color": "green"
                    }
                    
                    # 如果佇列滿了，先清空舊的，確保只顯示最新的
                    if self.data_queue.full():
                        try: self.data_queue.get_nowait()
                        except: pass
                    self.data_queue.put(package)
                    
                    # 休息一下，避免過度佔用 CPU (20Hz)
                    time.sleep(0.05)
                else:
                    self._report_error(f"Error: {response.status_code}")

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if self.connected_port:
                    self._report_error("Disconnected (Retrying...)")
                    self.connected_port = None # 斷線重連
                else:
                    self._report_error(f"Searching Port {target_port}...")
                    self.current_port_index = (self.current_port_index + 1) % len(TARGET_PORTS)
                time.sleep(0.5) # 失敗時休息久一點
            
            except Exception as e:
                self._report_error(f"Error: {str(e)}")
                time.sleep(1.0)

    def _report_error(self, msg):
        package = {
            "pose": None,
            "status": msg,
            "color": "red" if "Error" in msg else "orange"
        }
        if self.data_queue.full():
            try: self.data_queue.get_nowait()
            except: pass
        self.data_queue.put(package)

    def update_ui(self):
        """這是主執行緒：只負責從佇列拿數據更新畫面"""
        try:
            # 嘗試從佇列獲取數據，不等待 (Non-blocking)
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                
                # 更新狀態列
                self.status_label.config(text=data["status"], fg=data["color"])

                # 如果有 pose 數據才更新數值
                if data.get("pose"):
                    pose = data["pose"]
                    gripper = data["gripper"]

                    self.labels["X"].config(text=f"{pose[0]:.4f}")
                    self.labels["Y"].config(text=f"{pose[1]:.4f}")
                    self.labels["Z"].config(text=f"{pose[2]:.4f}")
                    self.labels["RX (qx)"].config(text=f"{pose[3]:.4f}")
                    self.labels["RY (qy)"].config(text=f"{pose[4]:.4f}")
                    self.labels["RZ (qz)"].config(text=f"{pose[5]:.4f}")
                    self.labels["RW (qw)"].config(text=f"{pose[6]:.4f}")

                    g_text = f"{gripper:.3f}"
                    if gripper > 0.8: g_text += " (OPEN)"
                    elif gripper < 0.1: g_text += " (CLOSED)"
                    self.labels["Gripper"].config(text=g_text)

        except queue.Empty:
            pass
        
        # 保持介面流暢，每 30ms 檢查一次
        if self.running:
            self.root.after(30, self.update_ui)

    def on_closing(self):
        self.running = False
        self.session.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()