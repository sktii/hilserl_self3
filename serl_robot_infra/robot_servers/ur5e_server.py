from flask import Flask, request, jsonify
import logging
import numpy as np
import time
from absl import app, flags
import ur5e_driver
import ur5e_robotiq_gripper_driver as robotiq_gripper_driver

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", "192.168.0.10", "IP address of the UR5e robot")
flags.DEFINE_string("gripper_ip", "192.168.0.10", "IP address of the Robotiq gripper")
flags.DEFINE_integer("port", 5000, "Port to run the Flask server on")

def main(_):
    # 1. 關閉 Flask 的 Log (解決卡頓的關鍵!)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    ur5e = ur5e_driver.UR_controller(ip=FLAGS.robot_ip)

    # 嘗試連接夾爪 (失敗也不會卡住)
    gripper = robotiq_gripper_driver.RobotiqGripper()
    try:
        gripper.connect(FLAGS.gripper_ip, 63352)
        print("Gripper connected")
        gripper.reset()
    except:
        print("Gripper not connected, proceeding...")
        gripper = None

    webapp = Flask(__name__)

    # === 核心路由: 實時關節控制 ===
    @webapp.route("/servoj", methods=["POST"])
    def servoj():
        try:
            q = np.array(request.json["q"])
            ur5e.servo_q(q) # 呼叫 driver 的新方法
            return "Moved"
        except Exception as e:
            return str(e), 500

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        if gripper:
            try:
                pos = int(np.clip(request.json["gripper_pos"], 0, 255))
                # 使用 force=0 (純位置控制) 可以減少延遲
                gripper.move(position=pos, speed=255, force=50)
            except: pass
        return "Moved Gripper"

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": ur5e.get_joint_q()})

    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        return "Clear"

    print(f"Starting High-Performance UR5e Server on port {FLAGS.port}...")
    # threaded=True 允許多個請求同時處理，減少阻塞
    webapp.run(host="0.0.0.0", port=FLAGS.port, threaded=True)

if __name__ == "__main__":
    app.run(main)