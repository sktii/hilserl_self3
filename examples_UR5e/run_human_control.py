import argparse
import numpy as np
import gymnasium as gym
import time
import sys
import os

# Add repo root to path to ensure imports work
sys.path.append(os.getcwd())

from ur5e_sim.envs.ur5e_stack_gym_env import UR5eStackCubeGymEnv
from examples_UR5e.experiments.stack_cube_sim.config import EnvConfig, KeyBoardIntervention2

# Compatibility patch
try:
    np.bool8 = np.bool_
except AttributeError:
    pass

def main():
    parser = argparse.ArgumentParser(description="Run UR5e Environment with Human Control (Keyboard)")
    parser.add_argument("--real", action="store_true", help="Enable connection to Real Robot")
    parser.add_argument("--ip", type=str, default="192.168.0.10", help="IP address of Real Robot Server")
    parser.add_argument("--hz", type=int, default=10, help="Control Frequency")
    args = parser.parse_args()

    print("Initializing Environment...")
    if args.real:
        print(f"!!! REAL ROBOT MODE ENABLED (IP: {args.ip}) !!!")
    else:
        print("Simulation Mode Only")

    # Initialize Environment
    # Note: KeyBoardIntervention2 requires 'human' render mode to capture keys via GLFW
    env = UR5eStackCubeGymEnv(
        render_mode="human",
        image_obs=True,
        hz=args.hz,
        config=EnvConfig(),
        real_robot=args.real,
        real_robot_ip=args.ip
    )

    # Wrap with Keyboard Control
    print("Wrapping with Keyboard Intervention (WASD to move, L to toggle gripper)...")
    env = KeyBoardIntervention2(env)

    obs, info = env.reset()

    print("\nReady! Focus on the MuJoCo Viewer window to control.")
    print("Controls:")
    print("  W/S: +/- X")
    print("  A/D: +/- Y")
    print("  J/K: +/- Z")
    print("  L:   Toggle Gripper")
    print("  ; :  Toggle Intervention (Must be ON/TRUE to control)")
    print("\nPress Ctrl+C in terminal to exit.\n")

    try:
        while True:
            # We pass a dummy action. The KeyBoard wrapper will overwrite it
            # if intervention is active.
            # Env expects 4 dimensions [x, y, z, grasp].
            # If we pass 6, and intervention is OFF, it will try to pass 6 to env.step -> Crash.
            dummy_action = np.zeros(4)

            # Note: KeyBoardIntervention2.action() generates a 4D action [x,y,z,grip] if wrapper disabled gripper?
            # Let's check config.
            # EnvConfig.ACTION_SCALE is (3,).
            # UR5eStackCubeGymEnv action space is Box(4,) -> x,y,z,grasp.

            # Step
            obs, reward, terminated, truncated, info = env.step(dummy_action)

            if terminated or truncated:
                print("Resetting...")
                env.reset()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
