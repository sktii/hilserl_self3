import numpy as np
import gymnasium as gym
import ur5e_sim

# Compatibility patch for gym 0.26.2 with numpy 2.x
try:
    np.bool8 = np.bool_
except AttributeError:
    pass

def main():
    # Create the environment
    # render_mode="human" allows you to see the simulation if you have a display/GUI
    env = gym.make("UR5eStackCube-v0", render_mode="human")
    
    print("Environment initialized. Resetting...")
    obs, info = env.reset()
    
    print("Running simulation loop...")
    try:
        for i in range(1000):
            # Sample a random action: [x, y, z, gripper]
            action = env.action_space.sample()
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print status every 50 steps
            if i % 50 == 0:
                print(f"Step {i}: Reward = {reward:.3f}, Success = {info.get('succeed', False)}")
                
            if terminated or truncated:
                print("Episode finished. Resetting...")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
