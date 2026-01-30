import gymnasium as gym
from ur5e_sim.envs.ur5e_stack_gym_env import UR5eStackCubeGymEnv

gym.register(
    id="UR5eStackCube-v0",
    entry_point="ur5e_sim.envs.ur5e_stack_gym_env:UR5eStackCubeGymEnv",
    max_episode_steps=500,
)
