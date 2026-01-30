import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten
import numpy as np


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        # Check if images exist and are not empty
        images_space = {}
        if "images" in self.env.observation_space.spaces:
             img_space_candidate = self.env.observation_space["images"]
             if hasattr(img_space_candidate, 'spaces') and len(img_space_candidate.spaces) > 0:
                 images_space = img_space_candidate
             elif isinstance(img_space_candidate, gym.spaces.Box) and img_space_candidate.shape != (0,):
                 images_space = {"images": img_space_candidate}

        if images_space:
             self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.proprio_space),
                    **images_space,
                }
            )
        else:
             self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.proprio_space),
                }
            )

    def observation(self, obs):
        # Optimization: Use np.concatenate instead of gym.flatten
        # This avoids dict creation and generic flattening overhead every step
        state_vals = [obs["state"][key] for key in self.proprio_keys]

        # Ensure all are 1D or flatten them if they are arrays
        flat_vals = []
        for v in state_vals:
            if isinstance(v, np.ndarray):
                flat_vals.append(v.ravel())
            elif isinstance(v, (float, int)):
                flat_vals.append([v])
            else:
                flat_vals.append(v)

        flat_state = np.concatenate(flat_vals).astype(np.float32)

        new_obs = {"state": flat_state}

        if "images" in obs and obs["images"] and "images" in self.observation_space.spaces:
             new_obs.update(obs["images"])

        return new_obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        # This helper also needs fixing if it's used elsewhere
        flat_state = flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            )
        new_obs = {"state": flat_state}
        if "images" in obs and obs["images"]:
             new_obs.update(obs["images"])
        return new_obs
