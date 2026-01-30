#!/usr/bin/env python3

# import debugpy
# debugpy.listen(10010)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")
import sys
sys.path.insert(0, '../../../')

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_string("render_mode", "rgb_array", "Render mode (human, rgb_array).")
flags.DEFINE_string("model_part", "full", "Curriculum stage (grasp, lift, move, place, full).")
flags.DEFINE_string("load_state_path", None, "Path to load curriculum states from (override auto-inference).")
flags.DEFINE_string("save_state_path", None, "Path to save curriculum states to (override auto-inference).")
flags.DEFINE_integer("port", 5588, "Port for the trainer server.")
flags.DEFINE_integer("broadcast_port", 5589, "Broadcast port for the trainer server.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

import threading
import glob
import queue
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from flax.training import checkpoints
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # 防止 JAX 佔滿顯存
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"  # 或者限制每個進程只用 50%
import copy
import pickle as pkl
# from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.networks.split_encoder import SplitObsEncoder
from serl_launcher.networks.transformer_encoder import SplitObsTransformerEncoder
from serl_launcher.networks.mlp import MLP
from serl_launcher.networks.actor_critic_nets import Critic, GraspCritic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from functools import partial
import flax.linen as nn

from experiments.mappings import CONFIG_MAPPING


devices = jax.local_devices()
num_devices = len(devices)
# Compatibility for newer JAX versions where PositionalSharding is removed
try:
    sharding = jax.sharding.PositionalSharding(devices)
except AttributeError:
    if num_devices == 1:
        sharding = jax.sharding.SingleDeviceSharding(devices[0])
        # Monkey patch replicate for compatibility
        sharding.replicate = lambda: sharding
    else:
        mesh = jax.sharding.Mesh(devices, ('devices',))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('devices'))
        sharding.replicate = lambda: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def make_state_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=(),
    encoder_type="mlp", # Ignored/MLP
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    # Additional args
    policy_kwargs = {
        "tanh_squash_distribution": True,
        "std_parameterization": "exp",
        "std_min": 1e-5,
        "std_max": 5,
    },
    critic_network_kwargs = {
        "activations": nn.tanh,
        "use_layer_norm": True,
        "hidden_dims": [1024, 1024],
    },
    policy_network_kwargs = {
        "activations": nn.tanh,
        "use_layer_norm": True,
        "hidden_dims": [1024, 1024],
    },
):
    # Encoders: Identity or MLP.
    # EncodingWrapper supports empty image_keys, effectively passing proprio.
    # However, create_pixels sets up encoders. create does not.
    # We will use SACAgent.create directly.

    # 1. Define Encoders
    # Replaced EncodingWrapper with SplitObsEncoder for state splitting

    # User requested SplitObsEncoder with ResNet architecture
    encoder_def = SplitObsEncoder()
    # Using Transformer Encoder for improved sample efficiency with obstacles
    # encoder_def = SplitObsTransformerEncoder()

    encoders = {
        "critic": encoder_def,
        "actor": encoder_def,
    }

    # 2. Define Networks
    critic_ensemble_size = 2

    critic_backbone = partial(MLP, **critic_network_kwargs)
    critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
        name="critic_ensemble"
    )
    critic_def = partial(
        Critic, encoder=encoders["critic"], network=critic_backbone
    )(name="critic")

    policy_def = Policy(
        encoder=encoders["actor"],
        network=MLP(**policy_network_kwargs),
        action_dim=sample_action.shape[-1],
        **policy_kwargs,
        name="actor",
    )

    temperature_def = GeqLagrangeMultiplier(
        init_value=1.0,
        constraint_shape=(),
        constraint_type="geq",
        name="temperature",
    )

    agent = SACAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        actor_def=policy_def,
        critic_def=critic_def,
        temperature_def=temperature_def,
        critic_ensemble_size=critic_ensemble_size,
        discount=discount,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        image_keys=image_keys,
    )

    return agent

##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(f'Delta time: {dt}')

                    # success_counter += reward
                    success_counter += int(info['succeed'])
                    print(f"Reward: {reward}    Success counter / episode: {success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit
    
    # start_step = (
    #     int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
    #     if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
    #     else 0
    # )
    # [修改] 先檢查有沒有 buffer 檔案，避免因為我們手動刪除 buffer 而崩潰
    buffer_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
    
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path) and len(buffer_files) > 0:
        start_step = int(os.path.basename(natsorted(buffer_files)[-1])[12:-4]) + 1
    else:
        start_step = 0  # 如果找不到 buffer 檔案，就從 0 開始計數
    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(port_number=FLAGS.port, broadcast_port=FLAGS.broadcast_port),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # === [FIX START] Asynchronous Network Update ===
    # Using a queue to pass ready-to-use params from background thread to main thread
    param_update_queue = queue.Queue(maxsize=1)

    def update_params(params):
        """
        Callback triggered by the networking thread when new params arrive.
        Crucially, we do the heavy JAX device transfer HERE (in the background),
        not in the main loop.
        """
        try:
            # 1. Pre-fetch to device (this is the slow part on low-mem GPUs)
            # We use jax.tree_map to ensure all leaves are put on the device
            # Note: 'agent' here refers to the one captured from outer scope,
            # but we only need its structure/sharding info if we were being fancy.
            # Ideally, just putting it on device is enough.
            # We assume params structure matches agent.state.params

            # This 'params' is likely a dict of numpy arrays from the network.
            # We cast it to JAX arrays on the correct device.
            device_params = jax.device_put(params, sharding.replicate())

            # 2. Block until transfer is done to ensure the main thread
            # receives fully ready data (optional but safer for "stop" diagnosis)
            # jax.block_until_ready(device_params)

            # 3. Put into queue for main thread to pick up instantly
            if param_update_queue.full():
                try:
                    param_update_queue.get_nowait() # Discard old update if main loop is slow
                except queue.Empty:
                    pass
            param_update_queue.put(device_params)

        except Exception as e:
            print(f"Error in background param update: {e}")

    client.recv_network_callback(update_params)
    # === [FIX END] =================================

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    
    transitions = []
    demo_transitions = []

    # === [FIX START] 新增：啟動一個專門的存檔執行緒 ===
    save_queue = queue.Queue(maxsize=3)

    def save_worker():
        while True:
            # 從 Queue 拿出資料 (data, filepath)
            data, filepath = save_queue.get()
            try:
                # 執行原本的存檔邏輯
                with open(filepath, "wb") as f:
                    pkl.dump(data, f)
            except Exception as e:
                print(f"Error saving file {filepath}: {e}")
            finally:
                # 標記任務完成，釋放記憶體
                del data 
                save_queue.task_done()

    # 啟動 Worker 執行緒
    worker_thread = threading.Thread(target=save_worker, daemon=True)
    worker_thread.start()
    # === [FIX END] =================================
    
    obs, _ = env.reset()
    done = False

    step_start_time = time.time()
    for step in pbar:
        step_duration = time.time() - step_start_time
        if step_duration > 1.0:
            # Use len(data_store) to get the current number of items
            print(f"[DEBUG] Actor Step {step-1} took {step_duration:.4f}s. Queue size: {len(data_store)}")
        step_start_time = time.time()

        timer.tick("total")

        # === [FIX START] Check for new params in main loop ===
        try:
            # Check if new weights are available from the background thread
            new_params = param_update_queue.get_nowait()
            # Swap instantly (pointer swap)
            agent = agent.replace(state=agent.state.replace(params=new_params))
            # print("Updated agent params!")
        except queue.Empty:
            pass
        # === [FIX END] =======================================

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            data_store.insert(transition)
            transitions.append(transition.copy())

            if already_intervened:
                intvn_data_store.insert(transition)
                # 同樣移除 deepcopy
                demo_transitions.append(transition.copy())

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                # Use total_episode_reward if available for correct display
                display_return = info.get("total_episode_reward", running_return)
                pbar.set_description(f"Total Reward: {display_return:.2f}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path): os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path): os.makedirs(demo_buffer_path)

            if len(transitions) > 0:
                if not save_queue.full():
                    # 傳送 copy 或讓 transitions 指向新 list，確保資料安全
                    save_queue.put((transitions, os.path.join(buffer_path, f"transitions_{step}.pkl")))
                else:
                    # 硬碟來不及寫，放棄這次存檔，避免記憶體爆炸
                    print(f"[Warning] Disk too slow! Skipping save at step {step} to prevent lag.")
                
                transitions = [] # 無論有無存檔，都要清空記憶體！
            if len(demo_transitions) > 0:
                if not save_queue.full():
                    save_queue.put((demo_transitions, os.path.join(demo_buffer_path, f"transitions_{step}.pkl")))
                else:
                    print(f"[Warning] Disk too slow! Skipping demo save at step {step}.")
                
                demo_transitions = [] # 務必清空
            
            # 選項：檢查 Queue 是否堆積過多，如果堆積太多可以印出警告
            if save_queue.qsize() > 2:
                print(f"Warning: Save queue is backing up! Size: {save_queue.qsize()}")
            # === [FIX END] ============================================
        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)

        # Log success rate to file
        if "environment" in payload and "success_rate" in payload["environment"]:
            rate = payload["environment"]["success_rate"]
            success_dir = os.path.join(FLAGS.checkpoint_path, "success_rate")
            if not os.path.exists(success_dir):
                os.makedirs(success_dir)

            # 1. Append to history file
            with open(os.path.join(success_dir, "history_success_rate.txt"), "a") as f:
                f.write(f"Step: {step} Success Rate: {rate:.1f}%\n")

            # Store latest rate for checkpoint naming
            server.latest_success_rate = rate

        return {}  # not expecting a response

    # Create server
    server = TrainerServer(
        make_trainer_config(port_number=FLAGS.port, broadcast_port=FLAGS.broadcast_port),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.latest_success_rate = 0.0 # Initialize
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": 11*config.batch_size // 12,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 12,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU

        learner_step_start = time.time()
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        learner_step_duration = time.time() - learner_step_start
        if learner_step_duration > 1.0:
             print(f"[DEBUG] Learner Step {step} took {learner_step_duration:.4f}s")

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )
            # Create checkpoint success rate marker file
            if hasattr(server, 'latest_success_rate'):
                rate_str = f"{server.latest_success_rate:.1f}%"
                success_dir = os.path.join(FLAGS.checkpoint_path, "success_rate")
                if not os.path.exists(success_dir): os.makedirs(success_dir)
                marker_file = os.path.join(success_dir, f"checkpoint_{step}_success_rate_{rate_str}.txt")
                with open(marker_file, "w") as f:
                    f.write(f"Step: {step}\nSuccess Rate: {rate_str}\n")

        time.sleep(0.01)


##############################################################################


def main(_):
    # Check JAX backend and warn if CPU
    if jax.default_backend() == "cpu":
        print("\n" + "="*60)
        print_green("WARNING: JAX is running on CPU!")
        print_green("This will cause very high CPU usage and low performance.")
        print_green("Please ensure CUDA and cuDNN are correctly installed and visible.")
        print_green("Note: MuJoCo physics always runs on CPU (unless MJX is used), so high CPU usage is expected with large batch sizes.")
        print("="*60 + "\n")

    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    if FLAGS.batch_size is not None:
        config.batch_size = FLAGS.batch_size

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    # Curriculum Path Logic
    STAGES = ["grasp", "lift", "move", "place", "full", "grasp_lift", "grasp_lift_move"]
    load_state_dir = FLAGS.load_state_path
    save_state_dir = FLAGS.save_state_path

    if FLAGS.checkpoint_path:
        # Auto-infer save path: checkpoint/curriculum/{model_part}
        if not save_state_dir:
            save_state_dir = os.path.join(FLAGS.checkpoint_path, "curriculum", FLAGS.model_part)

        # Auto-infer load path: checkpoint/curriculum/{prev_stage}
        # Note: grasp_lift and grasp_lift_move start from scratch, so we exclude them from auto-loading
        STAGES_WITH_PRIOR = ["lift", "move", "place", "full"]
        if not load_state_dir and FLAGS.model_part in STAGES_WITH_PRIOR:
            idx = STAGES.index(FLAGS.model_part)
            if idx > 0:
                prev_stage = STAGES[idx-1]
                load_state_dir = os.path.join(FLAGS.checkpoint_path, "curriculum", prev_stage)
                print_green(f"Auto-inferred curriculum load path: {load_state_dir}")

    if FLAGS.exp_name == 'pick_cube_sim'or FLAGS.exp_name == 'stack_cube_sim':
        if FLAGS.actor:
            env = config.get_environment(
                fake_env=FLAGS.learner,
                save_video=FLAGS.save_video,
                classifier=False,
                render_mode=FLAGS.render_mode,
                model_part=FLAGS.model_part,
                load_state_dir=load_state_dir,
                save_state_dir=save_state_dir,
            )
        else:
            env = config.get_environment(
                fake_env=FLAGS.learner,
                save_video=FLAGS.save_video,
                classifier=False,
                render_mode="rgb_array",
                model_part=FLAGS.model_part,
                load_state_dir=load_state_dir,
                save_state_dir=save_state_dir,
            )
    else:
        env = config.get_environment(
                fake_env=FLAGS.learner,
                save_video=FLAGS.save_video,
                classifier=True,
                render_mode="rgb_array",
                model_part=FLAGS.model_part,
                load_state_dir=load_state_dir,
                save_state_dir=save_state_dir,
            )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    if len(config.image_keys) == 0:
        # State-based agent
        agent = make_state_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True if "learned-gripper" in config.setup_mode else False
        print_green("Initialized State-Based SAC Agent")

    elif config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        
        # === [FIX START] 先檢查有沒有真的 checkpoint 檔案，避免 NoneType 錯誤 ===
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        
        if latest_ckpt is not None:
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
            )
            agent = agent.replace(state=ckpt)
            ckpt_number = os.path.basename(latest_ckpt)[11:]
            print_green(f"Loaded previous checkpoint at step {ckpt_number}.")
        else:
            print_green("⚠️ No checkpoint found in the existing path. Starting training from scratch.")
        # === [FIX END] ==========================================================

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        # Construct unique ID for WandB to enable deterministic resuming
        ckpt_name = os.path.basename(FLAGS.checkpoint_path.rstrip('/')) if FLAGS.checkpoint_path else "default"
        # Format: {exp_name}_{unique_id} -> stack_cube_sim_run27_grasp
        unique_id = f"{ckpt_name}_{FLAGS.model_part}"

        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
            unique_id=unique_id,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']                    
                    # ly debug
                    # transition['grasp_penalty'] = 0.0
                    
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        # if FLAGS.checkpoint_path is not None and os.path.exists(
        #     os.path.join(FLAGS.checkpoint_path, "buffer")
        # ):
        #     for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
        #         with open(file, "rb") as f:
        #             transitions = pkl.load(f)
        #             for transition in transitions:
        #                 replay_buffer.insert(transition)
        #     print_green(
        #         f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
        #     )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            # 1. 搜尋所有 .pkl 檔案
            all_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
            
            # 2. 進行自然排序 (確保順序是 1, 2, ... 10, 11 而不是 1, 10, 11, 2)
            all_files = natsorted(all_files)
            
            # 3. Load all buffer files to ensure full history is retained
            load_files = all_files
            
            print_green(f"Selected {len(load_files)} buffer files to load (out of {len(all_files)}).")

            # 4. 載入選定的檔案 (包含自動刪除壞檔機制)
            for file in load_files:
                try:
                    with open(file, "rb") as f:
                        transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
                except (EOFError, pkl.UnpicklingError, Exception) as e:
                    print(f"⚠️ 發現損壞的 Buffer 檔案: {file}, 正在自動刪除...")
                    try:
                        os.remove(file)
                    except:
                        pass

            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )


        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2500)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(2500)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)