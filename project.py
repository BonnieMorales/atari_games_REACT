import gymnasium as gym
import os
import ale_py
import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecTransposeImage,VecVideoRecorder, DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor



env_id = "ALE/DonkeyKong-v5"

def make_env(env_id: str, rank: int, seed: int = 0, obs_type: str = "ram", render_mode=None):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, obs_type=obs_type ,render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def train():
    # ----- Set up the environment -----
    num_cpu = 12  # Number of processes to use
    buffer_size = 100_000

    # Create the atari vectorized environment (vectorized by default)
    if args.obs_type == "rgb":
        vec_env = make_atari_env(env_id, n_envs=num_cpu, seed=0)
        # Stacking 4 consecutive frames together to understand ball velocity
        vec_env = VecFrameStack(vec_env, 4)
        vec_env = VecTransposeImage(vec_env)

        eval_env = make_atari_env(env_id, n_envs=1, seed=0)
        eval_env = VecFrameStack(eval_env, 4)
        eval_env = VecTransposeImage(eval_env)

    elif args.obs_type == "ram":
        vec_env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]))
        # Stacking 4 consecutive frames together to understand ball velocity
        vec_env = VecFrameStack(vec_env, 4)

        eval_env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(1)]))
        eval_env = VecFrameStack(eval_env, 4)

    else:
        print("Unknown observation type")

        exit()


    # ----- Set up the evaluation callback -----

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=10_000,  # evaluate every 10000 steps
        n_eval_episodes=5,  # average over 5 episodes
        deterministic=True,
        render=False,
    )

    # ----- Train the model -----
    # define the model
    if args.load and os.path.exists(args.load):
        print(f"Model path: {args.load} Loading model...")
        model = DQN.load(args.load,
                         vec_env,
                         tensorboard_log=tensorboard_path,
                         verbose=1
                         )
    else:
        if args.load:
            print(f"Warning: Model not found in {args.load}, creating new model...")
        else:
            print("Model not specified, creating new model...")
        policy = "CnnPolicy" if args.obs_type == "rgb" else "MlpPolicy"
        device = "cuda" if args.obs_type == "rgb" else "cpu"
        model = DQN(
            policy=policy,
            env = vec_env,
            buffer_size=buffer_size,
            verbose=1,
            tensorboard_log=tensorboard_path,
            device=device,
        )

    # Train the model`
    model.learn(args.timesteps, progress_bar=True, callback=eval_callback, reset_num_timesteps=not args.load)
    vec_env.close()
    eval_env.close()

def evaluate():
    if args.obs_type == "rgb":
        eval_env = make_atari_env(env_id, n_envs=1, seed=0, env_kwargs={"render_mode":"rgb_array"})
        # Stacking 4 consecutive frames together to understand ball velocity
        eval_env = VecFrameStack(eval_env, 4)
        eval_env = VecTransposeImage(eval_env)

    else:
        eval_env = DummyVecEnv([make_env(env_id, i, render_mode="rgb_array") for i in range(1)])
        eval_env = VecFrameStack(eval_env, 4)

    eval_env = VecVideoRecorder(eval_env,
                                video_path,
                                record_video_trigger=lambda x: x == 0,
                                video_length =50001,
                                name_prefix=f"best_model_{args.obs_type}"
                                )

    # ----- Evaluate the trained agent ----

    model = DQN.load(os.path.join(model_path, "best_model.zip"), env=eval_env)

    print("Evaluating and recording...")
    observation = eval_env.reset()
    done = False
    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, rewards, dones, info = eval_env.step(action)
        if dones[0]:
            done = True
        #eval_env.render()  # save instead, 10 million, save model, logdirectory to test model, replay_buffer 100 000, expectations
    eval_env.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DonkeyKong project")
    parser.add_argument("--obs_type", type=str, default="ram", choices=["ram", "rgb"], help="Observation type (ram/rgb)")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Total number of timesteps")
    parser.add_argument("-t", "--train", action="store_true", help="Train the model")
    parser.add_argument("-e", "--eval", action="store_true",  help="Evaluate the model")
    parser.add_argument("--load", type=str, default=None, help="Load path to model")

    # Parse the arguments
    args = parser.parse_args()

    # create directories
    log_path = os.path.join("logs", f"logs_{args.obs_type}")
    os.makedirs(log_path, exist_ok=True)

    tensorboard_path = os.path.join("tensorboard", args.obs_type)
    os.makedirs(tensorboard_path, exist_ok=True)

    model_path = os.path.join("models", args.obs_type)
    os.makedirs(model_path, exist_ok=True)

    video_path = os.path.join("logs", f"videos{args.obs_type}")
    os.makedirs(video_path, exist_ok=True)

    if args.train:
        train()
    elif args.eval:
        evaluate()
    else:
        print("Select train/eval mode")

