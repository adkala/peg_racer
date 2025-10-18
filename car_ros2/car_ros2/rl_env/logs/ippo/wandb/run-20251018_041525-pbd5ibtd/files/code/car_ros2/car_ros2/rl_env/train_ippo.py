import os
os.environ["OMP_NUM_THREADS"] = "1"

from rl_env.opp_train_env import RLMultiFromCarsEnv, SB3PolicyFn
from rl_env.old2gym import OldToGymnasium

import wandb
from wandb.integration.sb3 import WandbCallback

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.logger import configure

trajectory_type = "../../../simulators/params-num.yaml"
NUM_ENVS = 4
BASE_SEED = 42
ROOT_LOG_DIR = "./logs/ippo"
LOG_DIR = "./logs/opp_ppo_vec"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ROOT_LOG_DIR, exist_ok=True)


wandb.tensorboard.patch(root_logdir=ROOT_LOG_DIR)

run = wandb.init(
    project="opp-ippo",
    name="ippo-selfplay",
    group="ippo-selfplay",
    job_type="train",
    config={
        "algo": "PPO",
        "num_envs": NUM_ENVS,
        "trajectory_type": trajectory_type,
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps_effective": 128,
        "batch_size": 64,
        "n_epochs": 5,
    },
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
    reinit=False,
    dir=ROOT_LOG_DIR,
)


def make_wandb_callback(tag: str) -> WandbCallback:
    return WandbCallback(
        gradient_save_freq=0,
        model_save_path=f"./models/{run.id}/{tag}",
        model_save_freq=50_000,
        log="all",
        verbose=2,
    )

def make_env(control: str, rank: int, seed: int, opponent_resolver):

    def gen_monitor_env():

        old = RLMultiFromCarsEnv(
            trajectory=trajectory_type,
            control=control,
            opponents=opponent_resolver
        )
        env = OldToGymnasium(old)

        fname = os.path.join(LOG_DIR, f"monitor-{control}-{rank}.csv")
        env = Monitor(env, filename=fname)

        env.reset(seed=seed + rank)
        return env
    
    return gen_monitor_env


def set_vec_opponents(env, opponents_dict):

    env.env_method("set_opponents", opponents_dict)




def build_model(control: str, num_envs: int, tb_dir: str):

    env = DummyVecEnv([make_env(control, i, BASE_SEED, opponent_resolver=lambda: {}) for i in range(num_envs)])
    env = VecMonitor(env)

    os.makedirs(tb_dir, exist_ok=True)
    logger = configure(tb_dir, ["stdout", "csv", "tensorboard"])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        n_steps=max(256, 2048 // max(1, num_envs)),
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=2,
        tensorboard_log=tb_dir,
    )

    model.set_logger(logger)
    return model


def train_ippo_selfplay(total_steps=1_000_000, chunk=10_000, num_envs=NUM_ENVS):

    m0 = build_model("car0", num_envs, os.path.join(ROOT_LOG_DIR, "car0"))
    m1 = build_model("car1", num_envs, os.path.join(ROOT_LOG_DIR, "car1"))
    m2 = build_model("car2", num_envs, os.path.join(ROOT_LOG_DIR, "car2"))

    cb0 = make_wandb_callback("car0")
    cb1 = None
    cb2 = None

    steps_done = 0
 
    while steps_done < total_steps:

        opp = {"car1": SB3PolicyFn(m1), "car2": SB3PolicyFn(m2)}
        set_vec_opponents(m0.get_env(), opp)
        m0.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=True, callback=cb0)
        steps_done += chunk
        run.log({"trainer/steps_done": steps_done, "trainer/agent": "car0"}, commit=False)

        opp = {"car0": SB3PolicyFn(m0), "car2": SB3PolicyFn(m2)}
        set_vec_opponents(m1.get_env(), opp)
        m1.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=True, callback=cb1)
        steps_done += chunk
        run.log({"trainer/steps_done": steps_done, "trainer/agent": "car1"}, commit=False)


        opp = {"car0": SB3PolicyFn(m0), "car1": SB3PolicyFn(m1)}
        set_vec_opponents(m2.get_env(), opp)
        m2.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=True, callback=cb2)
        steps_done += chunk
        run.log({"trainer/steps_done": steps_done, "trainer/agent": "car2"})

        os.makedirs("models", exist_ok=True)
        m0.save("models/ippo_car0.zip")
        m1.save("models/ippo_car1.zip")
        m2.save("models/ippo_car2.zip")

 
    run.finish()

def main():
    train_ippo_selfplay()

if __name__ == "__main__":
    main()