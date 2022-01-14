import os
import argparse
import torch
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gym.wrappers import TimeLimit
from envs import ScaleDoseEnv, BernoulliEnv
from utils import SaveCallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env", type=str, default="ex1", choices=["ex1", "ex2"])
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--init_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--delayed_cost", default=False, action="store_true")
    args = parser.parse_args()

    savedir = f"logs_{args.env}/seed_{args.seed}"
    os.makedirs(savedir, exist_ok=True)
    np.random.seed(args.seed)

    if args.seed is None:
        args.seed = np.random.randint(100_000, 1_000_000)
    
    if args.env == "ex1":
        env_class = BernoulliEnv
        env_kwargs = dict(
            max_steps=args.max_steps + 1,
            lo=0.4,
            hi=0.6
        )
        xrange = [1.0, 50.0]  # for plotting
        yrange = [0.0, 1.0]
    elif args.env == "ex2":
        env_class = ScaleDoseEnv
        env_kwargs = dict(
            max_steps = args.max_steps + 1,
            init_steps = args.init_steps,
            lam0 = np.sqrt([0.5, 0.51]),
            mu0 = (1.0, 1.0),
            sig0 = (1.0, 0.5),
            dose_step = 1.0,
            add_stopped_to_state = False
        )
        xrange = [0.01, 1.0]
        yrange = [-1.0, 2.5]

    if args.algo == "ppo":
        policy_kwargs = dict(
            activation_fn=torch.nn.SiLU,
            net_arch=[dict(pi=[64, ], vf=[64, ])]
        )
        model_fn = PPO
        model_kwargs = dict(ent_coef=0.001, batch_size=256)
    elif args.algo == "dqn":
        policy_kwargs = dict(net_arch=[64, ], activation_fn=torch.nn.SiLU)
        model_fn = DQN
        model_kwargs = dict(buffer_size=1_000_000, batch_size=256)

    def env_fn():
        env = env_class(delayed_cost=args.delayed_cost, **env_kwargs)
        env = TimeLimit(env, args.max_steps)
        return env

    if args.nproc > 1:
        train_env = SubprocVecEnv(args.nproc * [env_fn], start_method="spawn")
    else:
        train_env = env_fn()
    test_env = Monitor(env_fn(), filename=f"{savedir}/test_monitor.csv")

    model = model_fn(
        "MlpPolicy",
        env=train_env,
        verbose=0,
        tensorboard_log=f"{savedir}/tensorboard",
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        gamma=1,
        **model_kwargs
    )

    callback_on_new_best = SaveCallback(
        args.algo, 
        env_class,
        savedir,
        xrange,
        yrange,
    )
    callback = EvalCallback(
        best_model_save_path=savedir,
        log_path=savedir,
        eval_env=test_env,
        eval_freq=10_000 // args.nproc,
        n_eval_episodes=1_000,
        deterministic=True,
        callback_on_new_best=callback_on_new_best
    )

    model.learn(
        total_timesteps=10_000_000,
        log_interval=10_000,
        callback=callback
    )
