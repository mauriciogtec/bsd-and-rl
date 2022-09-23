import os
import argparse
import torch
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from rl_envs import ScaleDoseEnv, BernoulliEnv
from rl_utils import BestSaveCallback, CustomEvalCallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--env", type=str, default="ex1", choices=["ex1", "ex2"])
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--init_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--delayed_cost", default=False, action="store_true")
    parser.add_argument("--add_stopped_to_state", default=False, action="store_true")
    parser.add_argument("--truth_reward", default=False, action="store_true")
    parser.add_argument("--cache_size", type=int, default=None)
    args = parser.parse_args()

    savedir = f"results/rl_{args.env}/seed_{args.seed}"
    os.makedirs(savedir, exist_ok=True)
    np.random.seed(args.seed)

    if args.seed is None:
        args.seed = np.random.randint(100_000, 1_000_000)
    
    if args.env == "ex1":
        env_class = BernoulliEnv
        env_kwargs = dict(
            max_steps=args.max_steps,
            lo=0.4,
            hi=0.6
        )
        xrange = [1.0 / 50, 1.0]  # for plotting
        yrange = [0.0, 1.0]
    elif args.env == "ex2":
        env_class = ScaleDoseEnv
        env_kwargs = dict(
            max_steps = args.max_steps,
            init_steps = args.init_steps,
            lam0 = np.array([1.0, 1.0 + 1e-6]),
            mu0 = (0.5, 1.0),
            sig0 = (1.0, 1.0),
            dose_step = 1.0,
            cache_size = args.cache_size,
            truth_reward = args.truth_reward
        )
        xrange = [0.01, 1.0]
        yrange = [-1.5, 2.5]

    if args.algo == "ppo":
        policy_kwargs = dict(
            activation_fn=torch.nn.SiLU,
            net_arch=[dict(pi=[32, 16], vf=[32, 16])]
        )
        model_fn = PPO
        model_kwargs = dict(
            # learning_rate=1e-3,
            # batch_size=1024,
            # n_epochs=50,
            n_steps=2**14 // args.nproc
        )
    elif args.algo == "dqn":
        policy_kwargs = dict(net_arch=[32, ], activation_fn=torch.nn.SiLU)
        model_fn = DQN
        model_kwargs = dict(
            buffer_size=1_000_000,
        )

    def env_fn():
        env = env_class(
            delayed_cost=args.delayed_cost,
            add_stopped_to_state = args.add_stopped_to_state,
            **env_kwargs
        )
        # env = TimeLimit(env, args.max_steps)
        return env

    if args.nproc > 1:
        train_env = SubprocVecEnv(args.nproc * [env_fn], start_method="spawn")
        test_env = SubprocVecEnv(args.nproc * [env_fn], start_method="spawn")
        # test_env = Monitor(test_env, filename=f"{savedir}/test_monitor.csv")  # throws error
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
        gamma=0.99,
        **model_kwargs
    )

    callback_on_new_best = BestSaveCallback(
        args.algo, 
        env_class,
        savedir,
        xrange,
        yrange,
        add_stopped_to_state=args.add_stopped_to_state
    )
    callback = CustomEvalCallback(
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
