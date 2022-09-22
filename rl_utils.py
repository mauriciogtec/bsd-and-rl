import numpy as np
import os
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class CustomEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.callback is not None:
                return self.callback._on_step("latest_eval.csv")


class  BestSaveCallback(BaseCallback):
    def __init__(
        self,
        algo,
        env_class,
        savedir,
        xrange,
        yrange,
        xname = "x",
        yname = "y",
        xres = 500,
        yres = 500,
        eval_name = "eval",
        add_stopped_to_state = False
    ):
        super().__init__()
        assert algo in ("dqn", "ppo")
        self.algo = algo
        self.savedir = savedir
        self.xname = xname
        self.yname = yname
        self.eval_name = eval_name
        self.add_stopped_to_state = add_stopped_to_state
        eval_states = []
        save_states = []
        for x in np.linspace(*xrange, num=xres):
            for y in np.linspace(*yrange, num=yres):
                save_states.append([x,  y])
                if add_stopped_to_state:
                    eval_states.append(env_class.get_state(x,  y, 0))
                else:
                    eval_states.append(env_class.get_state(x,  y))
        self.eval_states = torch.FloatTensor(eval_states)
        self.save_states = np.array(save_states, np.float32)

    def _on_step(self, file="best_eval.csv") -> bool:
        with torch.no_grad():
            if self.algo == "dqn":
                eval = self.model.q_net(self.eval_states)
            elif self.algo == "ppo":
                net = self.model.policy
                features = net.extract_features(self.eval_states)
                latent_pi = net.mlp_extractor.policy_net(features)
                eval = net.action_net(latent_pi).softmax(-1)
            act = eval.argmax(-1)
            df = pd.DataFrame({
                self.xname: self.save_states[:, 0],
                self.yname: self.save_states[:, 1],
                "act": act.cpu().numpy(),
                f"{self.eval_name}0": eval[:, 0].cpu().numpy(),
                f"{self.eval_name}1": eval[:, 1].cpu().numpy(),
                f"{self.eval_name}2": eval[:, 2].cpu().numpy()
            })
        savedir = os.path.join(self.savedir, file)
        df.to_csv(savedir, index=False)
