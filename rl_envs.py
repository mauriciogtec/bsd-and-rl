import numpy as np
import gym
from gym import spaces
from scipy.special import logit
from scipy.stats import norm
from collections import deque


class BernoulliEnv(gym.Env):
    def __init__(
        self,
        K: int = 100,
        max_steps=50,
        lo: float = 0.4,
        hi: float = 0.6,
        reward_scaler: float = None,
        delayed_cost: bool = True,
        add_stopped_to_state: bool = False
    ) -> None:
        super(BernoulliEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.obs_dim = int(add_stopped_to_state) + 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.add_stopped_to_state = add_stopped_to_state
        self.delayed_cost = delayed_cost
        self.K = K
        self.max_steps = max_steps
        self.trial_stopped = False
        self.lo = lo
        self.hi = hi
        if reward_scaler is None:
            self.reward_scaler = K
        else:
            self.reward_scaler = reward_scaler
        self.reset()

    @classmethod
    def get_state(self, t_rel, phat, stopped=None):
        scale = 1.0 / np.sqrt(50 * t_rel)
        out = [phat, t_rel, phat * scale, scale]
        if stopped is not None:
            out += [int(stopped)]
        return np.array(out, dtype=np.float32)

    def _get_state(self):
        d = self.trial_stopped if self.add_stopped_to_state else None
        trel = (self.t + 1) / self.max_steps
        return self.get_state(trel, self.phat, d)

    def step(self, action: int) -> tuple:
        # bernoulli sample
        y = int(np.random.rand() < self.true_p)

        self.t += 1
        # updates running mean
        self.phat = (self.phat * (self.t - 1) + y) / self.t

        if self.delayed_cost:
            self.trial_stopped = action != 0
            done = self.trial_stopped or self.t >= self.max_steps
            reward = - (self.t - 1) * self.trial_stopped
        else:
            self.trial_stopped = action != 0
            done = self.trial_stopped or self.t >= self.max_steps
            reward = -1.0
        
        info = {}
        
        state1 = self._get_state()

        if action != 0:
            guess_p = self.lo * (action == 1) + self.hi * (action == 2)
            reward += - self.K * (self.true_p != guess_p)

        reward /= self.reward_scaler
        return state1, reward, done, info

    def reset(self) -> None:
        self.t = 0
        self.true_p = np.random.choice([self.lo, self.hi])
        self.phat = np.random.rand()
        self.trial_stopped = False
        state = self._get_state()
        return state


class ScaleDoseEnv(gym.Env):
    def __init__(
        self,
        K: int = 100,
        max_steps=50,
        mu0=(0.0, 0.5),
        sig0=(1.0, 1.0),
        alpha=0.05,
        beta=0.2,
        ngrid=1000,
        lam0=(0.1, 1.0),
        init_steps=10,
        dose_step=1.0,
        reward_scaler: float = None,
        delayed_cost: bool = False,
        add_stopped_to_state: bool = False,
        cache_size: int = None,
        truth_reward: bool = False
    ) -> None:
        super(ScaleDoseEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.add_stopped_to_state = add_stopped_to_state
        self.obs_dim = 4 + int(add_stopped_to_state)
        self.cache_size = cache_size
        if cache_size is not None:
            self.cache = deque(maxlen=self.cache_size)
            self.cache_ptr = -1
        self.truth_reward = truth_reward

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        if reward_scaler is None:
            self.reward_scaler = K
        else:
            self.reward_scaler = reward_scaler
        self.K = K
        self.max_steps = max_steps
        self.done = False
        self.delayed_cost = delayed_cost
        self.trial_stopped = False
        self.mu0 = mu0
        self.sig0 = sig0
        self.x0 = dose_step
        self.alpha = alpha
        self.beta = beta
        self.ngrid = ngrid
        self.lam0 = lam0
        self.init_steps = init_steps
        self.dose_step = dose_step
        self.max_n2 = K
        self.reset()

    @classmethod
    def get_state(self, sl, dl, stopped=None):
        if dl is None:
            dl = 0.95 * self.b_post_mu
            sl = 0.95 * self.b_post_sig
        out = [dl, sl, dl / sl, sl * dl]
        if stopped is not None:
            out += [stopped]
        return np.array(out, dtype=np.float32)

    def _get_state(self):
        if self.add_stopped_to_state:
            stopped = float(self.trial_stopped)
        else:
            stopped = None
        return self.get_state(
            0.95 * self.b_post_sig,
            0.95 * self.b_post_mu,
            stopped
        )

    def step(self, action: int) -> tuple:
        self.t += 1

        if self.delayed_cost:
            self.trial_stopped = action != 0
            done = self.trial_stopped or self.t >= self.max_steps - 1
            reward = - (self.t - self.init_steps + 1) * self.trial_stopped
        else:
            self.trial_stopped = action != 0
            done = self.trial_stopped or self.t >= self.max_steps - 1
            reward = - 1.0

        info = {}

        if action == 2:
            pd, n2 = self._reject_post_prob()
            if self.truth_reward:
                reward += - n2 + int(self.b > 0) * self.K
            else:
                reward += - n2 + pd * self.K
        reward /= self.reward_scaler

        if self.cache_size is not None:
            _, _, states = self.cache[self.cache_ptr]
            self.mu_e, self.b_post_mu, self.b_post_sig = states[self.t]
        else:
            self._enroll_patient()

        state1 = self._get_state()

        return state1, reward, done, info

    def _reject_post_prob(self) -> float:
        qa = norm().ppf(1.0 - self.alpha)
        qb = norm().ppf(1.0 - self.beta)
        dl = 0.95 * self.b_post_mu
        sl = 0.95 * self.b_post_sig
        dstar = dl - sl
        n2 = np.ceil(4 * ((qa + qb) / dstar) ** 2)
        n2 = min(n2, self.max_n2)
        bb = (0.5 * dl * np.sqrt(n2) - qa) / np.sqrt(1.0 + 0.25 * n2 * sl ** 2)
        pd = norm().cdf(bb)
        return pd, n2

    def _enroll_patient(self) -> None:
        xs = np.array(self.xs)
        Y = np.array(self.ys)  # (N, )
        Phi = xs[:, None] / (
            self.e_grid[None, :] + xs[:, None]
        )  # (N, grid_size)
        obs_prec = 1 / np.square(self.obs_noise)
        prec0 = obs_prec # * 0.1
        e_prec_0 = obs_prec  #* 0.1

        V = 1.0 / (
            obs_prec * np.square(Phi).sum(0) + prec0
        )  # (grid_size, )
        aa = (
            obs_prec * (Phi * Y[:, None]).sum(0) + prec0 * self.mu0[0]
        )  # (grid_size, )
        m = aa * V  # (verified)
        log_wts = (
            0.5 * (aa ** 2) * V
            - 0.5 * e_prec_0 * (self.e_grid - self.mu0[1]) ** 2
            - np.log(np.sqrt(V))
        )  # (grid_size, )

        wts = np.exp(log_wts - max(log_wts)) + 1e-12
        wts /= wts.sum()
        wts_ = wts.copy()
        # truncval = np.quantile(wts_, 0.05)
        # wts_[wts < truncval - 1e-12] = 0.0  # truncate for stability
        # wts_ /= wts_.sum()
        mu_e = np.average(self.e_grid, weights=wts_)
        self.mu_e = mu_e
        mode_e = self.e_grid[wts_.argmax()]

        # ed95 estimate
        ed95 = (1.0 - self.alpha) / self.alpha * mu_e

        # evaluate b posterior
        self.b_post_mu = np.average(m, weights=wts_)
        self.b_post_sig = np.sqrt(
            np.average(V + m**2, weights=wts_) - self.b_post_mu ** 2
        )

        # update grid for future round
        thresh = 0.0 # 1e-2
        ix = np.where(wts > thresh / len(wts))[0]
        imin = max(ix[0] - 1, 0)
        imax = min(ix[-1] + 1, self.ngrid - 1)
        self.e_grid = np.linspace(
            self.e_grid[imin], self.e_grid[imax], num=self.ngrid
        )
        x = min(
            ed95, self.xs[-1] + self.dose_step
        )  # increase dose and stop at ed95
        noise = np.random.normal(scale=self.obs_noise)
        y = self.b * x / (x + self.e) + noise
        self.xs.append(x)
        self.ys.append(y)
    
        return self.mu_e, self.b_post_mu, self.b_post_sig

    def reset(self) -> None:
        if self.cache_size is not None:
            self.cache_ptr = (self.cache_ptr + 1) % self.cache_size
        self.xs = []
        self.ys = []
        self.trial_stopped = False
        self.obs_noise = np.random.uniform(*self.lam0)
        self.b = np.random.normal(self.mu0[0], self.sig0[0])
        self.e = max(0.1, np.random.normal(self.mu0[1], self.sig0[1]))
        self.e_grid = np.linspace(0.1, 10.0, num=self.ngrid)
        # self.init_steps = np.random.randint(1, 3)
        # cache an episode

        self.t = 0
        if self.cache_size is not None:
            if len(self.cache) != self.cache_size:
                states = []
                x = self.x0
                noise =  np.random.normal(scale=self.obs_noise)
                y = self.b * x / (x + self.e) + noise
                self.xs.append(x)
                self.ys.append(y)
                for _ in range(self.max_steps):
                    next_states = self._enroll_patient()
                    states.append(next_states)
                self.cache.append((self.e, self.b, states))
                self.t = self.init_steps - 1
                self.e, self.b, states = self.cache[self.cache_ptr]
                self.e_mu, self.b_post_mu, self.b_post_sig = states[self.t]
        else:
            x = self.x0
            noise =  np.random.normal(scale=self.obs_noise)
            y = self.b * x / (x + self.e) + noise
            self.xs.append(x)
            self.ys.append(y)
            self._enroll_patient()

        return self._get_state()
