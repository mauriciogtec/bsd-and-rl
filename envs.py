import numpy as np
import gym
from gym import spaces
from scipy.special import logit
from scipy.stats import norm


class BernoulliEnv(gym.Env):
    def __init__(
        self,
        K: int = 100,
        max_steps=50,
        lo: float = 0.4,
        hi: float = 0.6,
        reward_scaler: float = None,
        delayed_cost: bool = True
    ) -> None:
        super(BernoulliEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
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
    def get_state(self, t, phat, max_steps=50):
        scale = 1.0 if t == 0 else 1.0 / np.sqrt(t)
        out = [phat, t / max_steps, phat * scale, scale]
        return np.array(out, dtype=np.float32)

    def _get_state(self):
        return self.get_state(self.t, self.phat, self.max_steps)

    def step(self, action: int) -> tuple:
        # bernoulli sample
        y = int(np.random.rand() < self.true_p)

        self.t += 1
        # updates running mean
        self.phat = (self.phat * (self.t - 1) + y) / self.t

        if self.delayed_cost:
            self.trial_stopped = action != 0 or self.t >= self.max_steps
            done = self.trial_stopped
            reward = - self.t * self.trial_stopped
        else:
            self.trial_stopped = action != 0
            done = self.trial_stopped
            reward = -1.0
        
        info = {}
        
        state1 = self._get_state()

        if self.trial_stopped:
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
        max_steps=40,
        mu0=(0.0, 0.5),
        sig0=(1.0, 1.0),
        alpha=0.05,
        beta=0.2,
        ngrid=500,
        lam0=(0.1, 1.0),
        init_steps=10,
        dose_step=1.0,
        max_n2=100,
        reward_scaler: float = None,
        delayed_cost: bool = True,
        add_stopped_to_state: bool = False
    ) -> None:
        super(ScaleDoseEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.add_stopped_to_state = add_stopped_to_state
        self.obs_dim = 4 + int(add_stopped_to_state)
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
        self.max_n2 = max_n2
        self.reset()

    @classmethod
    def get_state(self, b_sig, b_mu, stopped=None):
        if b_mu is None:
            b_mu = self.b_post_mu
            b_sig = self.b_post_sig
        out = [b_mu, b_sig, b_mu / max(0.01, b_sig), b_mu * b_sig]
        if stopped is not None:
            out += [stopped]
        return np.array(out, dtype=np.float32)

    def _get_state(self):
        if self.add_stopped_to_state:
            stopped = float(self.trial_stopped)
        else:
            stopped = None
        return self.get_state(self.b_post_sig, self.b_post_mu, stopped)

    def step(self, action: int) -> tuple:
        # bernoulli sample
        self._enroll_patient()
        self.t += 1

        if self.delayed_cost:
            self.trial_stopped = action != 0 or self.t >= self.max_steps
            done = self.trial_stopped
            reward = - self.t * self.trial_stopped
        else:
            self.trial_stopped = action != 0
            done = self.trial_stopped
            reward = -1.0

        info = {}

        state1 = self._get_state()
        
        if action == 2:
            pd, n2 = self._reject_post_prob()
            # win_prize = int(self.b > 0)
            # reward += -n2 + win_prize * self.K
            # reward += -n2 + win_prize * self.K
            reward += - n2 + pd * self.K  # cheat?
        reward /= self.reward_scaler
        return state1, reward, done, info

    def _reject_post_prob(self) -> float:
        qa = norm().ppf(1.0 - self.alpha)
        qb = norm().ppf(1.0 - self.beta)
        dl = 0.95 * self.b_post_mu
        sl = 0.95 * self.b_post_sig
        dstar = max(0.001, dl - sl)
        n2 = np.ceil(4 * ((qa + qb) / dstar) ** 2)
        n2 = min(n2, self.max_n2)
        bb = (0.5 * dl * np.sqrt(n2) - qa) / np.sqrt(1.0 + 0.25 * n2 * sl ** 2)
        pd = norm().cdf(bb)
        return pd, n2

    def _enroll_patient(self) -> None:
        if self.t == 0:
            x = self.x0
            y = self.b * x / (x + self.e) + np.random.normal(
                scale=self.obs_noise
            )
            self.xs.append(x)
            self.ys.append(y)
            # self.e_wts = np.full((self.ngrid, ), 1.0 / self.ngrid)

            # set posterior to initial values
            self.b_post_mu = np.random.normal(self.mu0[0], self.sig0[0])
            self.b_post_sig = np.random.rand()
        else:
            xs = np.array(self.xs)
            Y = np.array(self.ys)  # (N, )
            obs_prec = 1.0 / np.square(self.obs_noise)
            Phi = xs[:, None] / (
                self.e_grid[None, :] + xs[:, None]
            )  # (N, grid_size)
            prec0 = 1.0 / self.sig0[0] ** 2
            # prec0 = 0.0 # 0.001
            V = 1.0 / (
                obs_prec * np.square(Phi).sum(0) + prec0
            )  # (grid_size, )
            aa = (
                obs_prec * (Phi * Y[:, None]).sum(0) + prec0 * self.mu0[0]
            )  # (grid_size, )
            m = aa * V  # (verified)
            e_prec_0 = 1.0 / self.sig0[1] ** 2
            # e_prec_0 = 0.01
            log_wts = (
                0.5 * aa ** 2 * V
                - 0.5 * e_prec_0 * (self.e_grid - self.mu0[1]) ** 2
                - np.log(np.sqrt(V))
            )  # (grid_size, )
            # num = -0.5 * np.square(Y[:, None] - Phi * self.b_post_mu).sum(0) * obs_prec
            # den = -0.5 * np.square(self.b_post_mu - m) / V  - np.log(np.sqrt(V))
            # prior = - 0.5 * e_prec_0 * np.square(self.e_grid - self.mu0[1])
            # log_wts = num - den + prior

            wts = np.exp(log_wts - max(log_wts))
            wts /= (wts.sum() + 1e-12)
            wts_ = wts.copy()
            # wts_[wts < 0.1 / len(wts)] = 0.0  # truncate for stability
            # wts_ /= wts_.sum()
            mu_e = np.average(self.e_grid, weights=wts_)

            # ed95 estimate
            ed95 = (1.0 - self.alpha) / self.alpha * mu_e
            x = min(
                ed95, self.xs[-1] + self.dose_step
            )  # increase dose and stop at ed95
            y = self.b * x / (x + self.e) + np.random.normal(
                scale=self.obs_noise
            )
            self.xs.append(x)
            self.ys.append(y)

            # evaluate b posterior
            self.b_post_mu = np.average(m, weights=wts_)
            b_post_var1 = np.average(V, weights=wts_)  # average variance
            b_post_var2 = np.cov(m, aweights=wts_)  # variance of averages
            self.b_post_sig = np.sqrt(b_post_var1 + b_post_var2)

            # import matplotlib.pyplot as plt; plt.plot(self.e_grid, wts);plt.show()
            # ## import matplotlib.pyplot as plt; plt.scatter(xs, Y)
            # # --- Dose response curves --- #
            nsim = 200
            samp = np.random.choice(
                range(len(wts)), size=nsim, replace=True, p=wts_
            )
            e_post = self.e_grid[samp]
            b_post = np.random.normal(m[samp], np.sqrt(V[samp]))
            # ---p
            # x_true = np.linspace(0.0, len(xs))
            # y_hat = b_post * x_true[:, None] / (x_true[:, None] + e_post)
            # y_true = (self.b * x_true / (x_true + self.e))
            # # plt.plot(xs, yhat); plt.show()
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(6, 4))
            # plt.plot(x_true, y_true, c="black", label="True dose-response function", ls="--")
            # plt.scatter(xs, Y, c="red", alpha=0.3, s=30, label="Observed data points")
            # # yhat = (self.b_post_mu * Phi)
            # # yhat = yhat[:, samp]
            # plt.plot(x_true, y_hat, c="black", lw=3, alpha=0.03); plt.show()
            # plt.plot(x_true, y_hat[:, 0], c="black", lw=3, alpha=0.1, label="Posterior predictive samples"); plt.show()
            # plt.xlabel("Dose (Xt)")
            # plt.ylabel("Response (Yt)")
            # plt.axvline(x=19 * self.e, c="blue", ls="dotted", label="True ED95")
            # plt.legend()
            # plt.savefig("figs/posterior_predictive_sample.png")
            # plt.close()
            #
            # ----- posterior b
            # import matplotlib.pyplot as plt; import seaborn as sns
            # nsim = 500
            # samp = np.random.choice(range(len(wts)), size=nsim, replace=True, p=wts_)
            # e_post = self.e_grid[samp]
            # b_post = np.random.normal(m[samp], np.sqrt(V[samp]))
            # plt.figure(figsize=(6, 4))
            # plt.hist(b_post, density=True, color="grey", bins=10, label="b posterior", alpha=0.5)
            # plt.axvline(x=self.b, c="red", ls="dotted", label="True b")
            # plt.xlabel("b")
            # plt.ylabel("density")
            # plt.legend()
            # plt.savefig("figs/posterior_b_example.png")

            # update grid for future round
            ix = np.where(wts > 0.001 / len(wts))[0]
            imin = max(ix[0] - 1, 0)
            imax = min(ix[-1] + 1, self.ngrid - 1)
            self.e_grid = np.linspace(
                self.e_grid[imin], self.e_grid[imax], num=self.ngrid
            )

    def reset(self) -> None:
        self.xs = []
        self.ys = []
        self.t = 0
        self.trial_stopped = False
        self.obs_noise = np.sqrt(np.random.uniform(*self.lam0))
        self.b = np.random.normal(self.mu0[0], self.sig0[0])
        self.e = max(0.1, np.random.normal(self.mu0[1], self.sig0[1]))
        self.e_grid = np.linspace(0.1, 5.0, num=self.ngrid)
        for _ in range(self.init_steps):
            self._enroll_patient()
        state = self._get_state()
        return state
