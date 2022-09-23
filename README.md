#  A Comparative Tutorial of Bayesian Sequential Design and Reinforcement Learning

Welcome to this repository containing the code for our paper: 

- [Tec, M., Duan, Y. and Müller, P., 2022. A Comparative Tutorial of Bayesian Sequential Design and Reinforcement Learning. To appear in: *The American Statistician*.](https://arxiv.org/abs/2205.04023)

The paper develops two examples of sequential stopping problems inspired by adaptive clinical trials, illustrating the common elements to Bayesian Sequential Design (BSD) and Reinforcement Learning (RL). In addition, all the figures in the paper are reproducible using the `rmarkdown` notebooks in the folder `notebooks`. 

The outputs of each method (BSD or RL) are stored in the folder `results/{method}_ex{example number}` folder. 

## Organization of the Code

For convenience, the code is divided in in scripts to run the BSD and RL algorithms separately. The examples in BSD are coded in `R` and the BSD examples are coded in `Python`. This is so because we wanted to use standard tools and libraries from each literature. For instance, the RL examples are coded using [`Stable Baselines 3`](https://stable-baselines3.readthedocs.io/en/master/), a popular `Python` for RL. Please review the paper for explanations of *Example 1* and *Example 2* (and to learn more about BSD and RL! =D)

## BSD

To apply Example 1 (with constrained backward induction) and Example 2 (with the parametric boundaries method) simply run
```
Rscript --vanilla bsd_ex1.R
Rscript --vanilla bsd_ex2.R
```

The outputs contain some diagnostic plots of the model output. For instance, `tracedata.csv` and `uhatdata.csv` are used by the `rmarkdown` notebooks for the paper plots.

Running both examples should take about 15 min on a regular laptop.

## RL

For RL, the script `rl_train.py` is the entrypoint to run both examples. used to run both examples with different algorithms. To reproduce the results from the paper run
```
python rl_train.py --env=ex1 --algo=dqn
python rl_train.py --env=ex2 --algo=ppo
```

The file `rl_envs.py` encodes Examples 1 and 2 as RL "environments" using the quite standard `OpenAI Gym` API. In this way, we can use the implementations of DQN and PPO in `StableBaselines3` out of the box.

Running both examples should take about 2 hours on a regular laptop.

## Reproducing paper figures

After executing the instructions above, run the Rmarkdown notebooks `plots_ex1.Rmd` and `plots_ex2.Rmd`. The easiest thing is to run them using Rstudio. Alternatively, run

```
cd notebooks
Rscript --vanilla -e "rmarkdown::render('plots_ex1.Rmd')"
Rscript --vanilla -e "rmarkdown::render('plots_ex2.Rmd')"
```

## Software requirements

The code was developed using `R 4.0` and `Python 3.9` with minimum software requirements. Here's a list of the required packages.

R
* tidyverse
* cowplot
* latex2exp
* MASS
* rmarkdown (for the paper figures)
* reticulate (for interoperability with Python, but only used for the plots)

Python
* numpy
* torch (deep learning engine)
* stable_baselines3 (reinforcement learning algorithms)
