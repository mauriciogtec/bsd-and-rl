## Example 2

## Decision boundaries
# p1 = b1 * s + c
# p2 = - b2 * s + c

library(tidyverse)
# library(hrbrthemes)
library(MASS)
library(cowplot)
library(latex2exp)

seed = 1234
set.seed(seed)

dir.create("results/bsd_ex2", showWarnings=FALSE)

## Generate spaghetti
gen_spa <- function(Nspa,Nmx,mu0,s0,sig2_0,x0=1){
  ## Nspa: number of spaghetti
  ## Nmx: number of time periods in each spaghetti
  ## mu0, s0: parameters in priors N(mu_b,sigmab) N(mu_e,sigma_e)
  ## x0: starting dose
  ## sig2_0: variance of the noise in the model
  y_true_mat <- y_mat <- x_mat <- dl_mat <- sl_mat <- x95_est_mat <- para_mat <- c()
  for (nspa in 1:Nspa){
    # Generate true parameters in the dose-response curve
    b <- rnorm(1,mean=mu0[1],sd=s0[1])
    e <- pmax(0.1, rnorm(1,mean=mu0[2],sd=s0[2]))
    # Store response y, assigned dose x 
    # Initialization 
    y_true <- y <- x <- dl <- sl <-rep(NA,Nmx)
    # posterior accuracy
    eps <- 0.000
    ngrid <- 1000
    imin <- 1
    imax <- ngrid
    e_grid <- seq(0.1,10,length.out = ngrid)
    for (t in 1:Nmx){
      # Start from the lowest dose
      if (t==1){
        x[t] <- x0
      }else{
        x[t] <- xnext
      }
      # Enroll next patient
      y_true[t] <- b * x[t] / (e + x[t])
      y[t] <- y_true[t] + sqrt(sig2_0) * rnorm(1)
      # Calculate summary statistics dl(\bar{\delta}), sl(s_{\delta})
      yt <- y[1:t]
      xt <- x[1:t]
      e_grid <- seq(e_grid[imin],e_grid[imax],length.out = ngrid)
      xy <- xx <- rep(NA,length(e_grid))
      for (ind in 1:length(e_grid)){
        e_temp <- e_grid[ind]
        x_tilde <- xt/(e_temp+xt)
        xy[ind] <- sum(x_tilde*yt)
        xx[ind] <- sum(x_tilde^2)
      }
    
      obs_prec = 1 / sig2_0
      e_prec0 = obs_prec / s0[2] ^ 2
      b_prec0 = obs_prec / s0[1] ^ 2

      v <- 1/(xx * obs_prec + b_prec0) # formula for Var[b | e]
      aa <- (xy * obs_prec + b_prec0 * mu0[1])
      m <- aa*v # formula for E[b | e]
      w <- exp(aa^2*v / 2 - (e_grid-mu0[2])^2 * e_prec0 / 2) / sqrt(v)
      # w <- aa^2*v/2 - (e_grid-mu0[2])^2/s0[2]^2/2
      # w <- w - max(w)
      w = w / sum(w)
      ix = which(w>eps)
      if (length(ix)>0){
        imin = max(1, ix[1] - 1)
        nix = length(ix)
        imax = min(ngrid, ix[nix] + 1)
      }
      mu_e = weighted.mean(e_grid, weights=w)
      mu_b <- weighted.mean(m, weights=w)
      var_b <- weighted.mean((v + m^2), weights=w) - weighted.mean(m, weights=w)^2
      # estimated ED95
      x95_est <- 19*mu_e
      # Generate eta=(a.b)|D_t,e \sim N(m,V)
      dl[t] <- 0.95*mu_b
      sl[t] <- 0.95*sqrt(var_b)
      # Always continue trial when generating spaghtti
      xnext <- min(x95_est,x[t]+1)
      para <- c(b,e)
    }
    y_mat <- rbind(y_mat,y)
    y_true_mat <- rbind(y_true_mat, y_true)
    x_mat <- rbind(x_mat,x)
    dl_mat <- rbind(dl_mat,dl)
    sl_mat <- rbind(sl_mat,sl)
    x95_est_mat <- rbind(x95_est_mat,x95_est)
    para_mat <- rbind(para_mat,para)
  }
  return(list(y_true=y_true_mat, y=y_mat,x=x_mat,dl=dl_mat,sl=sl_mat,x95_est=x95_est_mat,para=para_mat))
}

# Input
Nmx <- 50 # Number of maximum patients
# mu0 <- c(1,1.0) # Parameters in priors N(mu_b,sigmab) N(mu_e,sigma_e)
# s0 <- c(1,0.5) 
mu0 = c(1/2, 1)
s0 = c(1, 1)

sig2_0 <- 1^2 # var of the noise
# start dose
x0 <- 1
# number of spaghetti
Nspa <- 500

spagh <- gen_spa(Nspa,Nmx,mu0,s0,sig2_0,x0)

ix = 15
plot(spagh$x[ix, ], spagh$y[ix, ])
ord = order(spagh$x[ix, ])
lines(spagh$x[ix, ord], spagh$y_true[ix, ord])

## plot first 10 spaghetti
dl.v <- sl.v <- num.v <- c()
for (i in 1:10){
  dl.v <- c(dl.v,spagh$dl[i,],NA)
  sl.v <- c(sl.v,spagh$sl[i,],NA)
  num.v <- c(num.v,rep(as.character(i),(Nmx+1)))
}

tracedata <- data.frame(dl=dl.v,sl=sl.v,num=num.v)

# ggplot(data=tracedata, aes(x=sl, y=dl,color=num)) +
#   geom_point() +
#   geom_segment(
#     aes(
#       xend=c(tail(sl, n=-1), NA), 
#       yend=c(tail(dl, n=-1), NA)
#     ),
#     arrow=arrow(length=unit(0.3,"cm"))
#   ) +
#   # stat_function(fun = function(s) b1 * s + c, color="red")+ 
#   # stat_function(fun = function(s) -b2 * s + c, color="red") +
#   xlim(c(0,1)) +
#   xlab("s_delta") + ylab("delta_bar") + 
#   theme(legend.position="none")
# ggsave("figures/Eg2spagh.jpg",height=5,width=6.5)

#########################
## The estimated utility on the 3-dims grid

# parameters in the utility functions
c1 <- c2 <- 1
K <- 100
# significant level
alpha <- 0.05
# power (1-beta)
beta <- 0.2

# paragrid <- expand.grid(c=seq(0.5,1,length.out=10),b1=seq(1,2,length.out=10),b2=seq(0.1,0.4,length.out=10))

paragrid <- expand.grid(c=seq(0.3,0.8,length.out=10),b1=seq(0.5,1.5,length.out=10),b2=seq(0.5,1.0,length.out=10))
u_est <- rep(0,nrow(paragrid))
for (i in 1:nrow(paragrid)){
  c <- paragrid$c[i]
  b1 <- paragrid$b1[i]
  b2 <- paragrid$b2[i]
  u_vec <- rep(NA,Nspa)
  dl_mat <- spagh$dl
  sl_mat <- spagh$sl
  d_mat <- N_vec <- c()
  for (nspa in 1:Nspa){
    dl <- dl_mat[nspa,]
    sl <- sl_mat[nspa,]
    d <- rep(NA,Nmx)
    for (t in 1:Nmx){
      # Compare summary statistics with decision boundaries
      if (t == Nmx){
        # if reach the maximum number of patients, decisions are 1 and 2, must stop
        if (dl[t] < c){
          d[t] <- 2
        }else{
          d[t] <- 1
        }
      }else{
        if (dl[t] < (-b2*sl[t]+c)){
          d[t] <- 2
        }else if (dl[t] > (b1*sl[t]+c)){
          d[t] <- 1
        }else{
          d[t] <- 0
        }
      }
      # Calculate utility when stops
      if (d[t]==1){
        N <- t
        n2 <- ceiling(4*(qnorm(1-alpha)+qnorm(1-beta)/(dl[N]-sl[N]))^2)
        n2 = pmin(n2, K)
        bb <- (dl[N]*sqrt(n2/4)-qnorm(1-alpha))/sqrt(1+n2*sl[N]^2/4)
        pD <- pnorm(bb)
        utility <- -N*c1 - n2*c2 + K *pD
        break
      }else if (d[t]==2){
        N <- t
        utility <- -N*c1
        break
      }
    }
    d_mat <- rbind(d_mat,d)
    N_vec <- c(N_vec,N)
    u_vec[nspa] <- utility
  }
  u_est[i] <- mean(u_vec)
}

# #########
# plotdata <- paragrid
# plotdata$u <- u_est
# ind.hat <- which.max(u_est)
# c.hat <- plotdata[ind.hat,1]
# b1.hat <- plotdata[ind.hat,2]
# b2.hat <- plotdata[ind.hat,3]
# u_est.hat <- plotdata[ind.hat,4]
# u_est.hat

# ggplot() +
#   geom_tile() +
#   stat_function(fun = function(s) b1.hat * s + c.hat, color="red")+
#   stat_function(fun = function(s) -b2.hat * s + c.hat, color="red") +
#   ggtitle(paste0("c = ",round(c.hat,2), ", b1 = ", round(b1.hat,2), ", b2 = ", round(b2.hat,2))) +
#   xlab("s_delta") + ylab("delta_bar")
# ggsave("figures/Eg2bound.jpg",height=5,width=6.5)
# 
# # heatmap
# plotdata <- paragrid
# plotdata$u <- u_est
# plotdata_temp <- plotdata[which(round(plotdata$b2,2)==round(b2.hat,2)),]
# ggplot(plotdata_temp, aes(c, b1, fill= u)) +
#   geom_tile() +
#   ggtitle(paste0("b2 = ",round(b2.hat,2)))
# ggsave(paste0("Eg2uest_cb1.jpg"),height=5,width=6.5)
# 
# # heatmap
# plotdata <- paragrid
# plotdata$u <- u_est 
# plotdata_temp <- plotdata[which(round(plotdata$b1,2)==round(b1.hat,2)),] 
# ggplot(plotdata_temp, aes(c, b2, fill= u)) + 
#   geom_tile() + 
#   ggtitle(paste0("b1 = ",round(b1.hat,2)))
# ggsave(paste0("Eg2uest_cb2.jpg"),height=5,width=6.5)
# 
# # heatmap
# plotdata <- paragrid
# plotdata$u <- u_est 
# plotdata_temp <- plotdata[which(round(plotdata$c,2)==round(c.hat,2)),] 
# ggplot(plotdata_temp, aes(b1, b2, fill= u)) + 
#   geom_tile() +
#   ggtitle(paste0("c = ",round(c.hat,2)))
# ggsave(paste0("Eg2uest_b1b2.jpg"),height=5,width=6.5)
# 

###############################
# Linear regression
# u = beta1 b1^2 + beta2 b2^2 + beta3 b3^2 +
#     beta4 b1b2 + beta5 b2c + beta6 b1c +
#     beta7 b1 + beta8 b2 + beta9 c + intercept
X <- matrix(NA, nrow=nrow(paragrid),ncol=9)
for (ind in 1:nrow(paragrid)){
  b1 <- paragrid$b1[ind]
  b2 <- paragrid$b2[ind]
  c <- paragrid$c[ind]
  X[ind,] <- c(b1^2,b2^2,c^2,b1*b2,b2*c,b1*c,b1,b2,c)
}
lm_coef <- lm(u_est ~ X)$coefficients
lm_coef
beta <- lm_coef[2:10]

# Maximum
A <- matrix(c(beta[1],beta[4]/2,beta[6]/2,
              beta[4]/2,beta[2],beta[5]/2,
              beta[6]/2,beta[5]/2,beta[3]),byrow=TRUE,ncol=3)
b <- matrix(beta[7:9],ncol=1)
maxi <- -solve(A)%*%b/2
# maxi
# lm_coef[1] - t(b)%*%solve(A)%*%b/4
#
b1_hat <- maxi[1]
b2_hat <- maxi[2]
c_hat <- maxi[3]
# 
# u_est_max_reg <- sum(lm_coef*c(1,b1_hat^2,b2_hat^2,c_hat^2,b1_hat*b2_hat,b2_hat*c_hat,b1_hat*c_hat,b1_hat,b2_hat,c_hat))
# u_est_max_reg

# ggplot() +
#   geom_tile() +
#   stat_function(fun = function(s) b1_hat * s + c_hat, color="red")+
#   stat_function(fun = function(s) -b2_hat * s + c_hat, color="red") +
#   ggtitle(paste0("c = ",round(c_hat,3), ", b1 = ", round(b1_hat,3), ", b2 = ", round(b2_hat,3))) +
#   xlab("s_delta") + ylab("delta_bar")
# ggsave("figures/Eg2bound_reg.jpg",height=5,width=6.5)


#####################
## plot first 10 spaghetti
dl.v <- sl.v <- num.v <- c()
for (i in 1:10){
  dl.v <- c(dl.v,spagh$dl[i,],NA)
  sl.v <- c(sl.v,spagh$sl[i,],NA)
  num.v <- c(num.v,rep(as.character(i),(Nmx+1)))
}

tracedata <- data.frame(dl=dl.v,sl=sl.v,num=num.v) %>% 
  mutate(dl=pmin(pmax(-1.5, dl), 2.5)) %>%  # truncate for visualization
  mutate(sl=pmin(pmax(0.0, sl), 1.0))  # truncate for visualization
# %>% 
#   mutate(act = case_when(
#     dl > b1_hat * c(tail(sl, -1), NA) + c_hat ~ "2",
#     dl < - b2_hat * c(tail(sl, -1), NA) + c_hat ~ "1",
#     TRUE ~ "0"
#   )) %>% 
#   mutate(alpha=if_else(act == "0", 1.0, 0.3))


g = tracedata %>% 
  # na.omit() %>% 
  ggplot(aes(x=sl, y=dl, color=as.factor(num))) +
  # geom_point() +
  geom_segment(
    aes(
      xend=c(tail(sl, n=-1), NA),
      yend=c(tail(dl, n=-1), NA)
    ),
    arrow=arrow(length=unit(0.12,"cm")),
    alpha=0.6,
    size=0.8
  ) +
  # stat_function(fun = function(s) b1_hat * s + c_hat, color="red")+
  # stat_function(fun = function(s) -b2_hat * s + c_hat, color="red") +
  xlim(c(0.1, sqrt(sig2_0))) + ylim(c(-1.5,2.5)) +
  labs(
    x=TeX("Posterior std. dev. ($s_{\\delta}$)"),
    y=TeX("Posterior mean  ($\\bar{\\delta}_{95}$)")
  ) +
  theme_cowplot() +
  theme(legend.position="none") +
  scale_color_hue(l=40, c=40)
g
  
#ggtitle(paste0("c = ",round(c_hat,3), ", b1 = ", round(b1_hat,3), ", b2 = ", round(b2_hat,3)))
ggsave("results/bsd_ex2/Eg2spagh.jpg",height=4.5,width=6.5)


grid = expand.grid(
  dl=seq(-1.5, 2.5, length.out=500),
  sl=seq(0.01, 1.0, length.out=500)
) %>% 
  mutate(act = case_when(
    dl > b1_hat * c(tail(sl, -1), NA) + c_hat ~ "2",
    dl < - b2_hat * c(tail(sl, -1), NA) + c_hat ~ "1",
    TRUE ~ "0"
  ))


xlo = min(tracedata$sl, na.rm=TRUE)
xhi = max(tracedata$sl, na.rm=TRUE)

ggplot(grid) +
  scale_fill_manual(values=c("white", "grey", "black")) +
  geom_tile(aes(x=sl, y=dl, fill=act)) + 
  labs(
    fill="Decisions",
    x=TeX("Posterior std. dev. ($s_{\\delta}$)"),
    y=TeX("Posterior mean  ($\\bar{\\delta}_{95}$)")
  ) +
  xlim(c(xlo, xhi)) + ylim(c(-1.5,2.5)) +
  geom_segment(
    aes(
      x=sl,
      y=dl,
      xend=c(tail(sl, n=-1), NA),
      yend=c(tail(dl, n=-1), NA),
      color=num,
    ),
    alpha=0.5,
    arrow=arrow(length=unit(0.12,"cm")),
    size=0.8,
    data=tracedata
  ) +
  theme_cowplot() +
  scale_color_hue(l=40, c=40) +
  theme_cowplot() +
  theme(
    legend.position="top",
    legend.margin=margin(0,0,0,0)
  ) +
  guides(color=FALSE)
# +
#   stat_function(fun = function(s) b1_hat * s + c_hat, color="red")+
#   stat_function(fun = function(s) -b2_hat * s + c_hat, color="red")
ggsave("results/bsd_ex2/bound.jpg",height=4,width=5)
write_csv(tracedata, "results/bsd_ex2/tracedata.csv")

## examples of dose responde functions

# set.seed(110105)

mu0 = c(1/2, 1)
s0 = c(1, 1)

simulate = function(i) {
  b <- rnorm(1, mean=mu0[1],sd=s0[1])
  e <- pmax(0.1, rnorm(1,mean=mu0[2],sd=s0[2]))
  x = seq(0, 10, length.out=100)
  y = b * x / (e + x)
  return (tibble(num=i, b=b, e=e, x=x, y=y))
}

set.seed(110104)

sims = map(1:6, simulate) %>%
  bind_rows %>%
  mutate(pars=sprintf("%02d b=%.2f, q=%.2f", num, b, e))

ggplot(sims) +
  geom_line(aes(x=x, y=y)) +
  geom_hline(yintercept=0, lty=2) +
  scale_x_continuous(n.breaks=3) +
  facet_wrap(~ pars, ncol=3, labeller = labeller(pars=function(l) str_sub(l, 4))) +
  theme_cowplot() +
  theme(
    strip.background = element_rect(fill="white", color="black"),
    strip.text = element_text(size=10),
    axis.text = element_text(size=10),
    axis.title = element_text(size=10)
  ) +
  labs(x="Dose (x)", y="Response (y)")

# 
ggsave("results/bsd_ex2/Eg2_prior_dose_respose_samples.png", width=4, height=3.3)
