# Library
library(tidyverse)
library(cowplot)

########### 
## Decision Boundaries
## t \in 1,...,Tmax
## p1 = c*\sqrt{t}/\sqrt{T}
## p2 = (c-1)*\sqrt{t}/\sqrt{T} +1 
## Find optimal c on grid c \in (0,1),

# Decisions
# d=0,1,2 (Continue, Terminate theta=0.4, Terminate theta=0.6)

# Input
Tmax <- 50 # Max number of pats
a0 <- 0.5 # Prior Pr(theta=0.4)=Pr(theta=0.6)=0.5
K <- 100 # Prespicified parameter in the utility function

############ Functions
# Function to select decision in each step
Deci <- function(c,p,t){
  if (p < c*sqrt(t-1)/sqrt(Tmax-1)){
    d <- 1
  }else if (p <= ((c-1)*sqrt(t-1)/sqrt(Tmax-1) + 1) ){
    d <- 0 
  }else{
    d <- 2
  }
  return(d)
}

SimOne <- function(Tmax,c){
  theta <- ifelse(runif(1)<a0,0.4,0.6)
  y <- p <- d <-rep(NA,Tmax)
  for (t in 1:(Tmax)){
    # t \in 1,...,Tmax
    N <- t
    y[t] <- yt <- ifelse(runif(1)<theta,1,0)
    p[t] <- pt <- mean(y[1:t])
    d[t] <- dt <- Deci(c,pt,t)
    if (!dt==0){
      # Stop Trial
      break
    }
  }
  # utility
  if ((theta == 0.4 & d[N] == 1)||(theta == 0.6 & d[N] == 2)){
    u <- -N
  }else{
    u <- -N-K
  }
  onesim <- list(N=N, theta=theta, y=y, d=d, u=u)
  return(onesim)
}

set.seed(123)
Nsim <- 10000
Ngrid <- 500
cgrid <- seq(0,1,length.out=Ngrid)
uest <- rep(NA,Ngrid)
# grid c
for (i in 1:Ngrid){
  c <- cgrid[i]
  u <- c()
  for (k in 1:Nsim){
    u <- c(u,SimOne(Tmax,c)$u)
  }
  uest[i] <- mean(u)
  cat(i,"\t")
}


plotdata <- data.frame(cgrid=cgrid[2:(Ngrid-1)],uest=uest[2:(Ngrid-1)])
# Plot
ggplot(plotdata, aes(x=cgrid, y=uest)) +
  geom_line() +
  theme_cowplot() +
  labs(y="Expected utility", x="Decision boundary parameter (c)")
  
ggsave("cgridrough.jpg",height=4,width=6)

ind_low <- which(cgrid>0.4)[1]
ind_high <- which(cgrid<0.6)[length(which(cgrid<0.6))]

## Linear regression
u_y <- uest[ind_low:ind_high]
cgrid_cut <- cgrid[ind_low:ind_high]
X <- cbind(cgrid_cut,cgrid_cut^2)

lm_coef <- lm(u_y ~ X)$coefficients
lm_coef
maxi <- -lm_coef[2]/lm_coef[3]/2
c_hat <- maxi

cat("Optimal c is", round(c_hat,3))

data = read_csv("./logs_ex1/ex1_uhat_bsd.csv")
ggplot(data, aes(t, pt, fill= as.character(dstar))) + 
  geom_tile() +
  scale_fill_manual(values = c("white", "grey", "black"), name = "Decisions") +
  stat_function(fun = function(t) c_hat*sqrt(t-1)/sqrt(Tmax-1), color="red")+ 
  stat_function(fun = function(t) ((c_hat-1)*sqrt(t-1)/sqrt(Tmax-1) + 1), color="red") +
  theme_cowplot() +
  labs(fill="Decisions", x=TeX("Step ($t$)"), y=TeX("Running mean ($p_t$)")) +
  ylim(c(0,1))
ggsave("bound_gridc.jpg",height=4,width=6)



