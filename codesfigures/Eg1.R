# Library
library(ggplot2)
library(cowplot)
library(latex2exp)

# Input
Tmax <- 50 # Max number of pats
Nsim <- 500 # Number of simulated trials in each case
a0 <- 0.5 # Prior Pr(theta=0.4)=Pr(theta=0.6)=0.5
K <- 100
# Decisions
# d=0,1,2 (Continue, Terminate theta=0.4, Terminate theta=0.6)

######################## Functions #######################
# Simulate One trial, terminate at pat (N=Tmax), d=(0,0,...,0,dT=1 or 2)
SimOneTrial <- function(Tmax,a0,dT=1){
  onesim <- list(N=Tmax, theta=NA, y=rep(NA,Tmax), d=c(rep(0,Tmax-1),dT))
  theta <- ifelse(runif(1)<a0,0.4,0.6)
  onesim$theta <- theta
  for (t in 1:Tmax){
    yt <- ifelse(runif(1)<theta,1,0)
    onesim$y[t] <- yt
  }
  return(onesim)
}

# Generate data based on posterior if no trajectory in grid (t,pt), one trial
MissTmax <- function(t,N,dT,a0,ptindex){
  abt <- c(a0*2^(ptindex-1)*3^((t-1)-(ptindex-1)),(1-a0)*3^(ptindex-1)*2^((t-1)-(ptindex-1)))
  at <- abt[1]/(abt[1]+abt[2])
  onetrial <- SimOneTrial(Tmax=N-t+1,at,dT=dT)
  onetrial$N <- N
  onetrial$y <- c(rep(1,ptindex-1),rep(0,((t-1)-(ptindex-1))),onetrial$y)
  onetrial$d <- c(rep(0,t-1),onetrial$d)
  return(onetrial)
}
# MissTmax <- function(Tmax,a0,dT,ptindex){
#   onetrial <- list()
#   onetrial$N <- Tmax
#   onetrial$y <- c(rep(1,ptindex-1),rep(0,((Tmax-1)-(ptindex-1))))
#   onetrial$d <- c(rep(0,Tmax-1),dT)
#   abt <- c(a0*0.4^(ptindex-1)*0.6^((Tmax-1)-(ptindex-1)),(1-a0)*0.6^(ptindex-1)*0.4^((Tmax-1)-(ptindex-1)))
#   abt <- abt/sum(abt)
#   onetrial$theta <- ifelse(runif(1)<abt[1],0.4,0.6)
#   return(onetrial)
# }


##########################
# Set seed
set.seed(123)
# Simulate (Nsim) Trials in each case, terminate at time N=1,...,Tmax, dN=1 or 2
Trials <- list()
ntrial <- 1
for (t in 2:Tmax){
  for (dT in 1:2){
    for (sim in 1:Nsim){
      Trials[[ntrial]] <- SimOneTrial(t,a0,dT=dT)
      ntrial <- ntrial + 1
    }
  }
}

#####################
# plot spaghetti
# set.seed(1234)
# Ntrials <- length(Trials)
# pt <- t <- num <- c()
# for (ii in 1:10){
#   dataone <- Trials[[Ntrials-sample(100,1)]]
#   pttemp <- rep(0,50)
#   for (i in 1:50){
#     pttemp[i] <- sum(dataone$y[1:i])/i
#   }
#   pt <- c(pt,pttemp,NA)
#   t <- c(t,1:50,NA)
#   num <- c(num,rep(as.character(ii),51))
# }
# tracedata <- data.frame(pt=pt,t=t,num=num)

library(tidyverse)
ntrails = 10
p = sample(c(0.4, 0.6), size=ntrails, replace=TRUE)
p = matrix(rep(p, 50), ntrails, 50)
aux = runif(Tmax * ntrails)

tracedata =  as.integer(aux < p) %>% 
  matrix(nrow=ntrails, ncol=Tmax) %>% 
  apply(1, cummean) %>% 
  t %>% 
  as.data.frame() %>% 
  mutate(num=1:n()) %>% 
  pivot_longer(cols=-num, names_to="x", values_to="y") %>% 
  mutate(x=as.integer(str_remove(x, "V"))) %>% 
  arrange(num, x) %>% 
  group_by(num) %>% 
  mutate(xend=lead(x, 1), yend=lead(y, 1)) %>% 
  ungroup() %>% 
  mutate(num=as.factor(num))

tracedata %>% 
  na.omit() %>%  # only if using arrows
  ggplot() +
  # geom_point() +
  geom_segment(
    aes(x=x, y=y, xend=xend, yend=yend, color=num),
    arrow=arrow(length=unit(0.12,"cm")),
    alpha=0.6,
    size=0.8
  ) +
  # geom_line(
  #   # alpha=0.5,
  #   aes(x=t, y=pt, color=as.factor(num)) +
  #   size=0.8
  # ) +
  # stat_function(fun = function(s) b1 * s + c, color="red")+ 
  # stat_function(fun = function(s) -b2 * s + c, color="red") +
  #xlim(c(0,1)) +
  labs(x=TeX("Step ($t$)"), y=TeX("Running mean ($p_t$)")) +
  theme_cowplot() +
  theme(legend.position="none") +
  scale_color_hue(l=40, c=40)
ggsave("Eg1spagh.jpg",height=4.5,width=6.5)

####################
stop <- FALSE
nextstop <- FALSE
iter <- 0
while (stop == FALSE){
  if (nextstop == TRUE) {stop <- TRUE}
  iter <- iter + 1
  cat("iter",iter,"\n")
  
  # Simulated trials passing through (d=0,1,2,t,pt)
  A <- rep(list(rep(list(NULL), Tmax)),3)
  # Grid of (t,pt=sum_{i=1}^(t-1)y_i)
  for (t in 2:Tmax){
    ptgrid <- (0:(t-1))/(t-1)
    for (d in 1:3){
      A[[d]][[t]] <- rep(list(NULL),length(ptgrid))
    }
  }
  for (i in 1:length(Trials)){
    for (t in 2:Trials[[i]]$N){
      pt <- sum(Trials[[i]]$y[1:(t-1)])
      ptindex <- pt + 1
      dindex <- Trials[[i]]$d[t] + 1
      A[[dindex]][[t]][[ptindex]] <- c(A[[dindex]][[t]][[ptindex]],i)
    }
  }
  
  nextstop <- TRUE
  # Generate data based on posterior if no trajectory in grid (t,pt)
  for (t in 2:Tmax){
    if (t== Tmax){
      for (dindex in 2:3){
        for (ptindex in 1:t){
          if (is.null(A[[dindex]][[t]][[ptindex]])){
            nextstop <- FALSE
            if (dindex == 1){
              for (N in t:(Tmax)){
                for (rep in 1:(Nsim/100)){
                  for (dT in 1:2){
                    Trials[[ntrial]] <- MissTmax(t,N,dT,a0,ptindex)
                    ntrial <- ntrial + 1
                  }
                }
                
              }
            }else{
              N <- t
              for (rep in 1:(Nsim/100)){
                Trials[[ntrial]] <- MissTmax(t,N,dT=dindex-1,a0,ptindex)
                ntrial <- ntrial + 1
              }
            }
            
            
          }
        }
      }
    }else{
      for (dindex in 1:3){
        for (ptindex in 1:t){
          if (is.null(A[[dindex]][[t]][[ptindex]])){
            nextstop <- FALSE
            if (dindex == 1){
              for (N in t:(Tmax)){
                for (rep in 1:(Nsim/100)){
                  for (dT in 1:2){
                    Trials[[ntrial]] <- MissTmax(t,N,dT,a0,ptindex)
                    ntrial <- ntrial + 1
                  }
                }
                
              }
            }else{
              N <- t
              for (rep in 1:(Nsim/100)){
                Trials[[ntrial]] <- MissTmax(t,N,dT=dindex-1,a0,ptindex)
                ntrial <- ntrial + 1
              }
            }
            
            
          }
        }
      }
    }
    
  }
  
  
  if (stop == TRUE) {
    # Store uhat, dstar, ustar
    uhat <- rep(list(rep(list(NA), Tmax)),3)
    dstar <- ustar <- rep(list(NA), Tmax)
    for (t in 2:Tmax){
      ptgrid <- (0:(t-1))/(t-1)
      for (d in 1:3){
        uhat[[d]][[t]] <- rep(list(NA),length(ptgrid))
      }
      dstar[[t]] <- ustar[[t]] <- rep(list(NA),length(ptgrid))
    }
    
    # Find the trajectories passing through (t,pt) A_tj = {i: S_ti=(t,p_ti)=j}
    for (dindex in 2:3){
      for (t in 2:Tmax){
        for (ptindex in 1:t){
          idata <- A[[dindex]][[t]][[ptindex]]
          if (!is.null(idata)){
            uarray <- c()
            for (i in idata){
              theta <- Trials[[i]]$theta
              if (theta == (0.4*(dindex==2)+0.6*(dindex==3))){
                u <- -t
              }else{
                u <- -t-K
              }
              uarray <- c(uarray,u)
            }
            uhat[[dindex]][[t]][[ptindex]] <- mean(uarray)
          }
          
        }
      }
    }
    
    # Estimate expected utility uhat at t = Tmax
    for (ptindex in 1:Tmax){
      dstar[[Tmax]][[ptindex]] <- which.max(c(uhat[[1]][[Tmax]][[ptindex]],uhat[[2]][[Tmax]][[ptindex]],uhat[[3]][[Tmax]][[ptindex]]))
      ustar[[Tmax]][[ptindex]] <- uhat[[dstar[[Tmax]][[ptindex]] ]][[Tmax]][[ptindex]]
    }
    
    # iteration t=(Tmax-1):2
    dindex <- 1
    for (t in (Tmax-1):2){
      for (ptindex in 1:t){
        idata <- A[[dindex]][[t]][[ptindex]]
        if (!is.null(idata)){
          uarray <- c()
          for (i in idata){
            ptplus1 <- sum(Trials[[i]]$y[1:t])
            k <- ptplus1 + 1
            u <- ustar[[(t+1)]][[k]]
            uarray <- c(uarray,u)
          }
          uhat[[dindex]][[t]][[ptindex]] <- mean(uarray)
          dstar[[t]][[ptindex]] <- which.max(c(uhat[[1]][[t]][[ptindex]],uhat[[2]][[t]][[ptindex]],uhat[[3]][[t]][[ptindex]]))
          ustar[[t]][[ptindex]] <- uhat[[dstar[[t]][[ptindex]] ]][[t]][[ptindex]]
          
        }
      }
    }
  }
  
}


#Plot heatmap for estimated utilities
gridstep <- 0.01
data <- expand.grid(pt=seq(0,1,gridstep),t=2:Tmax)
dstardata <- c()
for (t in 2:Tmax){
  for (ptindex in 1:length(seq(0,1,gridstep))){
    # When (k-1)/(t-1) <= ptgrid < (k)/(t-1), i.e. case pt= (k-1)/(t-1), dstardata = dstar[[t]][[k]]
    ptgrid <- data$pt[ptindex]
    k <- floor(ptgrid*(t-1)) + 1
    dstardata <- c(dstardata,dstar[[t]][[k]])
  }
}
data$dstar <- dstardata - 1

uhatdata0 <- c()
for (t in 2:Tmax){
  for (ptindex in 1:length(seq(0,1,gridstep))){
    # When (k-1)/(t-1) <= ptgrid < (k)/(t-1), i.e. case pt= (k-1)/(t-1), dstardata = dstar[[t]][[k]]
    ptgrid <- data$pt[ptindex]
    k <- floor(ptgrid*(t-1)) + 1
    uhatdata0 <- c(uhatdata0,uhat[[1]][[t]][[k]])
  }
}
data$uhat0 <- uhatdata0

uhatdata1 <- c()
for (t in 2:Tmax){
  for (ptindex in 1:length(seq(0,1,gridstep))){
    # When (k-1)/(t-1) <= ptgrid < (k)/(t-1), i.e. case pt= (k-1)/(t-1), dstardata = dstar[[t]][[k]]
    ptgrid <- data$pt[ptindex]
    k <- floor(ptgrid*(t-1)) + 1
    uhatdata1 <- c(uhatdata1,uhat[[2]][[t]][[k]])
  }
}
data$uhat1 <- uhatdata1

uhatdata2 <- c()
for (t in 2:Tmax){
  for (ptindex in 1:length(seq(0,1,gridstep))){
    # When (k-1)/(t-1) <= ptgrid < (k)/(t-1), i.e. case pt= (k-1)/(t-1), dstardata = dstar[[t]][[k]]
    ptgrid <- data$pt[ptindex]
    k <- floor(ptgrid*(t-1)) + 1
    uhatdata2 <- c(uhatdata2,uhat[[3]][[t]][[k]])
  }
}
data$uhat2 <- uhatdata2

# Heatmap 
ggplot(data, aes(t, pt, fill= as.character(dstar))) + 
  geom_tile() +
  scale_fill_manual(values = c("white", "grey", "black"), name = "Decisions")
ggsave("dstar.jpg",height=5,width=6.5)

b <- c(-150,0,100)
ggplot(data, aes(t, pt, fill= uhat0)) + 
  geom_tile() + 
  scale_fill_gradientn(limits = c(-150,100),
                       colours=c("black", "grey", "white"),
                       breaks=b, labels=format(b), name = "d=0")
ggsave("uhat0.jpg",height=5,width=6.5)
ggplot(data, aes(t, pt, fill= uhat1)) + 
  geom_tile() + 
  scale_fill_gradientn(limits = c(-150,100),
                       colours=c("black", "grey", "white"),
                       breaks=b, labels=format(b), name = "d=1")
ggsave("uhat1.jpg",height=5,width=6.5)
ggplot(data, aes(t, pt, fill= uhat2)) + 
  geom_tile() + 
  scale_fill_gradientn(limits = c(-150,100),
                       colours=c("black", "grey", "white"),
                       breaks=b, labels=format(b), name = "d=2")
ggsave("uhat2.jpg",height=5,width=6.5)

write_csv(data, "./logs_ex1/ex1_uhat_bsd.csv")
