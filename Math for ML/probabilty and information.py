from numpy import *
from matplotlib.pyplot import *
rcParams.update({'font.size': 12})

p=[1e-7*2**k for k in range(0,int(log(1e7)/log(2))+1)]
p.append(1)
p=array(p)
figure(figsize=(14,6))
subplots_adjust(hspace=.5)
subplot(211)
plot(p,1/p,'.-k',label="$1/p$");title("$I\\approx 1/p$ is hard to work with due to its strong exponentiality, e.g.");legend();grid();xlabel("probability $p$ ");ylabel("$I\\approx 1/p$")
subplot(212)
title("Logarithm makes it easier + also see and think about Example L-8.1" )
plot(p,log(1/p)/log(2),'.-b',label="$h(p)=I(p)= -log_2(p)$ [bit] ...Shannon");legend();grid();xlabel("probability $p$")
plot(p,log(1/p),'.-m',label="$h(p)=I(p)= -log(p)$ [nat] ...natural");legend();grid();xlabel("probability $p$")
plot(p,log(1/p)/log(10),'.-c',label="$h(p)=I(p)= -log_{10}(p) $ [hart] ... Hartley");legend();grid();xlabel("probability $p$");ylabel("$I(p)$  [bit] [nat] [hart]")
show()
