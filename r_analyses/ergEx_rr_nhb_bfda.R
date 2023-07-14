# load necessary packages

library(devtools)

# Installing the specific version of BFDA we use in the analysis (BFDA is non on CRAN so cannot be installed using environment.yml)
if (!require('BFDA')) install_github("nicebread/BFDA@f1886176a8146576bc76da4c073867faa3397bad", subdir="package", 
                                     upgrade=FALSE); library('BFDA')
library(doParallel)

# Set up parallel processing
cores <- detectCores()
cl <- makeCluster(cores - 1)
registerDoParallel(cl)

# Clear console and plot windows
#cat("\014")
#dev.off()

#Initialise variables
nmin <- 20
nmax <- 150
nstep <- 2
bfbound <- 10
nsims <- 10
ntraj <- 1

#H0: d_h1 >= d_h0
sim.H0 <- BFDA.sim(expected.ES=0.0, type="t.paired", prior=list("Cauchy", list(prior.location=0, prior.scale=sqrt(2)/2)), n.min=nmin, n.max=nmax, 
                   alternative="greater", boundary=1/10, B=nsims, verbose=TRUE, cores=cores-1, stepsize = nstep)
BFDA.analyze(sim.H0, design="sequential", n.min=nmin, n.max=nmax, boundary=bfbound)

pdf('plot1.pdf', width=10, height=10)
print(plot(sim.H0, n.min=50, n.max=nmax, boundary=bfbound, n.trajectories = ntraj))
dev.off()

#H1: d_h1 < d_h0
sim.H1 <- BFDA.sim(expected.ES=0.5, type="t.paired", prior=list("Cauchy", list(prior.location=0, prior.scale=sqrt(2)/2)), n.min=nmin, n.max=nmax, 
                   alternative="greater", boundary=010, B=nsims, verbose=TRUE, cores=cores-1, stepsize = nstep)
BFDA.analyze(sim.H1, design="sequential", n.min=nmin, n.max=nmax, boundary=bfbound)

pdf('plot2.pdf', width=10, height=10)
print(plot(sim.H1, n.min=50, n.max=nmax, boundary=bfbound, n.trajectories = ntraj))
dev.off()

#analyse threshold hitting events 
evDens(BFDA.H0=sim.H0, BFDA.H1=sim.H1, n=nsims, boundary=c(1/bfbound, bfbound))

#sample size determination
SSD(sim.H0, alpha=.05, boundary=c(10))
SSD(sim.H1, power=.90, boundary=c(10))

# Stop parallel processing
stopCluster(cl)
