# load necessary packages
library(BFDA)
library(ggplot2)
library(devtools)
library(rngtools)
library(doParallel)

# Set up parallel processing
cores <- detectCores()
cl <- makeCluster(cores - 1)
registerDoParallel(cl)

# Clear console and plot windows
cat("\014")
dev.off()

#Initialise variables
nmin <- 20
nmax <- 150
nstep <- 1
bfbound <- 10
nsims <- 400
ntraj <- 100

#H0: eta_mul isnot> eta_add
sim.H0 <- BFDA.sim(expected.ES=0.0, type="t.paired", prior=list("Cauchy", list(prior.location=0, prior.scale=sqrt(2)/2)), n.min=nmin, n.max=nmax, alternative="greater", boundary=Inf, B=nsims, verbose=TRUE, cores=cores-1, stepsize = nstep)
BFDA.analyze(sim.H0, design="sequential", n.min=nmin, n.max=nmax, boundary=bfbound)
plot(sim.H0, n.min=50, n.max=nmax, boundary=bfbound, n.trajectories = ntraj)

#H1: eta_mul is > eta_add, (using 95% lower bound of effect size from pilot)
sim.H1 <- BFDA.sim(expected.ES=0.5, type="t.paired", prior=list("Cauchy", list(prior.location=0, prior.scale=sqrt(2)/2)), n.min=nmin, n.max=nmax, alternative="greater", boundary=Inf, B=nsims, verbose=TRUE, cores=cores-1, stepsize = nstep)
BFDA.analyze(sim.H1, design="sequential", n.min=nmin, n.max=nmax, boundary=bfbound)
plot(sim.H1, n.min=50, n.max=nmax, boundary=bfbound, n.trajectories = ntraj)

#analyse threshold hitting events 
evDens(BFDA.H0=sim.H0, BFDA.H1=sim.H1, n=nsims, boundary=bfbound)

#sample size determination
SSD(sim.H0, alpha=.05, boundary=c(10))
SSD(sim.H1, power=.90, boundary=c(10))

# Stop parallel processing
stopCluster(cl)
