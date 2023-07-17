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

# Initialise variables
nmin <- 25
nmax <- 150
nstep <- 1
bfbound <- 10
nsims <- 1000
ntraj <- 25
filename <- 'bfda_log.txt'
simH0_file <- 'simH0.RData'
simH1_file <- 'simH1.RData'
REDO <- TRUE

#H0: d_h1 >= d_h0
if (file.exists(simH0_file) & !REDO){
  load(simH0_file)
} else{
  sim.H0 <- BFDA.sim(expected.ES=0.0, type="t.paired", prior=list("Cauchy", list(prior.location=0, prior.scale=sqrt(2)/2)), n.min=nmin, n.max=nmax, 
                     alternative="greater", boundary=Inf, B=nsims, verbose=TRUE, cores=cores-1, stepsize = nstep, design = 'sequential')
  save(sim.H0, file=simH0_file)
}

sink(file = filename, type = c("output", "message"), split = FALSE)
BFDA.analyze(sim.H0, design="sequential", n.min=nmin, n.max=nmax, boundary=bfbound)
sink()

pdf('plot1.pdf', width=10, height=10)
print(plot(sim.H0, n.min=50, n.max=nmax, boundary=bfbound, n.trajectories = ntraj))
dev.off()

#H1: d_h1 < d_h0
if (file.exists(simH1_file) & !REDO){
  load(simH1_file)
} else{
  sim.H1 <- BFDA.sim(expected.ES=0.5, type="t.paired", prior=list("Cauchy", list(prior.location=0, prior.scale=sqrt(2)/2)), n.min=nmin, n.max=nmax, 
                     alternative="greater", boundary=Inf, B=nsims, verbose=TRUE, cores=cores-1, stepsize = nstep, design = 'sequential')
  save(sim.H1, file=simH1_file)
}
sink(file = filename, type = c("output", "message"), split = FALSE, append=TRUE)
BFDA.analyze(sim.H1, design="sequential", n.min=nmin, n.max=nmax, boundary=bfbound)
sink()

pdf('plot2.pdf', width=10, height=10)
print(plot(sim.H1, n.min=50, n.max=nmax, boundary=bfbound, n.trajectories = ntraj))
dev.off()

#analyse threshold hitting events 
#evDens(BFDA.H0=sim.H0, BFDA.H1=sim.H1, n=nsims, boundary=c(1/bfbound, bfbound))

#sample size determination
sink(file = filename, type = c("output", "message"), split = FALSE, append=TRUE)
SSD(sim.H0, alpha=.05, boundary=c(10))
SSD(sim.H1, power=.90, boundary=c(10))
sink()

unlink(filename)
# Stop parallel processing
stopCluster(cl)
