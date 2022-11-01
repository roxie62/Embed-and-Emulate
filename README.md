# Embed-and-Emulate
This is the repo for the paper: Embed and Emulate: Learning to estimate parameters
of dynamical systems with uncertainty quantification.


## Abstract
This github project explores learning emulators for 
parameter estimation with uncertainty estimation of high-dimensional dynamical systems. 
We assume access to a computationally complex simulator that inputs a candidate parameter and outputs a corresponding multichannel time series. Our task is to accurately estimate a range of likely values of the underlying parameters. 
Standard iterative approaches necessitate running the simulator many times, which is computationally prohibitive.
We describe a novel framework for learning feature embeddings of observed dynamics jointly with an emulator that can replace high-cost simulators for parameter estimation. 
Leveraging a contrastive learning approach, our method (Embed & Emulate) exploits intrinsic data properties within and across parameter and trajectory domains. On a coupled 396-dimensional multiscale Lorenz 96 system, our method (Embed & Emulate) significantly outperforms a typical parameter estimation
method based on predefined metrics and a classical numerical simulator, and with only 1.19\% of the baseline's computation time.
Ablation studies highlight the potential of explicitly designing learned emulators for parameter estimation by leveraging contrastive learning.
