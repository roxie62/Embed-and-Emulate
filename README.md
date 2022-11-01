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

## Overview of our method

<img src="https://github.com/roxie62/Embed-and-Emulate/blob/main/plots/our_method.png" width="500" alt="drawing"/>
We propose our method, Embed and Emulate, to jointly learn feature embeddings and the emulator. 
Unlike the standard setup, to fit well in our problem, we design our emulator to “emulate’’ the low-dimensional embeddings written in this composite form, instead of high-dimensional dynamics. And our goal is to find parameters that live close to the observations in the embedding space.


<img src="https://github.com/roxie62/Embed-and-Emulate/blob/main/plots/our_method_clip.png" width="500" alt="drawing"/>

We leverage contrastive learning to capture **intra-domain** structural information to learn meaningful embeddings.

Between the **inter-domains** of parameter and trajectory, we use CLIP-wise loss to align the metric space of the “emulator” and the embedding network. As shown in the diagram, we define the embeddings of the parameter and its paired trajectory as “positive pairs”, and our goal is to maximize the similarity between “positive” pairs on the diagonal, while minimizing the unmatched “negative” pairs off the diagonal.

## Experimental Results

<img src="https://github.com/roxie62/Embed-and-Emulate/blob/main/plots/lorenz96_results.png" width="500" alt="drawing"/>

## Acknowledgement

We would like to thank [Automatic Posterior Transformation for Likelihood-free Inference](http://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf) for open-source code.
We would like to thank [Neural Approximate Sufficient Statistics for Implicit Models]([http://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf](https://arxiv.org/pdf/2010.10079.pdf)) for open-source code.
