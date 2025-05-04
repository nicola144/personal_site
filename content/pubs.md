---
title: "Publications, preprints & working papers"
disqus: false
---

<style>
  .progress-bar {
    position: relative;
    display: flex;
    justify-content: center;
    padding-top: 20px;
  }

  progress {
    background-color: white;
    width: 60%;
    border-radius: 10px;
  }

  progress::-webkit-progress-bar {
    background-color: white;
    border-radius: 10px;
  }

  progress::-webkit-progress-value {
    background-color: orange;
    border-radius: 10px;
  }

  progress::-moz-progress-bar {
    background-color: orange;
    border-radius: 10px;
  }

  .progress-label {
    position: absolute;
    left: 50%;
    top: 5%;
    transform: translate(-50%, -50%);
    font-size: 25px;
    font-weight: bold;
    color: orange;
  }
    span.emoji {
    font-size: 40px;
    margin-top: -15px;
  }

  /* Button styling */
  button.collapsible {
    background-color: #f8f8f8;
    color: #555;
    cursor: pointer;
    padding: 6px 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 0.85em;
    transition: all 0.3s ease;
    margin-bottom: 10px;
  }
  
  button.collapsible:hover {
    background-color: #eee;
    color: #222;
  }
  
  button.collapsible:focus {
    outline: none;
  }

</style>

- [***Towards Adaptive Self-Normalized Importance Samplers***](https://arxiv.org/abs/2505.00372); Branchini, Nicola and Elvira, Víctor. *In: 2025 IEEE Statistical Signal Processing Workshop*
<button type="button" class="collapsible">Details about paper</button>
<div class="content">
<p>
 The TLDR; To estimate µ = E_p[f(θ)] when p's normalizing constant is unknown, instead of doing MCMC on p(θ) or even p(θ)|f(θ)|, or learning a parametric q(θ), we try MCMC directly on p(θ)|f(θ)- µ|, which is the asymptotic-variance minimizing proposal. 
 Note: we cannot do MCMC straightforwardly, as p(θ)|f(θ)- µ| cannot be evaluated - it contains µ, the quantity of interest ! So, we propose a simple iterative scheme that works: initial estimate µ₀ ; run a chain on the *approximation* p(θ)| f(θ)- µ₀ |; estimate µ again with SNIS, and keep iterating. I'm quite excited about extending this work. 
</p>
</div>

- [***Scalable Expectation Estimation with Subtractive Mixture Models (preprint)***](https://arxiv.org/abs/2503.21346); <span style="color: orange;">Zellinger, Lena</span><sup style="color: orange;">♦</sup> and <span style="color: orange;">Branchini, Nicola</span><sup style="color: orange;">♦</sup> and Elvira, Víctor, and Vergari, Antonio. <span style="font-size: 0.8em; color: orange;">(♦equal contribution.)</span>
<button type="button" class="collapsible">Details about paper</button>
<div class="content">
<p>
 Importance sampling with mixture models is all over the place (even where you don't see it). Subtractive mixture models - MMs with negative weights - are super cool and can model complex distributions more efficiently. It'd be great to use them for IS, but sampling from them is a pain. We propose an estimator that exploits that a SMM is a difference of two regular MMs, so that we can do IS and scale in higher dimension (note: sampling from an SMM requires costly autoregressive inverse transform sampling). 
</p>
</div>

- [***The role of tail dependence in estimating posterior expectations***](https://openreview.net/forum?id=Zxk07UdWEy); Branchini, Nicola and Elvira, Víctor. *In NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty*.

- [***Generalized self-normalized importance sampling (preprint)***](https://arxiv.org/abs/2406.19974); Branchini, Nicola and Elvira, Víctor. [**Video from SMC 2024**](https://www.youtube.com/watch?v=tG9mjp6GgtE&list=PLUbgZHsSoMEUq6vqSLjwuXfrGDBNLbZRu&index=11); [**Xi'an's comments in his blog**](https://xianblog.wordpress.com/2024/06/05/6th-workshop-on-sequential-monte-carlo-methods-2/).

<button type="button" class="collapsible">Details about paper</button>
<div class="content">
<p>
  The self-normalized IS estimator is widely used to estimate expectations with intractable normalizing constants, for example, in Bayesian leave-one-out cross validation or likelihood free inference. In this paper, we propose a framework to understand when SNIS works and when it does not, with a generalization that allows us to overcome its limitations, with connections to continuous optimal transport. See paper abstract for more info. 
</p>
</div>

- [***Adaptive importance sampling for heavy-tailed distributions via α-divergence minimization***](https://proceedings.mlr.press/v238/guilmeau24a.html); <span style="color: orange;">Guilmeau, Thomas</span><sup style="color: orange;">♦</sup> and <span style="color: orange;">Branchini, Nicola</span><sup style="color: orange;">♦</sup> and Chouzenoux, Emilie and Elvira, Víctor. *In 27th Conference on Artificial Intelligence and Statistics (AISTATS), Proceedings of Machine Learning Research, 2024*. <span style="font-size: 0.8em; color: orange;">(♦equal contribution.)</span>

<button type="button" class="collapsible">Details about paper</button>
<div class="content">
<p>
  Many adaptive IS (and some VI) methods are based on matching the moments of a target distributions. When the target has heavy tails, these moments can be undefined or their estimation can have high variance. We propose an AIS method that overcomes this by matching the moments of a (lighter tailed) modified target, which is exponentiated to a power alpha. Despite this, the procedure actually minimizes the alpha-divergence between the proposal and the true target. Note: many previous works propose AIS methods with heavy-tailed *proposals*, but not necessarily suitable for heavy-tailed *targets*.
</p>
</div>


- [***Variational Resampling***](https://proceedings.mlr.press/v238/kviman24a.html); Kviman, Oskar and Branchini, Nicola and Elvira, Víctor and Lagergren, Jens. *In 27th Conference on Artificial Intelligence and Statistics (AISTATS), Proceedings of Machine Learning Research, 2024*. 

<button type="button" class="collapsible">Details about paper</button>
<div class="content">
<p>
  A very neat idea stemming from Oskar's Master's thesis (he's impressive, isn't he ?); when we resample in PFs, we usually would like the resulting equally-weighted distribution of the resampled particles to be ``close'' in some sense to the distribution before resampling (which was unequally-weighted, in general). 
  Usually, resampling schemes enforce this by saying that the number of times a particle gets replicated is, on average, equal to its weight in the pre-resampling distribution. What we do here instead is to optimize the number of times a particle gets replicated so as to minimize a divergence between the post-resampling distribution and the pre-resampling distribution directly ! With a very smart algorithm again entirely due to Oskar. 
</p>
</div>

- [***Causal optimal transport of abstractions***](https://proceedings.mlr.press/v236/felekis24a.html); Felekis, Yorgos and Zennaro, Fabio and Branchini, Nicola and Damoulas, Theodoros. *In 3rd Conference on Causal Learning and Reasoning (CLeaR 2024)*. 
<button type="button" class="collapsible">Details about paper</button>
<div class="content">
<p>
  The task of causal abstraction involves finding a mapping (a measurable transport map) between structural causal models (SCMs) and their corresponding "abstracted versions", which can be simplified or coarser SCMs (fewer variables or different functional relationships). We consider the problem of learning causal abstractions from data. We propose a framework that does so without specifying parametric relationships for the SCM functions. The method involves a multimarginal OT problem (as many marginals as there are considered interventions (not really, but roughly to get the idea)) with soft constraints and a cost function econding knowledge of the underlying causal DAGs. Nicely, the soft constraints have a do-calculus interpretation. 
</p>
</div>

- [***An adaptive mixture view of particle filters***](https://www.aimsciences.org/article/doi/10.3934/fods.2024017); Branchini, Nicola and Elvira, Víctor. *FoDS (Foundations of Data Science)*. 

<button type="button" class="collapsible">Details about paper</button>
<div class="content">
<p>
  Coming !
</p>
</div>



<!-- <div class="progress-bar">
  <span class="emoji">🍳</span> <progress value="95" max="100"></progress>
  <div class="progress-label">95%</div>  <span class="emoji">🍳</span>
</div>
 -->

<!-- - [***On dependence and bias in importance sampling for high dimensional test functions***](https://proceedings.mlr.press/v161/branchini21a.html); Branchini, Nicola and Elvira, Víctor. (**In preparation**; **no link**).

<div class="progress-bar">
  <span class="emoji">🍳</span> <progress value="40" max="100"></progress>
  <div class="progress-label">40%</div>  <span class="emoji">🍳</span>
</div> -->


- [***Causal Entropy Optimization***](https://proceedings.mlr.press/v206/branchini23a.html); Branchini, Nicola and Aglietti, Virginia and Dhir, Neil and Damoulas, Theodoros. In *26th Conference on Artificial Intelligence and Statistics (AISTATS), Proceedings of Machine Learning Research, 2023*.

<button type="button" class="collapsible">Details about paper</button>
<div class="content">
<p>
  In this paper, we studied the problem of "causal global optimization": finding the optimum intervention that is the minimizer of several causal effects (that is, we consider possibly intervening on many different subset of variables). When the underlying causal graph is not known, the first step is studying what happens if we assume any one of the possible graphs is the true one, and run "CBO"- causal Bayesian optimization - as normal. We studied what the effect of this kind of incorrect causal assumption is for optimization purposes. Further, since in many cases the underlying function can be optimized efficiently even if the graph is not fully known, we designed an acquisition function that automatically trades-off optimization of the effect and structure learning.  
</p>
<img src="/ceo.svg" width="1000" height="300">
</div>

- [***Optimized Auxiliary Particle Filters: adapting mixture proposals via convex optimization***](https://proceedings.mlr.press/v161/branchini21a.html), Branchini, Nicola and Elvira, Víctor. In *37th Conference on Uncertainty in Artificial Intelligence (UAI), Proceedings of Machine Learning Research, 2021*.

<button type="button" class="collapsible">Details about paper</button>
<div class="content">
<p>
  In this paper we wanted to improve on the Auxiliary Particle Filter (APF), which is thought for estimating the likelihood in sequential latent variable models with very informative observations. This algorithm however still has severe drawbacks; among some, the resampling weights are chosen independently, i.e. each particle chooses its own without "knowing" what the others are doing.
  We devise a new way to optimize these resampling weights by viewing them as mixture weights of an importance sampling mixture proposal. It turns out that choosing mixture weights in order to minimize the resulting empirical variance of the importance weights leads to a convex optimization problem.
</p>
<a href="https://underline.io/speakers/119464-nicola-branchini">Video and slides from UAI</a>

<img src="/eq_oapf.svg" width="1000" height="300">
</div>

