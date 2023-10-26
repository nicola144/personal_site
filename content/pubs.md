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

</style>


- [***Adaptive importance sampling for heavy-tailed distributions via α-divergence minimization***](https://arxiv.org/pdf/2310.16653.pdf); <span style="color: orange;">Branchini, Nicola</span><sup style="color: orange;">♦</sup> and <span style="color: orange;">Guilmeau, Thomas</span><sup style="color: orange;">♦</sup> and Chouzenoux, Emilie and Elvira, Víctor. *Preprint;submitted*. <span style="font-size: 0.8em; color: grey;">(♦ indicates equal contribution)</span>


- [***Generalized self-normalized importance sampling***](https://www.dropbox.com/s/scm7vnn74i5inrs/gensnis_poster%20%286%29.pdf?dl=0); Branchini, Nicola and Elvira, Víctor. (**Submitted; presented at MCM 2023**; **link to BayesComp2023 poster**).

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

