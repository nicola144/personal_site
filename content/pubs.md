---
title: "Publications, preprints & working papers"
disqus: false
---

<div class="pub-page">

<div class="pub-section-title">Selected papers</div>

<div class="pub-featured">
  <article class="pub-item pub-featured-item" data-groups="monte-carlo optimal-transport">
    <div class="pub-venue">Preprint</div>
    <h3 class="pub-title"><a href="https://arxiv.org/abs/2406.19974">Generalized self-normalized importance sampling</a></h3>
    <p class="pub-authors"><span class="pub-me">Branchini, Nicola</span> and Elvira, Víctor</p>
    <div class="pub-tags">
      <span class="pub-tag">Monte Carlo</span>
      <span class="pub-tag">Optimal transport</span>
    </div>
    <p class="pub-links">
      <a href="https://www.youtube.com/watch?v=tG9mjp6GgtE&amp;list=PLUbgZHsSoMEUq6vqSLjwuXfrGDBNLbZRu&amp;index=11">Video from SMC 2024</a>
      <a href="https://xianblog.wordpress.com/2024/06/05/6th-workshop-on-sequential-monte-carlo-methods-2/">Xi'an's blog comments</a>
    </p>
    <button type="button" class="collapsible">Details</button>
    <div class="content">
      <p>The self-normalized IS estimator is widely used to estimate expectations with intractable normalizing constants, for example, in Bayesian leave-one-out cross validation or likelihood free inference. In this paper, we propose a framework to understand when SNIS works and when it does not, with a generalization that allows us to overcome its limitations, with connections to continuous optimal transport.</p>
    </div>
  </article>

  <article class="pub-item pub-featured-item" data-groups="monte-carlo">
    <div class="pub-venue">AISTATS 2026</div>
    <h3 class="pub-title"><a href="https://arxiv.org/abs/2503.21346">How to approximate inference with subtractive mixture models</a></h3>
    <p class="pub-authors">Zellinger, Lena and <span class="pub-me">Branchini, Nicola</span> and De Smet, Lennert and Elvira, Víctor and Malkin, Nikolay and Vergari, Antonio</p>
    <div class="pub-tags">
      <span class="pub-tag">Monte Carlo</span>
    </div>
    <button type="button" class="collapsible">Details</button>
    <div class="content">
      <p>Coming soon!</p>
    </div>
  </article>

  <article class="pub-item pub-featured-item" data-groups="monte-carlo">
    <div class="pub-venue">NeurIPS 2024 Workshop</div>
    <h3 class="pub-title"><a href="https://openreview.net/forum?id=Zxk07UdWEy">The role of tail dependence in estimating posterior expectations</a></h3>
    <p class="pub-authors"><span class="pub-me">Branchini, Nicola</span> and Elvira, Víctor</p>
    <p class="pub-note">NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty</p>
    <div class="pub-tags">
      <span class="pub-tag">Monte Carlo</span>
    </div>
    <button type="button" class="collapsible">Details</button>
    <div class="content">
      <p>To estimate posterior expectations consistently, we need to use self-normalized importance sampling (or MCMC, but SNIS has a better variance lower bound). It is a ratio of two IS estimators. Typical diagnostics forget this, and only look at IS-weights for numerator or denominator separately. We try to capture this information with the concept of tail dependence of random variables, which applies in heavy-tailed scenarios. Ongoing journal extension.</p>
    </div>
  </article>
</div>

<div class="pub-section-title">All publications</div>

<div class="pub-filters" role="group" aria-label="Filter publications by topic">
  <button type="button" class="pub-filter is-active" data-filter="all">All</button>
  <button type="button" class="pub-filter" data-filter="monte-carlo">Monte Carlo methods</button>
  <button type="button" class="pub-filter" data-filter="causality">Statistical causality</button>
  <button type="button" class="pub-filter" data-filter="optimal-transport">Optimal transport</button>
</div>

<div class="pub-list">

<article class="pub-item" data-groups="monte-carlo">
  <div class="pub-venue">AISTATS 2026</div>
  <h3 class="pub-title"><a href="https://arxiv.org/abs/2503.21346">How to approximate inference with subtractive mixture models</a></h3>
  <p class="pub-authors">Zellinger, Lena and <span class="pub-me">Branchini, Nicola</span> and De Smet, Lennert and Elvira, Víctor and Malkin, Nikolay and Vergari, Antonio</p>
  <div class="pub-tags"><span class="pub-tag">Monte Carlo</span></div>
  <button type="button" class="collapsible">Details</button>
  <div class="content"><p>Coming soon!</p></div>
</article>

<article class="pub-item" data-groups="monte-carlo optimal-transport">
  <div class="pub-venue">ICLR 2026</div>
  <h3 class="pub-title"><a href="https://arxiv.org/abs/2510.01159">Multimarginal Flow Matching with Adversarially Learnt Interpolants</a></h3>
  <p class="pub-authors">Kviman, Oskar and Tamogashev, Kirill and <span class="pub-me">Branchini, Nicola</span> and Elvira, Víctor and Lagergren, Jens and Malkin, Nikolay</p>
  <p class="pub-note">Earlier version: NeurIPS workshop — 2nd edition of Frontiers in Probabilistic Inference: Learning meets Sampling</p>
  <div class="pub-tags">
    <span class="pub-tag">Monte Carlo</span>
    <span class="pub-tag">Optimal transport</span>
  </div>
  <button type="button" class="collapsible">Details</button>
  <div class="content">
    <p>Existing multimarginal flow matching (FM) methods either do not scale well with dimension or encourage trajectories to pass through intermediate marginal samples, rather than the intermediate distributions. We learn a parameterised interpolant for FM via a GAN-inspired loss, which addresses these shortcomings.</p>
  </div>
</article>

<article class="pub-item" data-groups="monte-carlo">
  <div class="pub-venue">AISTATS 2026</div>
  <h3 class="pub-title"><a href="https://arxiv.org/abs/2503.21346">On the bias of variational resampling</a></h3>
  <p class="pub-authors">Finke, Axel and Kviman, Oskar and <span class="pub-me">Branchini, Nicola</span> and Elvira, Víctor</p>
  <div class="pub-tags"><span class="pub-tag">Monte Carlo</span></div>
  <button type="button" class="collapsible">Details</button>
  <div class="content"><p>Coming soon!</p></div>
</article>

<article class="pub-item" data-groups="monte-carlo">
  <div class="pub-venue">SSP 2025</div>
  <h3 class="pub-title"><a href="https://arxiv.org/abs/2505.00372">Towards Adaptive Self-Normalized Importance Samplers</a></h3>
  <p class="pub-authors"><span class="pub-me">Branchini, Nicola</span> and Elvira, Víctor</p>
  <p class="pub-note">2025 IEEE Statistical Signal Processing Workshop</p>
  <div class="pub-tags"><span class="pub-tag">Monte Carlo</span></div>
  <button type="button" class="collapsible">Details</button>
  <div class="content">
    <p>To estimate µ = E_p[f(θ)] when p's normalizing constant is unknown, instead of doing MCMC on p(θ) or even p(θ)|f(θ)|, or learning a parametric q(θ), we try MCMC directly on p(θ)|f(θ)- µ|, which is the asymptotic-variance minimizing proposal. We propose a simple iterative scheme that works: initial estimate µ₀; run a chain on the approximation p(θ)|f(θ)- µ₀|; estimate µ again with SNIS, and keep iterating.</p>
  </div>
</article>

<article class="pub-item" data-groups="monte-carlo">
  <div class="pub-venue">NeurIPS 2024 Workshop</div>
  <h3 class="pub-title"><a href="https://openreview.net/forum?id=Zxk07UdWEy">The role of tail dependence in estimating posterior expectations</a></h3>
  <p class="pub-authors"><span class="pub-me">Branchini, Nicola</span> and Elvira, Víctor</p>
  <p class="pub-note">NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty</p>
  <div class="pub-tags"><span class="pub-tag">Monte Carlo</span></div>
  <button type="button" class="collapsible">Details</button>
  <div class="content">
    <p>To estimate posterior expectations consistently, we need to use self-normalized importance sampling. Typical diagnostics forget that SNIS is a ratio of two IS estimators. We capture dependence between numerator and denominator via tail dependence of random variables in heavy-tailed scenarios. Ongoing journal extension.</p>
  </div>
</article>

<article class="pub-item" data-groups="monte-carlo optimal-transport">
  <div class="pub-venue">Preprint</div>
  <h3 class="pub-title"><a href="https://arxiv.org/abs/2406.19974">Generalized self-normalized importance sampling</a></h3>
  <p class="pub-authors"><span class="pub-me">Branchini, Nicola</span> and Elvira, Víctor</p>
  <div class="pub-tags">
    <span class="pub-tag">Monte Carlo</span>
    <span class="pub-tag">Optimal transport</span>
  </div>
  <p class="pub-links">
    <a href="https://www.youtube.com/watch?v=tG9mjp6GgtE&amp;list=PLUbgZHsSoMEUq6vqSLjwuXfrGDBNLbZRu&amp;index=11">Video from SMC 2024</a>
    <a href="https://xianblog.wordpress.com/2024/06/05/6th-workshop-on-sequential-monte-carlo-methods-2/">Xi'an's blog comments</a>
  </p>
  <button type="button" class="collapsible">Details</button>
  <div class="content">
    <p>A framework to understand when SNIS works and when it does not, with a generalization that overcomes its limitations, with connections to continuous optimal transport.</p>
  </div>
</article>

<article class="pub-item" data-groups="monte-carlo">
  <div class="pub-venue">AISTATS 2024</div>
  <h3 class="pub-title"><a href="https://proceedings.mlr.press/v238/guilmeau24a.html">Adaptive importance sampling for heavy-tailed distributions via α-divergence minimization</a></h3>
  <p class="pub-authors"><span class="pub-equal">Guilmeau, Thomas♦</span> and <span class="pub-equal">Branchini, Nicola♦</span> and Chouzenoux, Emilie and Elvira, Víctor <span class="pub-equal">(♦ equal contribution)</span></p>
  <div class="pub-tags"><span class="pub-tag">Monte Carlo</span></div>
  <button type="button" class="collapsible">Details</button>
  <div class="content">
    <p>Many adaptive IS (and some VI) methods match moments of a target. When the target has heavy tails, these moments can be undefined or hard to estimate. We propose an AIS method that matches moments of a lighter-tailed modified target (exponentiated to power alpha), while minimizing the alpha-divergence to the true target.</p>
  </div>
</article>

<article class="pub-item" data-groups="monte-carlo">
  <div class="pub-venue">AISTATS 2024</div>
  <h3 class="pub-title"><a href="https://proceedings.mlr.press/v238/kviman24a.html">Variational Resampling</a></h3>
  <p class="pub-authors">Kviman, Oskar and <span class="pub-me">Branchini, Nicola</span> and Elvira, Víctor and Lagergren, Jens</p>
  <div class="pub-tags"><span class="pub-tag">Monte Carlo</span></div>
  <button type="button" class="collapsible">Details</button>
  <div class="content">
    <p>Instead of enforcing that particle replication counts match pre-resampling weights in expectation, we optimize replication counts to minimize a divergence between the post- and pre-resampling distributions directly.</p>
  </div>
</article>

<article class="pub-item" data-groups="causality optimal-transport">
  <div class="pub-venue">CLeaR 2024</div>
  <h3 class="pub-title"><a href="https://proceedings.mlr.press/v236/felekis24a.html">Causal optimal transport of abstractions</a></h3>
  <p class="pub-authors">Felekis, Yorgos and Zennaro, Fabio and <span class="pub-me">Branchini, Nicola</span> and Damoulas, Theodoros</p>
  <div class="pub-tags">
    <span class="pub-tag">Statistical causality</span>
    <span class="pub-tag">Optimal transport</span>
  </div>
  <button type="button" class="collapsible">Details</button>
  <div class="content">
    <p>We learn causal abstractions from data without specifying parametric SCM functions, via a multimarginal OT problem with soft constraints and a cost encoding knowledge of the underlying causal DAGs. The soft constraints have a do-calculus interpretation.</p>
  </div>
</article>

<article class="pub-item" data-groups="monte-carlo">
  <div class="pub-venue">FoDS</div>
  <h3 class="pub-title"><a href="https://www.aimsciences.org/article/doi/10.3934/fods.2024017">An adaptive mixture view of particle filters</a></h3>
  <p class="pub-authors"><span class="pub-me">Branchini, Nicola</span> and Elvira, Víctor</p>
  <p class="pub-note">Foundations of Data Science</p>
  <div class="pub-tags"><span class="pub-tag">Monte Carlo</span></div>
  <button type="button" class="collapsible">Details</button>
  <div class="content">
    <p>A journal extension of the optimized APF paper: at each iteration we want a mixture proposal close to a mixture target. Literature often matches term-by-term; this view suggests methods that match the two mixtures directly.</p>
  </div>
</article>

<article class="pub-item" data-groups="causality">
  <div class="pub-venue">AISTATS 2023</div>
  <h3 class="pub-title"><a href="https://proceedings.mlr.press/v206/branchini23a.html">Causal Entropy Optimization</a></h3>
  <p class="pub-authors"><span class="pub-me">Branchini, Nicola</span> and Aglietti, Virginia and Dhir, Neil and Damoulas, Theodoros</p>
  <div class="pub-tags"><span class="pub-tag">Statistical causality</span></div>
  <button type="button" class="collapsible">Details</button>
  <div class="content">
    <p>We study causal global optimization under unknown graphs: the effect of incorrect causal assumptions, and an acquisition function that trades off optimization of the effect and structure learning. <img src="/ceo.svg" width="1000" height="300" alt="Causal Entropy Optimization figure"></p>
  </div>
</article>

<article class="pub-item" data-groups="monte-carlo">
  <div class="pub-venue">UAI 2021</div>
  <h3 class="pub-title"><a href="https://proceedings.mlr.press/v161/branchini21a.html">Optimized Auxiliary Particle Filters: adapting mixture proposals via convex optimization</a></h3>
  <p class="pub-authors"><span class="pub-me">Branchini, Nicola</span> and Elvira, Víctor</p>
  <div class="pub-tags"><span class="pub-tag">Monte Carlo</span></div>
  <p class="pub-links"><a href="https://underline.io/speakers/119464-nicola-branchini">Video and slides from UAI</a></p>
  <button type="button" class="collapsible">Details</button>
  <div class="content">
    <p>We improve the Auxiliary Particle Filter by optimizing resampling weights as mixture weights of an importance sampling mixture proposal. Choosing mixture weights to minimize empirical variance of importance weights leads to a convex optimization problem. <img src="/eq_oapf.svg" width="1000" height="300" alt="Optimized APF equation"></p>
  </div>
</article>

</div>

</div>

<script src="/js/pubs-filter.js"></script>
