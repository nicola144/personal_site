---
title: "(WORK IN PROGRESS) Importance Sampling: (much!) more than numerical integration"
date: 2022-05-26T22:10:06Z
type: page
draft: true
---
***Disclaimer (1)***: *For the Importance Sampling - expert reader: I will be using the term in a very broad sense.*

***Disclaimer (2)***: *This post is not a generic introduction to Importance Sampling. It is an overview of many of the places where the key ideas behind the methodology are used. It assumes previous knowledge about it.*

Black-box Variational inference, offline Reinforcement Learning, covariate shift, treatment effect estimation, rare event simulation, training Energy Based Models, gradient estimation ,"target-aware" Bayesian inference, fast training of deep neural networks, optimal control …
What could be an idea that underlies *all* of these ?
Importance Sampling !!


### Numerical integration
The general idea that an average approximates an expectation is pervasive. Similarly so, is the idea that observations (data, samples, etc.) can be modelled/interpreted as realizations of an underlying random variable. The idea of Monte Carlo is related to both.
Traditionally, Monte Carlo methods come up in the context of numerical integration. Here, the problem to solve is simply to approximate the value of an integral:
$$
\int_{\mathcal{X}} h(\mathbf{x}) \mathrm{d}\mathbf{x}
\tag{1}\label{eq1}
$$
  Deterministic algorithms like *[Simpson's](https://en.wikipedia.org/wiki/Simpson%27s_rule)* or the *[trapezoid](https://en.wikipedia.org/wiki/Trapezoidal_rule)* rules scale terribly with the dimension of the integration variable. These rely on dividing the space into grid: not a good idea when the dimension increases.
 Monte Carlo provides a framework to develop *randomized* algorithms that are more efficient, theoretically and practically. Often, the integral of interest is already in the form of an expectation:
 $$
 \int\_{\mathcal{X}} h(\mathbf{x}) \mathrm{d}\mathbf{x} = \int\_{\mathcal{X}} f(\mathbf{x}) \cdot \pi(\mathbf{x}) \mathrm{d}\mathbf{x} .
 \tag{2}\label{eq2}
 $$
In these cases, it is natural to think of generating points distributed according to $\pi(\mathbf{x})$: this leads to approximating \eqref{eq2} with an arithmetic average (convenient), and many things can be proved about this solution (also convenient). When the integral of interest is *not* an expectation, which is in the more general setting of numerical integration (\eqref{eq1}), things become more interesting.

### Importance Sampling as a randomized algorithm for numerical integration
 To approximate \eqref{eq1}, we want to generate (or obtain from someone else) points from the integration space $\mathcal{X}$ *randomly*. To do a good job, these points ought to be in regions where the integrand has large values.  
The IS-savvy reader may say that the generated points need to follow a distribution of some known form. That a density should be available. Maybe the Radon-Nykodim derivative needs to exist, and absolute continuity conditions need to hold, etc. etc. I want to take a broader view, that allows me to consider "IS" a method even if some of these conditions are relaxed. The only one I don't want to relax is that points need to be generated randomly. For example, the IS weights may not be computable. Some may say, then it's not IS - I'm fine with this too, let's not dabble around semantics too much.   

### Beyond (explicit) numerical integration

Let me start going through the applications, beyond explicit numerical integration, where the IS idea is key. Most people reading this post likely have a good understanding of how central the idea of Monte Carlo estimation is to countless domains, but not necessarily of IS specifically.

#### Reinforcement learning

A classic example where the IS idea comes up all over the place is Reinforcement Learning, where the objective function (that needs to be *maximized*, and not *estimated*) is an expectation. More specifically, it is an expectation w.r.t, among other things, a quantity that can be controlled by the algorithm (i.e. , the *policy* of the agent). Similarly to how an IS algorithm gets to choose the proposal, i.e. decide how samples are generated. More concretely, in RL the IS idea has been used to derive certain estimators of the gradient of this objective function (see e.g. [(Tang \& Abbeel, 2010)](https://proceedings.neurips.cc/paper/2010/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html)), to derive <span style="color:#0695FF"> *policy gradient algorithms*</span>. In fact, recently [Parmas \& Sugiyama (2021)](https://proceedings.mlr.press/v130/parmas21a) unify both the common "REINFORCE" (or the log-trick) and the pathwise/reparametrization estimators under an importance sampling perspective, in the general setting (not restricted to RL objectives). Quoting from the paper: "*We on the other
hand, suggest importance sampling as a key component
of any gradient estimator,[...]*"  .
IS also naturally comes up in off-policy evaluation, where the objective is to estimate the state-value function, using samples from policies *other* than the one actually used by the agent to take actions. Off-policy evaluation is a particular task within the more general field of *offline RL*; in [Levine et al. (2021)]()

#### Variational inference

#### Decision making

#### Covariate shift

#### Energy based models

#### Small probabilities
(include bootstrap and p-values)

#### Deep learning

#### Optimal control

#### More recent cool stuff

Thus it
might be possible to obtain better lower bounds by using
methods from the importance sampling literature such as
control variates and adaptive importance sampling

We hope that this work highlights the potential for further improving variational
techniques by drawing upon the vast body of research on (adaptive) importance
sampling in the computational statistics literature.


## References
- Jie, T. and Abbeel, P., 2010. On a connection between importance sampling and the likelihood ratio policy gradient. Advances in Neural Information Processing Systems, 23.
- Parmas, P. &amp; Sugiyama, M.. (2021).  A unified view of likelihood ratio and reparameterization gradients . <i>Proceedings of The 24th International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 130:4078-4086 Available from https://proceedings.mlr.press/v130/parmas21a.html.
- Francisco J. R. Ruiz et al. (UAI 2016). Overdispersed black-box variational inference
- Mohamed, S., Rosca, M., Figurnov, M. and Mnih, A., 2020. Monte Carlo Gradient Estimation in Machine Learning. J. Mach. Learn. Res., 21(132), pp.1-62.
- Mnih, A. &amp; Rezende, D.. (2016). Variational Inference for Monte Carlo Objectives. <i>Proceedings of The 33rd International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 48:2188-2196 Available from https://proceedings.mlr.press/v48/mnihb16.html.
- Levine, S., Kumar, A., Tucker, G. and Fu, J., 2020. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643.
- Metelli, A.M., Papini, M., Montali, N. and Restelli, M., 2020. Importance Sampling Techniques for Policy Optimization. J. Mach. Learn. Res., 21, pp.141-1.
- Sugiyama, M., Krauledat, M. and Müller, K.R., 2007. Covariate shift adaptation by importance weighted cross validation. Journal of Machine Learning Research, 8(5).
- Sugiyama, M. and Ridgeway, G., 2006. Active learning in approximately linear regression based on conditional expectation of generalization error. Journal of Machine Learning Research, 7(1).
- Ranganath, R., Gerrish, S. &amp; Blei, D.. (2014). Black Box Variational Inference. <i>Proceedings of the Seventeenth International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 33:814-822 Available from https://proceedings.mlr.press/v33/ranganath14.html.
- Domke, J. and Sheldon, D.R., 2018. Importance weighting and variational inference. Advances in neural information processing systems, 31.
- Agrawal, A., Sheldon, D.R. and Domke, J., 2020. Advances in black-box VI: Normalizing flows, importance weighting, and optimization. Advances in Neural Information Processing Systems, 33, pp.17358-17369.
- Daudel, K., 2021. Adaptative Monte-Carlo methods for complex models (Doctoral dissertation, Institut polytechnique de Paris).
- Bugallo, M.F., Elvira, V., Martino, L., Luengo, D., Miguez, J. and Djuric, P.M., 2017. Adaptive importance sampling: The past, the present, and the future. IEEE Signal Processing Magazine, 34(4), pp.60-79.
- Finke, A. and Thiery, A.H., 2019. On importance-weighted autoencoders. arXiv preprint arXiv:1907.10477.
- Dieng, A.B. and Paisley, J., 2019. Reweighted expectation maximization. arXiv preprint arXiv:1906.05850.
- Khan, S. and Ugander, J., 2021. Adaptive normalization for IPW estimation. arXiv preprint arXiv:2106.07695.
- Kuzborskij, I., Vernade, C., Gyorgy, A. and Szepesvári, C., 2021, March. Confident off-policy evaluation and selection through self-normalized importance weighting. In International Conference on Artificial Intelligence and Statistics (pp. 640-648). PMLR.
- Swaminathan, A. and Joachims, T., 2015. The self-normalized estimator for counterfactual learning. advances in neural information processing systems, 28.
- Lopez, R., Boyeau, P., Yosef, N., Jordan, M. and Regier, J., 2020. Decision-making with auto-encoding variational bayes. Advances in Neural Information Processing Systems, 33, pp.5081-5092.
- Grover, A., Song, J., Kapoor, A., Tran, K., Agarwal, A., Horvitz, E.J. and Ermon, S., 2019. Bias correction of learned generative models using likelihood-free importance weighting. Advances in neural information processing systems, 32.
- Will Grathwohl, Jacob Kelly, Milad Hashemi, Mohammad Norouzi, Kevin Swersky, David Duvenaud
ICLR 2021. No MCMC for me: Amortized sampling for fast and stable training of energy-based models.
- Brekelmans et al. (ICLR 2022). Improving Mutual Information Estimation with Annealed and Energy-Based Bounds
- Kappen, H.J. and Ruiz, H.C., 2016. Adaptive importance sampling for control and inference. Journal of Statistical Physics, 162(5), pp.1244-1266.
- Asmar, D.M., Senanayake, R., Manuel, S. and Kochenderfer, M.J., 2022. Model Predictive Optimized Path Integral Strategies. arXiv preprint arXiv:2203.16633.

<p>Cited as:</p>
<pre tabindex="0"><code>@article{branchini2022is,
  title   = Learning with not Enough Data Part 3: Data Generation,
  author  = Branchini, Nicola,
  journal = https://www.branchini.fun,
  year    = 2022,
}