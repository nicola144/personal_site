---
title: "(WORK IN PROGRESS) Importance Sampling: (much!) more than numerical integration"
date: 2022-05-26T22:10:06Z
type: page
draft: true
---
***Disclaimer (1)***: *For the Importance Sampling - expert reader: I will be using the term in a very broad sense.*

***Disclaimer (2)***: *This post is not a generic introduction to Importance Sampling. It is an overview of many of the places where the key ideas behind the methodology are used. It is biased towards the machine learning literature. It assumes previous knowledge about it.*


Black-box Variational Inference, offline Reinforcement Learning, probabilistic programming, covariate shift, treatment effect estimation, rare event simulation, training Energy Based Models, gradient estimation ,"target-aware" Bayesian inference, fast training of deep neural networks, optimal control …
What could be an idea that is crucial *all* of these really cool topics ?
Importance Sampling !!


### Numerical integration
The general idea that an average approximates an expectation is pervasive. Similarly so, is the idea that observations (data, samples, etc.) can be modelled/interpreted as realizations of an underlying random variable. The idea of Monte Carlo is related to both.
Traditionally, Monte Carlo methods come up in the context of numerical integration. Here, the problem to solve is simply to approximate the value of an integral:
$$
\int_{\mathcal{X}} h(\mathbf{x}) \mathrm{d}\mathbf{x}
\tag{1}\label{eq1}
$$
  Notice there is nothing inherently statistical about this problem (I should write a blogpost about this too). Deterministic algorithms like *[Simpson's](https://en.wikipedia.org/wiki/Simpson%27s_rule)* or the *[trapezoid](https://en.wikipedia.org/wiki/Trapezoidal_rule)* rules scale terribly with the dimension of the integration variable. These rely on dividing the space into grid: not a good idea when the dimension increases.
 Monte Carlo provides a framework to develop *randomized* algorithms that are more efficient, theoretically and practically. Often, the integral of interest is already in the form of an expectation:
 $$
 \int\_{\mathcal{X}} h(\mathbf{x}) \mathrm{d}\mathbf{x} = \int\_{\mathcal{X}} f(\mathbf{x}) \cdot \pi(\mathbf{x}) \mathrm{d}\mathbf{x} .
 \tag{2}\label{eq2}
 $$
In these cases, it is natural to think of generating points distributed according to $\pi(\mathbf{x})$: this leads to approximating \eqref{eq2} with an arithmetic average (convenient), and many things can be proved about this solution (also convenient). When the integral of interest is *not* an expectation, which is in the more general setting of numerical integration \eqref{eq1}, things become more interesting.

### Importance Sampling as a randomized algorithm for numerical integration
 To approximate \eqref{eq1}, we want to generate (or obtain from someone else) points from the integration space $\mathcal{X}$ *randomly*. To do a good job, these points ought to be in regions where the integrand has large values.  
The IS-savvy reader may say that the generated points need to follow a distribution of some known form. That a density should be available. Maybe the Radon-Nykodim derivative needs to exist, and absolute continuity conditions need to hold, etc. etc. I want to take a broader view, that allows me to consider "IS" a method even if some of these conditions are relaxed. The only one I don't want to relax is that points need to be generated randomly. And that the resulting *estimators* have bounded error. For example, the IS weights may not be computable. Some may say, then it's not IS - I'm fine with this too, let's not dabble around semantics too much.   

### Beyond (explicit) numerical integration

Let me start going through the applications, beyond explicit numerical integration, where the IS idea is key. Most people reading this post likely have a good understanding of how central the idea of Monte Carlo estimation is to countless domains, but not necessarily of IS specifically.

#### Reinforcement learning

A classic example where the IS idea comes up all over the place is Reinforcement Learning, where the objective function (that needs to be *maximized*, and not *estimated*) is an expectation. More specifically, it is an expectation w.r.t, among other things, a quantity that can be controlled by the algorithm (i.e. , the *policy* of the agent). Similarly to how an IS algorithm gets to choose the proposal, i.e. decide how samples are generated. More concretely, in RL the IS idea has been used to derive certain estimators of the gradient of this objective function (see e.g. [(Tang \& Abbeel, 2010)](https://proceedings.neurips.cc/paper/2010/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html)), to derive *policy gradient algorithms*. In fact, recently [Parmas \& Sugiyama (2021)](https://proceedings.mlr.press/v130/parmas21a) unify both the common  *REINFORCE/score function* (AKA the log-derivative trick) and the pathwise/reparametrization estimators under an importance sampling perspective, in the general setting (not restricted to RL objectives). Quoting from the paper: "<span style="color:#0695FF">*We on the other hand, suggest importance sampling as a key component
of any gradient estimator,[...]*</span>"  .
In off-policy evaluation, where the objective is to estimate the state-value function $V\_{\pi}$, using samples from policies *other* than that used by the agent to take actions. The function $V\_{\pi}$ is then used as a building block of many algorithms to optimize the RL objective.  
Off-policy evaluation is a particular task within the more general field of *offline RL* [(Levine et al., (2021)](https://arxiv.org/abs/2005.01643). Many of the techniques in this field are based on what they call "Marginalized" Importance Sampling, where the agent's action random variables are (approximately) integrated out from the importance ratio to reduce variance (related to the "Rao-Blackwellization" idea in Monte Carlo [(Robert \& Roberts, 2021)](https://onlinelibrary.wiley.com/doi/abs/10.1111/insr.12463)), obtaining a "state-marginal importance ratio".
In particular, [Metelli et al. (2021)](https://www.jmlr.org/papers/volume21/20-124/20-124.pdf) develop an IS-based framework to develop model-free (i.e. not using knowledge of how the environment responds to the agent's actions) policy optimization algorithms that mix on–line and off–line optimization. [Kuzborskij et al. (2021)](http://proceedings.mlr.press/v130/kuzborskij21a.html) ..
A very exciting recent work [(Metelli et al., 2022)](https://openreview.net/forum?id=5y35LXrRMMz) starts using the more advanced IS idea of taking into account *the whole integrand*, as opposed to the target distribution only, when designing a sampling scheme. All in order to improve RL algorithms, which interestigly (recall), do not try to estimate an integral but to optimize it. Also (Understanding the Curse of Horizon in Off-Policy Evaluation via Conditional Importance Sampling)

#### Variational inference
I was quite surprised to notice that the (very nice) paper which reviews Variational Inference (VI) (*for statisticians, too!*) [(Blei et al., 2017)](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773?casa_token=wpBj9k7gAU0AAAAA%3AzrAT46qgG3uN30hvYd0DleI2K8Rdzi58eJPzPoc16de6MGMXUSlNXjWkIn_x928QtDG3NvroWLuw) does not have a *single mention* of IS. Researchers (myself included) love to say things like "there are connections", without being too specific - risking that the statement can be vacuously true. I will **not** be claiming that VI and (adaptive) IS are the same thing: for example, when doing VI with Gaussian processes, it's not clear to me (at the moment) that there would be a connection. However, there are important contexts where saying "I am doing VI" or "I am doing adaptive IS", essentially becomes a matter of jargon/semantics.

So, what is VI ? Let's base our discussion on the authoritative [(Blei et al., 2017)](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773?casa_token=wpBj9k7gAU0AAAAA%3AzrAT46qgG3uN30hvYd0DleI2K8Rdzi58eJPzPoc16de6MGMXUSlNXjWkIn_x928QtDG3NvroWLuw). There is a joint distribution in the wild. This already implies there are ***two***, distinct random vectors. One is always the observed data; this is what we will condition on. The other can be seen as a latent variable, or a set of parameters over which we are doing Bayesian inference. The narrative goes that we will turn "inference into optimization"<sup>[1](#myfootnote1)</sup>: find the density $q$ (if exists, otherwise distribution function, measure etc.) that is the argmin of $KL(q || p)$, where $p$ is the **conditional** of the latent given the observed. So, where is the connection with (adaptive IS)?

Well. In adaptive IS, we are looking at an integral, which may involve an expectation over some distribution, as we discussed previously. Let's say it does, for simplicity, and that this distibution is the $p$ we were talking about for VI. Then, maybe because we cannot sample from $p$, we say that we want to find a $q$ that is "close" to $p$ . You see it, right? Sure, in IS, we normally implicitly assume parametric models. But so does almost all of modern VI. By which, I mean black-box VI (BBVI). Indeed, in BBVI we assume that nothing is closed form, so we have to optimize the KL iteratively wrt to our $q$, which is parameterized by a neural network. The optimization will be done by gradient descent. Since the KL is an expectation, the gradient of the KL is an expression that involves an expectation. We are back to gradient estimation as in policy gradients, as discussed before. As we mentioned, IS is relevant to form gradient estimators. But now hopefully you can see that we have this $q$, and we are ***sampling from it***, changing its parameters (by following the direction of the approximated gradient), and so on iteratively. This is nothing but adapting an IS proposal to $p$.


They (re)discover that from the samples generated by this process, one can get an accurate (consistent) approximation of $p$, by looking at the distribution of the resampled particles, a concept well understood in self-normalized IS (for a simple proof, see Paul Fearnhead's PhD thesis, page 16, for a simple proof).

In the context of machine learning, the promise of using a statistical model (called "generative") is that of going beyond a discriminative (or purely predictive) model  

Symbolic Parallel Adaptive Importance Sampling for Probabilistic Program Analysis

#### Decision making: treatment effect estimation, policy learning

#### Covariate shift

### Probabilistic programming

#### Generative, energy-based and diffusion models  

#### Small probabilities
(include bootstrap and p-values)

#### Deep learning
Techniques to train modern neural networks rely on stochastic optimization schemes (i.e., Monte Carlo approximations of expectated values in the context of an optimization objective), so unsuprisingly the idea of IS (sampling where it matters) can be exploited.

#### Optimal control

#### Miscellanea, recent cool stuff

adversarial robustness, Object Counting from Satellite Images, Sahra, maybe Safety critical systems , Fair Generative Modeling via Weak Supervision ,  Bayesian leave-one-out
(LOO) predictive densities for cross-validation (Vehtari), Fong and Holmes, Accelerating HEP simulations with Neural Importance Sampling , AAAI 2022, https://thisistian.github.io/publication/real-time-subsurface-with-adaptive-sampling/ 

Thus it might be possible to obtain better lower bounds by using
methods from the importance sampling literature such as
control variates and adaptive importance sampling (Mnih)

We hope that this work highlights the potential for further improving variational
techniques by drawing upon the vast body of research on (adaptive) importance
sampling in the computational statistics literature. (finke et al)

. In most contemporary applications, we do not use large samples—often we use a single sample in
the estimator—and it is in this small sample regime that non-linear controls may have applicability (Mohamed et al)

<div id="disqus_thread"></div>
<script>
    /**
    *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
    *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables    */

    var disqus_config = function () {
    this.page.url = "https://personal-site-lemon-seven.vercel.app/posts/2020-03-17-sequential-monte-carlo-and-improved-auxiliary-particle-filters/";  
    this.page.identifier = "smc-apf"; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
    };

    (function() { // DON'T EDIT BELOW THIS LINE
    var d = document, s = d.createElement('script');
    s.src = 'https://personal-website-g7y0elzvjn.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>


<a name="myfootnote1">1</a>: I hate the word "inference", as it means essentially opposite things in different contexts. Often it's actually quite vague: what does it mean to "know the posterior"? Does it mean being able to compute its density pointwise ? Sample from it? Both? Who knows.


## References
- Jie, T. and Abbeel, P., 2010. On a connection between importance sampling and the likelihood ratio policy gradient. Advances in Neural Information Processing Systems, 23.
- Parmas, P. &amp; Sugiyama, M.. (2021).  A unified view of likelihood ratio and reparameterization gradients . <i>Proceedings of The 24th International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 130:4078-4086 Available from https://proceedings.mlr.press/v130/parmas21a.html.
- Ruiz, F. et al. (UAI 2016). Overdispersed black-box variational inference
- Mohamed, S., Rosca, M., Figurnov, M. and Mnih, A., 2020. Monte Carlo Gradient Estimation in Machine Learning. J. Mach. Learn. Res., 21(132), pp.1-62.
- Mnih, A. &amp; Rezende, D.. (2016). Variational Inference for Monte Carlo Objectives. <i>Proceedings of The 33rd International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 48:2188-2196 Available from https://proceedings.mlr.press/v48/mnihb16.html.
- Levine, S., Kumar, A., Tucker, G. and Fu, J., 2020. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643.
- Metelli, A.M., Papini, M., Montali, N. and Restelli, M., 2020. Importance Sampling Techniques for Policy Optimization. J. Mach. Learn. Res., 21, pp.141-1.
- Hanna, J.P., Niekum, S. and Stone, P., 2021. Importance sampling in reinforcement learning with an estimated behavior policy. Machine Learning, 110(6), pp.1267-1317.
- Sugiyama, M., Krauledat, M. and Müller, K.R., 2007. Covariate shift adaptation by importance weighted cross validation. Journal of Machine Learning Research, 8(5).
- Sugiyama, M. and Ridgeway, G., 2006. Active learning in approximately linear regression based on conditional expectation of generalization error. Journal of Machine Learning Research, 7(1).
- Blei, D.M., Kucukelbir, A. and McAuliffe, J.D., 2017. Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518), pp.859-877.
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
- Vahdat, A., Kreis, K. and Kautz, J., 2021. Score-based generative modeling in latent space. Advances in Neural Information Processing Systems, 34.
- Kappen, H.J. and Ruiz, H.C., 2016. Adaptive importance sampling for control and inference. Journal of Statistical Physics, 162(5), pp.1244-1266.
- Asmar, D.M., Senanayake, R., Manuel, S. and Kochenderfer, M.J., 2022. Model Predictive Optimized Path Integral Strategies. arXiv preprint arXiv:2203.16633.
- Robert, C.P. and Roberts, G., 2021. Rao–Blackwellisation in the Markov Chain Monte Carlo Era. International Statistical Review, 89(2), pp.237-249.
- Meng, C., Liu, E., Neiswanger, W., Song, J., Burke, M., Lobell, D. and Ermon, S., 2021. IS-COUNT: Large-scale Object Counting from Satellite Images with Covariate-based Importance Sampling. arXiv preprint arXiv:2112.09126.
- Rotskoff, G.M., Mitchell, A.R. and Vanden-Eijnden, E., 2022, April. Active importance sampling for variational objectives dominated by rare events: Consequences for optimization and generalization. In Mathematical and Scientific Machine Learning (pp. 757-780). PMLR.

<p>Cited as:</p>
<pre tabindex="0"><code>@article{branchini2022is,
  title   = Importance Sampling: (much!) more than numerical integration,
  author  = Branchini, Nicola,
  journal = https://www.branchini.fun,
  year    = 2022,
}
