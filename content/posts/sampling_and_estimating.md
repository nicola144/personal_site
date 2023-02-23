---
title: "Why do sampling and estimating the normalizing constant avoid each other?"
date: 2023-02-23T15:34:02Z 
tags: ["Short"]
---

We want to get samples from $p(\mathbf{x})$, exactly or approximately. Except for some cases (and for inverse transform sampling), in general, it does not matter at all whether we know the normalizing constant of $p$, i.e. $Z\_{p} = \int \widetilde{p}(\mathbf{x}) d \mathbf{x}$ where $\widetilde{p}(\mathbf{x})$ is the unnormalized density (see e.g. [this X-validated response by Xi'an](https://stats.stackexchange.com/questions/465787/can-sampling-be-difficult-even-with-access-to-the-normalized-version-of-the-dist)). The most generic method (class of methods) to obtain samples is Markov Chain Monte Carlo (MCMC), which by design avoids the need for $Z\_{p}$. Importantly, even if we knew it (some oracle gave it to us), it would not help us at all (with the current methods I am aware of) in improving MCMC speed or otherwise. Another way is to use sampling importance resampling (SIR), based on self-normalized importance sampling (IS) plus resampling. Also with this method, since the IS weights need to be normalized for resampling, the $Z_p$ cancels out. 

Some other times, all we are interested in is approximating $Z\_{p}$, viewed as nothing more than a numerical integration task. For example, for model comparison in Bayesian statistics. The way to do it with a randomized algorithm is Monte Carlo (MC), whose generalization (suited for this task) is IS. Suppose again, an oracle gives us samples from $p$. That's great, it is provably the optimal density to sample from to minimize the IS variance of the estimator (hence MC variance). But again, it does not help as at all. We have nothing to do with those samples, because $Z\_{p}$ is not an expectation w.r.t to $p$, so we cannot use plain MC, and IS requires that we know the normalizing constant of the proposal. If the samples come from $p$, then the proposal is $p$, and we don't know its normalizing constant by definition (it is literally what we are trying to estimate). 

<span style="color:#0695FF"> **Knowing the constant does not help us in sampling. Sampling does not help us in estimating the constant.** </span>

The statement would be more accurate adding caveats like ("by itself, does not help .."), but also less catchy.

In my view, there is something a bit unintuitive about all of this.  
So what is going on here ? Is it just trivial that the two tasks should be uninformative about each other? 
<!-- $$\begin{equation}\begin{aligned}
\mu = \mathbb{E}\_{p}[f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x} , 
\end{aligned}\end{equation}\tag{1}\label{eq1}$$ -->





<div id="disqus_thread"></div>
<script>
    /**
    *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
    *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables    */

    var disqus_config = function () {
    this.page.url = "https://www.branchini.fun/posts/sampling_and_estimating";  
    this.page.identifier = "samplingandestimating"; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
    };

    (function() { // DON'T EDIT BELOW THIS LINE
    var d = document, s = d.createElement('script');
    s.src = 'https://personal-website-g7y0elzvjn.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>


<p>Cited as:</p>
<pre tabindex="0"><code>@article{branchini2022isobvious,
  title   = "Why do sampling and estimating the normalizing constant avoid each other?",
  author  = Branchini, Nicola,
  journal = https://www.branchini.fun,
  year    = 2022,
}
