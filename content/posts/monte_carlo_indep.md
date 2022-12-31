---
title: "Simple Monte Carlo is independent of dimension. Or is it ?"
date: 2022-11-08T16:16:09Z 
tags: ["Short"]
---

It is essentially the status quo to claim that the error (Mean Squared Eror = MSE) of Monte Carlo integration is "independent of the dimension" of the variable being integrated, or some equivalent variant of this statement. I will provide evidence and citations later. In this post, I provide a simple reasoning for why this can be misleading, especially for the novice that approaches the Monte Carlo literature with the aim to learn how to chooose among estimators in practice. 
Defining the notation needed for the post, let the integral (expectation) to be estimated as 

$$\begin{equation}\begin{aligned}
\mu = \mathbb{E}\_{p}[f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x} , 
\end{aligned}\end{equation}\tag{1}\label{eq1}$$

where $p(\mathbf{x})$ is a density, $\mathbf{x} \in \mathbb{R}^{D}$, the test function $f(\mathbf{x})$ is normally considered $f: \mathbb{R}^D \rightarrow \mathbb{R}$ and the corresponding Monte Carlo estimator as 

$$\begin{equation}\begin{aligned}
\widehat{\mu}\_{\text{MC}} = \frac{1}{N} \sum\_{n=1}^{N} f(\mathbf{x}^{(n)}) , ~~ \mathbf{x}^{(n)} \sim p(\mathbf{x}) ,
\end{aligned}\end{equation}\tag{2}\label{eq2}$$

with samples being i.i.d. 

At this point, many (if not most) authoritative sources proceed with some variant of the following: because the variance (hence the MSE) of $\widehat{\mu}\_{\text{MC}}$ is given simply by

$$\begin{equation}\begin{aligned}
\mathbb{E}\_p[(\widehat{\mu}\_{\text{MC}} - \mu)^2] = \mathbb{V}\_p[\widehat{\mu}\_{\text{MC}}] = \frac{\mathbb{V}\_p[f(\mathbf{x})]}{N} , 
\end{aligned}\end{equation}\tag{3}\label{eq3}$$

which is readily seen to be $\mathcal{O}(1/N)$, then the error is independent of $D$. To start with, the [Wikipedia page for Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_integration) states "<span style="color:#0695FF">*This result does not depend on the number of dimensions of the integral, which is the promised advantage of Monte Carlo integration against most deterministic methods that depend exponentially on the dimension* </span>", referring to \eqref{eq3}. 

Further, Art B. Owen in his textbook \[1\] (my personal favourite source on Monte Carlo) (Chapter 2, page 17) mentions: "<span style="color:#0695FF"> *A striking feature about the formula [equivalent of our Eq. 3] is that the dimension* $D$ *does not appear in it anywhere.* </span> ".

Another example, Doucet and Johansen in their (great) tutorial on particle filtering \[2\], mention: "<span style="color:#0695FF"> *The main advantage of Monte Carlo methods over standard approximation techniques is that the variance of the approximation error decreases at a rate of*  $\mathcal{O}(1/N)$ *regardless of the dimension of the space* </span>".  And more, this well-cited tutorial from the respected "Acta Numerica" \[3\] begins with "<span style="color:#0695FF"> *Its convergence rate,* $\mathcal{O}(1/\sqrt{N})$, *[here referring to RMSE] independent of dimension* </span>". 

In fact, even proponents of competitor methods to Monte Carlo, seem to list this "independence of dimensionality" property as a potential advantage, see e.g. from the Probabilistic Numerics textbook \[4\] (page 110): "<span style="color:#0695FF"> *A defender of Monte Carlo might argue that its most truly desirable characteristic is the fact that its convergence (see Lemma 9.2) does not depend on the dimension of the problem. Performing well even in high dimension is a laudable goal.* </span>". They later propose alternative reasons to what I will explain for why this can be misleading; however, they also mention "<span style="color:#0695FF"> *rather than being equally good for any number of dimensions, Monte Carlo is perhaps better thought of as being equally bad.* </span>"; I am not sure I agree with this: as I will show shortly, even for simple integrands it is clear that a higher dimension is generally **worse**, not equally bad. 

I could go on (another example can be found at page 358 here \[8\]). Iain Murray, in his PhD thesis on MCMC \[5\], seems to take a more careful stance (although not elaborating on it): " <span style="color:#0695FF"> *Monte Carlo is usually simple and its* $\mathcal{O}(1/\sqrt{N})$ *scaling of error bars “independent of dimensionality” may be good enough* </span>"; the quotes added here seem to suggest us that there is more to the story.  

And indeed, there is more to the story. Is $\mathbb{V}\_p[f(\mathbf{x})]$ really independent of the dimension of $\mathbf{x}$, namely $D$ ? Maybe for some choices of $f$, but clearly also not for many others. What is a simple, perhaps the simplest, way to see that the statements previously mentioned can be very misleading?  

Suppose that the function $f$ is non-negative. This is not very restrictive: it includes e.g. normalizing constant estimation, and much of the particle physics literature uses an adaptive Monte Carlo method, "VEGAS", which requires this assumption. More general functions will be the subject of a future blogpost. Then, the MSE in \eqref{eq3} can be written as 

$$\begin{equation}\begin{aligned}
\mathbb{E}\_p[(\widehat{\mu}\_{\text{MC}} - \mu)^2] = \frac{\mu^2}{N} ~ \chi^{2}(p \cdot f \mid \mid p) , 
\end{aligned}\end{equation}\tag{4}\label{eq4}$$

where we $p \cdot f$ denotes the **normalized version** of the product $ f(\mathbf{x}) p(\mathbf{x}) $, i.e. 


$$\begin{equation}\begin{aligned}
p \cdot f = \frac{f(\mathbf{x}) p(\mathbf{x})}{\int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x}} =  \frac{f(\mathbf{x}) p(\mathbf{x})}{\mu}
\end{aligned}\end{equation}\tag{5}\label{eq5}$$

(which is a density) and I have used the chi-squared divergence between densities $p$ and $q$:

$$\begin{equation}\begin{aligned}
\chi^{2}(p \mid \mid q) = \int \frac{p(\mathbf{x})^2}{q(\mathbf{x})} \mathrm{d}\mathbf{x} - 1. 
\end{aligned}\end{equation}\tag{6}\label{eq6}$$

What has changed ? Nothing, essentially, but perhaps people familiar with divergences already start having some intuition, since they usually do not behave well with dimension and in general $p \cdot f$ and $p$ are different densities. I personally like this chi-squared view, among other reasons, because it makes apparent that simple Monte Carlo is a special case of importance sampling. More on this in following posts, likely. 

Now, we take a concrete (and very simple!) example to show how the divergence in \eqref{eq4} can easily scale badly with $D$. Let $f(\mathbf{x})$ be (the square root of) a Gaussian density $f(\mathbf{x}) = \sqrt{\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}\_{1}, \boldsymbol{\Sigma}\_{1})}$, and let $p(\mathbf{x})$ be an actual Gaussian, $p(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}\_{2},  \boldsymbol{\Sigma}\_{2})$. 

We can then expand the MSE of the Monte Carlo estimator as 

$$\begin{equation}\begin{aligned}
\frac{\mu^2}{N} ~ \chi^{2}(p \cdot f \mid \mid p) &= \frac{1}{N}  \left ( \int f(\mathbf{x})^2 p(\mathbf{x}) \mathrm{d}\mathbf{x} - \mu^2 \right )  \\\\\\
&= \frac{1}{N}  \left ( \int \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}\_{1},  \boldsymbol{\Sigma}\_{1}) \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}\_{2},  \boldsymbol{\Sigma}\_{2})  \mathrm{d}\mathbf{x} - \mu^2 \right ) \\\\\\
&=  \frac{1}{N}  \left ( \mathcal{O}((2 \pi)^{D/2}) - \mu^2 \right )
\end{aligned}\end{equation}\tag{7}\label{eq7}$$

and we see that there is a very clear **exponential** dependence on the dimension $D$. In the derivation, we have used simple closure properties of Gaussian densities and the closed form solution for the normalizing constant of a Gaussian. Is it fair to characterize this exponential dependence as "independent of dimension" ? The feeling is that it would be more appropriate to consider an error analysis in the spirit of those done in high-dimensional statistics (e.g. \[6,7\]), where the **ratio** (or similar functions) between the dimension $D$ and the sample size $N$ are studied. Indeed, in \eqref{eq7} as long as $N$ grows exponentially with $D$, the MSE can be controlled to be constant (in $N$ *and* $D$). This is clearly a strong requirement on the number of samples. 


## Caveat 
In some contexts, it seems common to emphasize a difference between $ \mathcal{O}(N^{-1/D}) $ and $\mathcal{O}( c^{D} \cdot N^{-1} )$, where $c$ is a constant (in both $N$ and $D$). While both suffer from a "curse" of dimension, the former is generally worse, and is typical of deterministic quadrature rules for integration. It makes sense that Monte Carlo is seen as an improvement, since its error looks like the latter (and for some functions, there need not be an exponential dependence on $D$ at all). However, it is still misleading to claim "independence of dimension" for the error. Some claim that the *"rate"* $1/N$ is independent of dimension, which to me means that the error does not look like $ \mathcal{O}(N^{-1/D}) $.  This is fair, but often the specific reference to the rate (and what it means) is not clear. Also, ultimately what we care about is how the overall error (MSE) scales.  

## Conclusions

The contents of this post are obvious to Monte Carlo experts, for example the authors of the previously cited works. However, I can imagine easily how outsiders from the literature can be misled by the narrative I have described. 

Indeed, perhaps a finishing thought is that this "mantra" of dimensionality independence of Monte Carlo, **together with the other mantra** of importance sampling (IS) suffering from the curse of dimension, has likely hindered research in IS unfairly, as if the problem with dimension is specifically a feature introduced by IS. We saw, it is not. In fact, we know well that not only is MC a special case of IS with $p$ being the proposal (it is made even more obvious by the chi-squared view of the MSE), but also we know that the freedom of choosing a sampling distribution other than $p$ can bring better (not worse) results. 

<span style="color:#0695FF"> **Addendum** </span>

It has come to my attention after publishing the post, that there is [this very nice note on high dimensional Monte Carlo integration](https://arxiv.org/pdf/2206.09036.pdf) by Yanbo Tang \[9\], using tools from high dimensional statistics.  

<span style="color:#0695FF"> **Addendum (2)** </span>

An obviously simpler example is to take $f(\mathbf{x}) = x\_1 \cdot \dots \cdot x\_D $ and $p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \sigma^{2} \cdot \mathbf{I})$, so then $\mathbb{V}\_p[f(\mathbf{x})] = (\sigma^{2})^{D}$ and the MSE is $N^{-1} \cdot (\sigma^{2})^{D} $

## References
1. Art B. Owen. Monte Carlo theory, methods and examples. 2013
2. Doucet, A. and Johansen, A.M., 2009. A tutorial on particle filtering and smoothing: Fifteen years later. Handbook of nonlinear filtering, 12(656-704), p.3.
3. Caflisch, R.E., 1998. Monte carlo and quasi-monte carlo methods. Acta numerica, 7, pp.1-49.
4. Philipp Hennig, Michael A. Osborne, Hans Kersting. Probabilistic Numerics: Computation as Machine Learning. Cambridge University Press, 2022.
5. Murray, I., 2007. Advances in Markov chain Monte Carlo methods. University of London, University College London (United Kingdom).
6. H. S. Battey, D. R. Cox. "Some Perspectives on Inference in High Dimensions." Statistical Science, 37(1) 110-122 February 2022.
7. Wainwright, M.J., 2019. High-dimensional statistics: A non-asymptotic viewpoint (Vol. 48). Cambridge University Press.
8. MacKay, D.J. and Mac Kay, D.J., 2003. Information theory, inference and learning algorithms. Cambridge university press.
9. Tang, Y., 2022. A Note on Monte Carlo Integration in High Dimensions. arXiv preprint arXiv:2206.09036.

<div id="disqus_thread"></div>
<script>
    /**
    *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
    *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables    */

    var disqus_config = function () {
    this.page.url = "https://www.branchini.fun/posts/monte_carlo_indep/";  
    this.page.identifier = "montecarloindep"; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
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
<pre tabindex="0"><code>@article{branchini2022montecarloindep,
  title   = Simple Monte Carlo is independent of dimension. Or is it ?,
  author  = Branchini, Nicola,
  journal = https://www.branchini.fun,
  year    = 2022,
}
