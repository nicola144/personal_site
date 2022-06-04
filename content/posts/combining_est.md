---
title: "(WORK IN PROGRESS) Combining independent and unbiased estimators"
date: 2022-05-26T22:10:06Z
type: page
draft: true
---

I came across this 1.25 page paper by Don Rubin and Sanford Weisberg [(Rubin \& Weisberg)](https://academic.oup.com/biomet/article-abstract/62/3/708/257707) in Biometrika from 1975.
It considers the problem of finding the "best" linear combination (*whose weights sum to 1* !) of $K$ estimators of the *same* quantity. The estimators are all assumed to be unbiased, and independent. I think this is still a very much relevant topic; however, I won't try to convince you of this, because I want to keep this short.
If anything, it can be seen as a fun little exercise. The result is simple, so probably has been used independently by many authors, without them being aware of this paper (which only has 18 citations!).

We let $\tau$ be the true, unknown quantity of interest. Estimators of $t$ will just be sub-indexed, as $t\_1,\dots,t\_K$. These are *independent* (not necessarily i.i.d.) and *unbiased*.  We will assess the quality of the estimators by their mean squared error. We now define an estimator: $\widehat{t} := \sum\_{k=1}^{K} \hat{\alpha}\_{k} t\_{k} $, with the weights $\hat{\alpha}\_k$ be ***random variables*** and such that $\sum\_{k=1}^{K} \widehat{\alpha\_{k}} = 1$. They are mutually independent, and also independent of $t\_1,\dots,t\_K$. Why did they define weights as random variables ? Probably as we shall see, because the optimal weights involve a quantity that needs to be estimated. That is, $\widehat{t}$ is the estimator we can *actually* use, and we will compare it to some intractable optimal solution.  
The $\widehat{t}$ estimator is unbiased by applying the law of iterated expectation:

$$\begin{equation}\begin{aligned}
\mathbb{E}\_{\mathbf{P}\_{\widehat{t}}}[\widehat{t}] = \mathbb{E}\_{\bigotimes\_k \mathbf{P}\_{\alpha\_{k}}}[\mathbb{E}\_{\mathbf{P}\_{\widehat{t} | \widehat{\boldsymbol{\alpha}}}}[ \widehat{t} | \widehat{\alpha}\_{1}, \dots,  \widehat{\alpha}\_{K}]]  = \tau \cdot \mathbb{E}\_{\bigotimes\_k \mathbf{P}\_{\alpha\_{k}}} \left \[ \left (\sum\_{k=1}^{K} \widehat{\alpha\_{k}} \right ) \right \] = \tau
\end{aligned}\end{equation}\tag{1}\label{eq1}$$

where $\bigotimes\_k \mathbf{P}\_{\alpha\_{k}}$ is the joint law of the $ \widehat{\boldsymbol{\alpha}} := [\widehat{\alpha}\_{1},\dots,\widehat{\alpha}\_{K}]^\top$, $\mathbf{P}\_{\widehat{t} | \widehat{\boldsymbol{\alpha}}}$ the conditional of $\widehat{t}$ given $\widehat{\boldsymbol{\alpha}}$, and $\mathbf{P}\_{\widehat{t}}$ the marginal of $\widehat{t}$. Note that actually we need the weights to sum to $1$ only in expectation, for unbiasedness. However, we will need that they sum to 1 for *all* realizations of the random variables for the next derivation.
Because of the unbiasedness, the mean squared error of the estimator $\widehat{t}$ will be just equal to its variance, for which we apply the law of total variance:
$$\begin{equation}\begin{aligned}
 \mathbb{V}\_{\mathbf{P}\_{\widehat{t}}}[\widehat{t}] &= \mathbb{E}\_{\bigotimes\_k \mathbf{P}\_{\alpha\_{k}}} \left [ \mathbb{V}\_{\mathbf{P}\_{\widehat{t} | \widehat{\boldsymbol{\alpha}}}} \left [ \widehat{t} | \widehat{\alpha}\_{1}, \dots,  \widehat{\alpha}\_{K} \right ] \right ] + \mathbb{V}\_{\bigotimes\_k \mathbf{P}\_{\alpha\_{k}}} \left [ \mathbb{E}\_{\mathbf{P}\_{\widehat{t} | \widehat{\boldsymbol{\alpha}}}} \left [ \widehat{t} | \widehat{\alpha}\_{1}, \dots,  \widehat{\alpha}\_{K} \right ] \right ] \\\\\\
 &= \mathbb{E}\_{\bigotimes\_k \mathbf{P}\_{\alpha\_{k}}} \left [ \mathbb{V}\_{\mathbf{P}\_{\widehat{t} | \widehat{\boldsymbol{\alpha}}}} \left [ \sum\_{k=1}^{K} \hat{\alpha}\_{k} t\_{k}  \right ] \right ] + \tau^2 \cdot \underbrace{\mathbb{V}\_{\bigotimes\_k \mathbf{P}\_{\alpha\_{k}}} \left [ \sum\_{k=1}^{K} \widehat{\alpha\_{k}} \right ]}\_{=~ 0} \\\\\\
 &= \mathbb{E}\_{\bigotimes\_k \mathbf{P}\_{\alpha\_{k}}} \left [ \sum_{k=1}^{K} \hat{\alpha}\_{k}^{2} V\_{k} \right ] .
\end{aligned}\end{equation}\tag{2}\label{eq2}$$
In the third line, the second term is $0$ since the variance of $1$ is $0$. At this point Rubin \& Weisberg use a little trick to link this variance to that of the optimal one. Let us define $t^\star := \sum\_{k=1}^{K} \alpha\_{k}^{\star} t\_{k}$ (the paper uses $\alpha\_{k}$ instead, but my notation is better). The optimum weights are now **deterministic**, and they can be shown to be equal to $\alpha\_{k}^{\star} = \frac{1}{W \cdot V\_{k}}$ with $W = \sum\_{k=1}^{K} \frac{1}{V\_k}$. Therefore, let the optimum estimator be:
$$\begin{equation}\begin{aligned}
t^\star := \sum\_{k=1}^{K} \alpha\_{k}^{\star} t\_{k} = \frac{\sum\_{k=1}^{K} \frac{1}{V\_{k}} t\_{k}}{\sum\_{k^\prime=1}^{K} \frac{1}{V\_{k^\prime}}}
\end{aligned}\end{equation}\tag{3}\label{eq3}$$
<!-- This is the so-called "BLUE" (Best Unbiased Linear Estimator) in this specific context. Note that by design $\sum\_{k=1}^{K} \alpha\_{k}^{\star} = 1$. Now that weights are deterministic, it is even more obvious that $\mathbb{E}[t^\star] = \mathbb{E}\_{\mathbf{P}\_{\widehat{t}}}[t^\star] = \tau$. The variance of $t^\star$ is readily seen as $\mathbb{V}\_{\mathbf{P}\_{\widehat{t}}}[\sum\_{k=1}^{K} \alpha\_{k}^{\star} t\_{k}] = \sum\_{k=1}^{K} (\alpha\_{k}^{\star})^2 V\_{k}$. -->

## Thoughts



## References

Rubin, D.B. and Weisberg, S., 1975. The variance of a linear combination of independent estimators using estimated weights. Biometrika, 62(3), pp.708-709.


<div id="disqus_thread"></div>
<script>
    /**
    *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
    *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables    */

    var disqus_config = function () {
    this.page.url = "https://www.branchini.fun/posts/combining_est";  
    this.page.identifier = "combiningest"; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
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
<pre tabindex="0"><code>@article{branchini2022combining,
  title   = Combining independent and unbiased estimators,
  author  = Branchini, Nicola,
  journal = https://www.branchini.fun,
  year    = 2022,
}
