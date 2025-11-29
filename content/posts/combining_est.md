---
title: "Combining independent and unbiased estimators"
date: 2022-05-26T22:10:06Z 
tags: ["Short", "Paper Summary"]
---

I came across this 1.25 page paper by Don Rubin and Sanford Weisberg [(Rubin \& Weisberg)](https://academic.oup.com/biomet/article-abstract/62/3/708/257707) in Biometrika from 1975. Good old times.


It considers the problem of finding the "best" linear combination (*whose weights sum to 1* !) of $K$ estimators of the *same* quantity. The estimators are all assumed to be unbiased, and independent. I think this is still a very much relevant topic; however, I won't try to convince you of this, because I want to keep this short.
If anything, it can be seen as a fun little exercise. The result is simple, so probably has been used independently by many authors, without them being aware of this paper (which only has 18 citations!).

We let $\tau$ be the true, unknown quantity of interest. Estimators of $t$ will just be sub-indexed, as $t\_1,\dots,t\_K$. These are *mutually independent* (not necessarily i.i.d.) and *unbiased*.  We will assess the quality of the estimators by their mean squared error. We now define an estimator: $\widehat{t} := \sum\_{k=1}^{K} \hat{\alpha}\_{k} t\_{k} $, with the weights $\hat{\alpha}\_k$ be ***random variables*** and such that $\sum\_{k=1}^{K} \widehat{\alpha\_{k}} = 1$. They are independent of $t\_1,\dots,t\_K$. We will see that the $\hat{\alpha}\_k$'s need ***not*** be mutually independent in order for the result to hold. That's all the assumptions on the distribution of the weights. We further denote the variance of the individual estimators $t\_{k}$ as $V\_{k}$.

Why did they define weights as random variables ? As we shall see, because the optimum weights involve a quantity that needs to be estimated. That is, $\widehat{t}$ is the estimator we can *actually* use, and we will compare it to some intractable optimum solution.  
We see that $\widehat{t}$ is unbiased by applying the law of iterated expectation (*and* the law of the unconsciuous statistician):

$$\begin{equation}\begin{aligned}
\mathbb{E}[\widehat{t}] = \mathbb{E}\_{\mathbf{P}\_{\widehat{t}}}[\widehat{t}] = \mathbb{E}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}}[\mathbb{E}\_{\bigotimes\_k \mathbf{P}\_{t\_{k} | \widehat{\boldsymbol{\alpha}}}}[ \widehat{t} | \widehat{\alpha}\_{1}, \dots,  \widehat{\alpha}\_{K}]]  = \tau \cdot \mathbb{E}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}} \left[ \left (\sum\_{k=1}^{K} \widehat{\alpha\_{k}} \right ) \right] = \tau
\end{aligned}\end{equation}\tag{1}\label{eq1}$$

where $\mathbf{P}_{\widehat{\boldsymbol{\alpha}}}$ is the joint law of the $ \widehat{\boldsymbol{\alpha}} := [\widehat{\alpha}_{1},\dots,\widehat{\alpha}_{K}]^\top$, $\bigotimes_k \mathbf{P}_{t_{k} | \widehat{\boldsymbol{\alpha}}}$ the conditional of $\widehat{t}$ given $\widehat{\boldsymbol{\alpha}}$, and $\mathbf{P}_{\widehat{t}}$ the marginal of $\widehat{t}$. If you are not familiar with $\bigotimes_k \mathbf{P}$, it just means a joint which factorizes as the product of its $K$ marginals. Note that actually we need the weights to sum to $1$ only in expectation, for unbiasedness. However, we will need that they sum to 1 for *all* realizations of the random variables for the next derivation.<sup>[1](https://www.branchini.fun/posts/combining\_est/#myfootnote1)</sup>

Because of the unbiasedness, the mean squared error of the estimator $\widehat{t}$ will be just equal to its variance, for which we apply the law of total variance:
$$\begin{equation}\begin{aligned}
 \mathbb{V}[\widehat{t}] = \mathbb{V}\_{\mathbf{P}\_{\widehat{t}}}[\widehat{t}] &= \mathbb{E}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}} \left [ \mathbb{V}\_{\bigotimes\_k \mathbf{P}\_{t\_{k} | \widehat{\boldsymbol{\alpha}}}} \left [ \widehat{t} | \widehat{\alpha}\_{1}, \dots,  \widehat{\alpha}\_{K} \right ] \right ] + \mathbb{V}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}} \left [ \mathbb{E}\_{\bigotimes\_k \mathbf{P}\_{t\_{k} | \widehat{\boldsymbol{\alpha}}}} \left [ \widehat{t} | \widehat{\alpha}\_{1}, \dots,  \widehat{\alpha}\_{K} \right ] \right ] \\\\
 &= \mathbb{E}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}} \left [ \mathbb{V}\_{\bigotimes\_k \mathbf{P}\_{t\_{k} | \widehat{\boldsymbol{\alpha}}}} \left [ \sum\_{k=1}^{K} \hat{\alpha}\_{k} t\_{k}  \right ] \right ] + \tau^2 \cdot \underbrace{\mathbb{V}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}} \left [ \sum\_{k=1}^{K} \widehat{\alpha\_{k}} \right ]}\_{=~ 0} \\\\
 &= \mathbb{E}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}} \left [ \sum\_{k=1}^{K} \hat{\alpha}\_{k}^{2} V\_{k} \right ] .
\end{aligned}\end{equation}\tag{2}\label{eq2}$$
In the third line, the second term is $0$ since the variance of a constant (here, $1$) is $0$. At this point Rubin \& Weisberg use a little trick to link this variance to that of the optimum one. That is, the best variance possible *in this setting*. Let us define $t^\star := \sum_{k=1}^{K} \alpha_{k}^{\star} t_{k}$ (the paper uses $\alpha_{k}$ instead, but my notation is better). The optimum weights are now **deterministic**, and they can be shown to be equal to $\alpha_{k}^{\star} = \frac{1}{W \cdot V_{k}}$ with $W = \sum_{k=1}^{K} \frac{1}{V_k}$. Therefore, let the optimum estimator be:
$$\begin{equation}\begin{aligned}
t^\star := \sum\_{k=1}^{K} \alpha\_{k}^{\star} t\_{k} = \frac{\sum\_{k=1}^{K} \frac{1}{V\_{k}} t\_{k}}{\sum\_{k^\prime=1}^{K} \frac{1}{V\_{k^\prime}}} .
\end{aligned}\end{equation}\tag{3}\label{eq3}$$

 Note that by design $\sum_{k=1}^{K} \alpha_{k}^{\star} = 1$. Now that weights are deterministic, it is even more obvious that $\mathbb{E}[t^\star] = \tau$. The variance of $t^\star$ is readily seen as $\mathbb{V}[t^\star] = \mathbb{V}_{\mathbf{P}_{t^\star}}[t^\star]= \mathbb{V}_{\bigotimes_k \mathbf{P}_{t_{k}}}[\sum_{k=1}^{K} \alpha_{k}^{\star} t_{k}] = \sum_{k=1}^{K} (\alpha_{k}^{\star})^2 V_{k}$.
Now we are going to express the variance of $t^\star$ in a way that will allows us for comparison witht that of $\widehat{t}$ (and indeed, prove $\mathbb{V}[t^\star] \leq \mathbb{V}[\widehat{t}]$). We write:
$$\begin{equation}\begin{aligned}
\mathbb{V}[t^\star] = \sum\_{k=1}^{K} (\alpha\_{k}^{\star})^2 V\_{k} = \frac{\sum\_{k=1}^{K} \left ( \frac{1}{V\_{k}} \right )^2 \cdot V\_{k}}{W^2} = \frac{\cancel{\sum\_{k=1}^{K} \frac{1}{V\_{k}}}}{\cancel{W} \cdot W} = \frac{1}{W} = \frac{1}{V\_{k} W} \cdot V\_{k} = \alpha\_{k} V\_{k} .
\end{aligned}\end{equation}\tag{4}\label{eq4}$$
Now, the trick is to *add and subtract* $\mathbb{V}[t^\star]$ from \eqref{eq2}, *and* replacing $V_{k}$'s for $\frac{\mathbb{V}[t^\star]}{\alpha_{k}^{\star}}$ (given to us by \eqref{eq4}):
$$\begin{equation}\begin{aligned}
\mathbb{V}[\widehat{t}] &=  \mathbb{V}[t^\star] + \mathbb{E}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}} \left [ \sum\_{k=1}^{K} \hat{\alpha}\_{k}^{2} \color{LimeGreen}{V\_{k}} \right ] - \overbrace{\sum\_{k=1}^{K} (\alpha\_{k}^{\star})^2 \color{LimeGreen}{V\_{k}}}^{=~\mathbb{V}[t^\star]} \\\\
&= \mathbb{V}[t^\star] + \mathbb{E}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}} \left [ \sum\_{k=1}^{K} \hat{\alpha}\_{k}^{2} \color{LimeGreen}{\frac{\mathbb{V}[t^\star]}{\alpha\_{k}^{\star}}} \right ] -  \sum\_{k=1}^{K} (\alpha\_{k}^{\star})^{2} \color{LimeGreen}{\frac{\mathbb{V}[t^\star]}{\alpha\_{k}^{\star}}} \\\\
&= \mathbb{V}[t^\star] \left [
1 + \sum\_{k=1}^{K} \mathbb{E}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}} \left [
\frac{\left (   \widehat{\alpha\_{k}} -(\alpha\_{k}^{\star})^2  \right )^2}{\alpha\_{k}^{\star}}
\right ]
\right ] \\\\
&= \mathbb{V}[t^\star] \left [
1 + \sum\_{k=1}^{K} \alpha\_{k}^{\star} \mathbb{E}\_{\mathbf{P}\_{\widehat{\boldsymbol{\alpha}}}} \left [
 \left (\frac{ \widehat{\alpha\_{k}} -(\alpha\_{k}^{\star})^2  }{\alpha\_{k}^{\star}} \right )^2
\right ]
\right ] .
\end{aligned}\end{equation}\tag{5}\label{eq5}$$
Now we see that, indeed, since the rightmost term is always positive, $\mathbb{V}[\widehat{t}] \geq  \mathbb{V}[t^\star]$. The authors note that $\mathbb{V}[\widehat{t}]$ depends on $\widehat{\boldsymbol{\alpha}}$ (which we can think of as estimates for the $\alpha_{k}^{\star}$'s) only through their squared error to $\alpha_{k}^{\star}$. Therefore, it does not matter whether the estimators $\widehat{\alpha}_{1},\dots,\widehat{\alpha}_{K}$ are dependent or not.

## Thoughts
A little food for thought (on which I won't elaborate too much, since well, this is a blogpost).
<ul>
  <li>From the last line of \eqref{eq5}, we see that it doesn't matter whether the weight estimates are positive or negative. </li>
  <li>We also see that correlation between the weight estimates $\widehat{\alpha}_{K}$ does not influence the variance</li>
  <li>What is the most restrictive constraint here? The unbiasedness? The independence of the $t_k$'s ? </li>
</ul>	
 

### Footnotes
<a name="myfootnote1">1</a>: I could have written \eqref{eq1} as just $\mathbb{E}[\widehat{t}] = \mathbb{E}_{\alpha_{k}}[\mathbb{E}[\widehat{t} | \alpha_{1}, \dots, \alpha_{K}]] = \tau$. Wouldn't that be nicer. Compact notation is great if you know exactly what the writer's doing - I want to force the reader to recall all the assumptions that are being made.


## References

Rubin, D.B. and Weisberg, S., 1975. The variance of a linear combination of independent estimators using estimated weights. *Biometrika*, 62(3), pp.708-709.


<div id="disqus_thread"></div>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>


<p>Cited as:</p>
<pre tabindex="0"><code>@article{branchini2022combining,
  title   = Combining independent and unbiased estimators,
  author  = Branchini, Nicola,
  journal = https://www.branchini.fun,
  year    = 2022,
}
