---
title: "(WORK IN PROGRESS) Combining independent and unbiased estimators"
date: 2022-05-26T22:10:06Z
type: page
draft: true
---

I came across this 1.25 page paper by Don Rubin and Sanford Weisberg [(Rubin \& Weisberg)](https://academic.oup.com/biomet/article-abstract/62/3/708/257707) in Biometrika from 1975.
It considers the problem of finding the "best" linear combination (*whose weights sum to 1* !) of $K$ estimators of the *same* quantity. The estimators are all assumed to be unbiased. I think this is still a very much relevant topic; however, I won't try to convince you of this, because I want to keep this short.
If anything, it can be seen as a nice little exercise.

We let $\tau$ be the true, unknown quantity of interest. Estimators of $\tau$ will just be sub-indexed, as $\tau_1,\dots,\tau_K$. These are *independent* (not necessarily i.i.d.) and *unbiased*.  We will assess the quality of the estimators by their mean squared error. We now define an estimator: $\widehat{\tau} := \sum\_{k=1}^{K} \hat{\alpha}_k \tau_k $, with the weights $\hat{\alpha}_k$ be ***random variables*** and such that $\sum\_{k=1}^{K} \widehat{\alpha\_{k}} = 1$. Why did they define weights as random variables ? Probably as we shall see, because the optimal weights involve a quantity that needs to be estimated. That is, $\widehat{\tau}$ is the estimator we can *actually* use, and we will compare it to some intractable optimal solution.  \\
The $\widehat{\tau}$ estimator is unbiased by applying the law of iterated expectation:
$$
\mathbb{E}\_{\widehat{\tau}} = \mathbb{E}\_{\bigotimes \mathbf{P}\_{\alpha\_{k}}}[\mathbb{E}\_{\mathbf{P}\_\widehat{t}} \widehat{t} \mid ]
\tag{1}\label{eq1}
$$


## Thoughts


## References

Rubin, D.B. and Weisberg, S., 1975. The variance of a linear combination of independent estimators using estimated weights. Biometrika, 62(3), pp.708-709.

<p>Cited as:</p>
<pre tabindex="0"><code>@article{branchini2022is,
  title   = Combining independent and unbiased estimators,
  author  = Branchini, Nicola,
  journal = https://www.branchini.fun,
  year    = 2022,
}
