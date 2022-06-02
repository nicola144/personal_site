---
title: "(WORK IN PROGRESS) Combining estimators"
date: 2022-05-26T22:10:06Z
type: page
draft: true
---

I came across this 1.25 page paper by Don Rubin and Sanford Weisberg [(Rubin \& Weisberg)](https://academic.oup.com/biomet/article-abstract/62/3/708/257707) in Biometrika from 1975.
It considers the problem of finding the "best" linear combination (*whose weights sum to 1* !) of $K$ estimators of the *same* quantity. The estimators are all assumed to be unbiased. I think this is still a very much relevant topic; however, I won't try to convince you of this, because I want to keep this short.
If anything, it can definitely be seen as a nice exercise.

We let $\tau$ be the true, unknown quantity of interest. Estimators of $\tau$ will just be sub-indexed, as $\tau_1,\dots,\tau_K$.  We will assess the quality of the estimators by their mean squared error as usual. First, notice that the linear combination with weights summing to 1 will be unbiased:
$$

\tag{1}\label{eq1}
$$






Rubin, D.B. and Weisberg, S., 1975. The variance of a linear combination of independent estimators using estimated weights. Biometrika, 62(3), pp.708-709.
