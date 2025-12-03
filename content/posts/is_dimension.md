---
title: "Curse of dimension in Monte Carlo and importance sampling to the rescue: a contrived toy example"
date: 2025-11-30T00:00:00Z
tags: ["Short", "importance sampling", "monte carlo"]
draft: true
---

The folklore story is that importance sampling "suffers from the curse of dimensionality". In practice that is often true. What is less often stated is that plain Monte Carlo can be just as bad, or worse.

In this post I go an example where

1. plain Monte Carlo has a variance that explodes with the dimension, and  
2. a simple choice of importance sampling proposal leads to a variance that *shrinks* with the dimension.

It is most certainly a silly example and does not have any aim of representing anything more realistic. I just find it an interesting curiosity.

## Setup

Let dimension $D \ge 1$. For a variance parameter $\sigma_p^2 > 0$ define the target

$$
p(x; \sigma_p^2) = \mathcal N\bigl(0, \sigma_p^2 I_D\bigr),
\qquad x \in \mathbb R^D.
$$

Consider the function

$$
f(x) = \prod_{d=1}^D x_d .
$$

We want to estimate the expectation, 

$$
\mu = \mathbb E_p[f(x)] .
$$

The coordinates are independent and identically distributed under $p(\cdot; \sigma_p^2)$, with

$$
\mathbb E_p[x_d] = 0, \,
\qquad
\mathbb E_p[x_d^2] = \sigma_p^2 .
$$

The interesting regime is $\sigma_p^2 > 1$.

## Plain Monte Carlo

The plain Monte Carlo estimator is

$$
\hat\mu_N^{mc}
= \frac{1}{N}\sum_{n=1}^N f\bigl(x^{(n)}\bigr),
\qquad x^{(n)} \sim p(\cdot; \sigma_p^2) \text{ i.i.d.}
$$

Since each coordinate has mean zero,

$$
\mu
= \mathbb E_p\Bigl[\prod_{d=1}^D x_d\Bigr]
= \prod_{d=1}^D \mathbb E_p[x_d]
= 0 .
$$

The variance of one draw is

$$
\operatorname{Var}_p(f(x))
= \mathbb E_p\Bigl[\prod_{d=1}^D x_d^2\Bigr]
= \prod_{d=1}^D \mathbb E_p[x_d^2]
= (\sigma_p^2)^D .
$$

So it follows easily 

$$
\operatorname{Var}_p\bigl(\hat\mu_N^{mc}\bigr)
= \frac{1}{N}(\sigma_p^2)^D .
$$

If $\sigma_p^2 > 1$ this explodes like $(\sigma_p^2)^D$. That is the "curse of dimensionality" here. 

## Importance sampling

Now let us bring in a proposal. For a second variance parameter $\sigma_q^2 > 0$ define

$$
q(x; \sigma_q^2) = \mathcal N\bigl(0, \sigma_q^2 I_D\bigr).
$$

The usual importance sampling estimator is

$$
\hat\mu_N^{is}
= \frac{1}{N}\sum_{n=1}^N
f\bigl(x^{(n)}\bigr)\,
\frac{p\bigl(x^{(n)}; \sigma_p^2\bigr)}{q\bigl(x^{(n)}; \sigma_q^2\bigr)},
\qquad x^{(n)} \sim q(\cdot; \sigma_q^2) \text{ i.i.d.}
$$

Unbiasedness is standard:

$$
\mathbb E_q\Bigl[
f(x)\frac{p(x; \sigma_p^2)}{q(x; \sigma_q^2)}
\Bigr]
= \mathbb E_p[f(x)] = \mu .
$$

The only non-trivial part is the variance.

Both $p(\cdot; \sigma_p^2)$ and $q(\cdot; \sigma_q^2)$ factorise:

$$
p(x; \sigma_p^2) = \prod_{d=1}^D p_d(x_d; \sigma_p^2), \,
\qquad
q(x; \sigma_q^2) = \prod_{d=1}^D q_d(x_d; \sigma_q^2),
$$

where each marginal is one dimensional,

$$
p_d(x_d; \sigma_p^2) = \mathcal N(0,\sigma_p^2), \,
\qquad
q_d(x_d; \sigma_q^2) = \mathcal N(0,\sigma_q^2),
$$

and is the same for all $d$.

Define one dimensional random variables

$$
y_d = x_d\,\frac{p_d(x_d; \sigma_p^2)}{q_d(x_d; \sigma_q^2)}, \,
\qquad x_d \sim q_d(\cdot; \sigma_q^2).
$$

Then the weight times integrand is

$$
y = \prod_{d=1}^D y_d .
$$

One can check that $\mathbb E_q[y_d] = 0$. Therefore $\mathbb E_q[y] = 0$ and

$$
\operatorname{Var}_q(y)
= \mathbb E_q\Bigl[\prod_{d=1}^D y_d^2\Bigr]
= \prod_{d=1}^D \mathbb E_q[y_d^2]
= \prod_{d=1}^D \kappa(\sigma_p^2,\sigma_q^2),
$$

where the one dimensional variance factor is

$$
\kappa(\sigma_p^2,\sigma_q^2)
= \mathbb E_q[y_1^2]
= \int x^2 \frac{p_1(x; \sigma_p^2)^2}{q_1(x; \sigma_q^2)}\,\mathrm{d}x .
$$

Therefore

$$
\operatorname{Var}_q\bigl(\hat\mu_N^{is}\bigr)
= \frac{1}{N}\,\kappa(\sigma_p^2,\sigma_q^2)^D .
$$

Everything now depends on the size of $\kappa(\sigma_p^2,\sigma_q^2)$.

## computing $\kappa(\sigma_p^2,\sigma_q^2)$

In one dimension the densities are

$$
p_1(x; \sigma_p^2) = \frac{1}{\sqrt{2\pi}\sigma_p}
\exp\Bigl(-\frac{x^2}{2\sigma_p^2}\Bigr), \,
\qquad
q_1(x; \sigma_q^2) = \frac{1}{\sqrt{2\pi}\sigma_q}
\exp\Bigl(-\frac{x^2}{2\sigma_q^2}\Bigr).
$$

A short Gaussian calculation gives

$$
\kappa(\sigma_p^2,\sigma_q^2)
= \int x^2 \frac{p_1(x; \sigma_p^2)^2}{q_1(x; \sigma_q^2)}\,\mathrm{d}x
= \frac{\sigma_p \sigma_q^4}
       {\bigl(2\sigma_q^2 - \sigma_p^2\bigr)^{3/2}},
$$

which is finite if

$$
2\sigma_q^2 > \sigma_p^2, \,
\qquad
\sigma_p^2 > 0, \,
\quad
\sigma_q^2 > 0.
$$

Putting this back in,

$$
\operatorname{Var}_q\bigl(\hat\mu_N^{is}\bigr)
= \frac{1}{N}
\left(
\frac{\sigma_p \sigma_q^4}
     {(2\sigma_q^2 - \sigma_p^2)^{3/2}}
\right)^{\!D}.
$$

For comparison, plain Monte Carlo has

$$
\operatorname{Var}_p\bigl(\hat\mu_N^{mc}\bigr)
= \frac{1}{N}(\sigma_p^2)^D .
$$

Per dimension, plain Monte Carlo multiplies the variance by $\sigma_p^2$, whereas importance sampling multiplies by $\kappa(\sigma_p^2,\sigma_q^2)$.

If we pick $\sigma_p^2 > 1$ and $\sigma_q^2$ such that

$$
\kappa(\sigma_p^2,\sigma_q^2) < 1,
$$

then as $D$ increases

- plain Monte Carlo variance explodes like $(\sigma_p^2)^D$;
- importance sampling variance shrinks like $\kappa(\sigma_p^2,\sigma_q^2)^D$.

For example, with $\sigma_p^2 = 1.2$ and $\sigma_q^2 = 2$ one gets

$$
\kappa(1.2, 2) \approx 0.94 < 1,
$$

so the importance sampler gets better and better with the dimension, while the plain Monte Carlo estimator gets worse and worse.

This does not mean that importance sampling will fix a real high dimensional problem. It only shows that the blanket statement "importance sampling always suffers from a curse of dimension" is false. As usual, everything depends on how you choose the proposal.

## link to the optimal proposal

Classical importance sampling theory says that, for a given integrand $f$ and target density $p$, the proposal that minimises the variance of the (unnormalised) estimator is proportional to $|f(x)|\,p(x)$, up to a normalising constant and sign tricks.

In our example

$$
f(x) = \prod_{d=1}^D x_d, \,
\qquad
p(x; \sigma_p^2) = \prod_{d=1}^D p_d(x_d; \sigma_p^2),
$$

so the idealised proposal would be

$$
q^{\star}(x) \propto |f(x)|\,p(x; \sigma_p^2)
= \prod_{d=1}^D |x_d|\,p_d(x_d; \sigma_p^2).
$$

This is still factorised over coordinates, but each marginal is tilted by a factor $|x_d|$. Intuitively, $q^{\star}$ spends more time in regions where $|x_d|$ is large, which is exactly where the magnitude of $f(x)$ is large. Sampling from $q^{\star}$ would typically have much smaller variance than sampling directly from $p(\cdot; \sigma_p^2)$.

Our Gaussian proposal family $\{q(\cdot; \sigma_q^2)\}$ is a crude approximation to that idea: by taking $\sigma_q^2 > \sigma_p^2$ we fatten the tails in each coordinate, which partially mimics the effect of the $|x_d|$ factor. The calculation above shows that, in this artificial setting, this is enough to turn the curse of dimensionality into a "blessing" for importance sampling.

The point is not that this toy family is good in practice. The point is that the curse lives in the mismatch between the proposal and the optimal importance density, not in importance sampling itself.

## visualising $\kappa$

Here is a simple Python script that plots $\kappa(\sigma_p^2,\sigma_q^2)$ over a grid and marks the contour $\kappa = 1$. In the region below that dark red curve we have $\kappa < 1$, which is the "blessing of dimensionality" region for this toy.


![Visualization of $\kappa(\sigma_p^2,\sigma_q^2)$](/kappa.png) 


<div id="disqus_thread"></div> <noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript> <p>Cited as:</p> <pre tabindex="0"><code>@article{branchini2025isdimension, title = Curse of dimension in Monte Carlo and importance sampling to the rescue: a contrived toy example, author = Branchini, Nicola, journal = https://www.branchini.fun, year = 2025, }
