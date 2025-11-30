---
title: "Importance sampling and a fake cure for the curse of dimensionality"
date: 2025-11-30T00:00:00Z
tags: ["Short", "Importance sampling", "Monte Carlo"]
draft: true
---

Here is a small toy example where plain Monte Carlo clearly suffers from the curse of dimensionality, while a suitably chosen importance sampler does the exact opposite: its variance *shrinks* with the dimension.

It is contrived, of course, but it is a good sanity check on how variance scales and on how much freedom you actually have when you are allowed to pick the proposal.

## Setup

Let the target be a zero mean Gaussian on $\mathbb R^D$

$$
p(x) = \mathcal N\bigl(0, \sigma_p^2 I_D\bigr),
$$

and consider the function

$$
f(x) = \prod_{d=1}^D x_d .
$$

We want

$$
\mu = \mathbb E_p[f(X)] .
$$

The component marginals are independent and identically distributed, with

$$
\mathbb E_p[X_d] = 0, \qquad
\mathbb E_p[X_d^2] = \sigma_p^2 .
$$

## Plain Monte Carlo

The usual estimator is

$$
\hat\mu_N^{MC}
= \frac{1}{N}\sum_{n=1}^N f\bigl(X^{(n)}\bigr),
\qquad X^{(n)} \sim p \text{ i.i.d.}
$$

Since each component has mean zero,

$$
\mu
= \mathbb E_p\Bigl[\prod_{d=1}^D X_d\Bigr]
= \prod_{d=1}^D \mathbb E_p[X_d]
= 0 .
$$

The variance of a single draw is

$$
\operatorname{Var}_p\bigl(f(X)\bigr)
= \mathbb E_p\Bigl[\prod_{d=1}^D X_d^2\Bigr]
= \prod_{d=1}^D \mathbb E_p[X_d^2]
= (\sigma_p^2)^D .
$$

So

$$
\operatorname{Var}_p\bigl(\hat\mu_N^{MC}\bigr)
= \frac{1}{N}(\sigma_p^2)^D .
$$

If \(\sigma_p^2 > 1\) this explodes like \((\sigma_p^2)^D\). That is the usual curse.

## Importance sampling

Now introduce a proposal

$$
q(x) = \mathcal N\bigl(0, \sigma_q^2 I_D\bigr)
$$

and the usual unbiased importance sampling estimator

$$
\hat\mu_N^{IS}
= \frac{1}{N}\sum_{n=1}^N
f\bigl(X^{(n)}\bigr)\,
\frac{p\bigl(X^{(n)}\bigr)}{q\bigl(X^{(n)}\bigr)},
\qquad X^{(n)} \sim q \text{ i.i.d.}
$$

Unbiasedness is standard:

$$
\mathbb E_q\Bigl[
f(X)\frac{p(X)}{q(X)}
\Bigr]
= \mathbb E_p[f(X)] = \mu .
$$

The interesting part is the variance.

Because both \(p\) and \(q\) factorise over coordinates, we can write

$$
p(x) = \prod_{d=1}^D p_d(x_d), \qquad
q(x) = \prod_{d=1}^D q_d(x_d),
$$

with one dimensional marginals \(p_d = \mathcal N(0,\sigma_p^2)\), \(q_d = \mathcal N(0,\sigma_q^2)\).

Define the one dimensional random variables

$$
Y_d = X_d\,\frac{p_d(X_d)}{q_d(X_d)}, \qquad X_d \sim q_d.
$$

Then the full importance sampling weight times integrand is

$$
Y = \prod_{d=1}^D Y_d .
$$

One checks that \(\mathbb E_q[Y_d] = 0\), hence \(\mathbb E_q[Y] = 0\) and

$$
\operatorname{Var}_q(Y)
= \mathbb E_q\Bigl[\prod_{d=1}^D Y_d^2\Bigr]
= \prod_{d=1}^D \mathbb E_q[Y_d^2]
= \prod_{d=1}^D \kappa(\sigma_p^2,\sigma_q^2),
$$

where I have defined the one dimensional variance factor

$$
\kappa(\sigma_p^2,\sigma_q^2)
= \mathbb E_{q_d}\bigl[Y_d^2\bigr]
= \int x^2 \frac{p_d(x)^2}{q_d(x)}\,dx .
$$

Therefore

$$
\operatorname{Var}_q\bigl(\hat\mu_N^{IS}\bigr)
= \frac{1}{N}\,\kappa(\sigma_p^2,\sigma_q^2)^D .
$$

Everything now hinges on the size of \(\kappa\).

## Computing \(\kappa(\sigma_p^2,\sigma_q^2)\)

In one dimension we have

$$
p_d(x) = \frac{1}{\sqrt{2\pi}\sigma_p}
\exp\!\Bigl(-\frac{x^2}{2\sigma_p^2}\Bigr),
\qquad
q_d(x) = \frac{1}{\sqrt{2\pi}\sigma_q}
\exp\!\Bigl(-\frac{x^2}{2\sigma_q^2}\Bigr).
$$

A short but slightly tedious Gaussian calculation gives

$$
\kappa(\sigma_p^2,\sigma_q^2)
= \int x^2 \frac{p_d(x)^2}{q_d(x)}\,dx
= \frac{\sigma_p \sigma_q^4}
       {\bigl(2\sigma_q^2 - \sigma_p^2\bigr)^{3/2}},
$$

which exists as long as

$$
2\sigma_q^2 > \sigma_p^2, \qquad
\sigma_p^2 > 0, \ \sigma_q^2 > 0.
$$

(The intermediate representation in terms of a helper variance \(\Delta^2\) also requires \(\sigma_q^2 > \sigma_p^2\), which is a stricter but safe condition.)

Plugging this back in,

$$
\operatorname{Var}_q\bigl(\hat\mu_N^{IS}\bigr)
= \frac{1}{N}
\left(
\frac{\sigma_p \sigma_q^4}
     {(2\sigma_q^2 - \sigma_p^2)^{3/2}}
\right)^{\!D}.
$$

Compare this with the plain Monte Carlo variance

$$
\operatorname{Var}_p\bigl(\hat\mu_N^{MC}\bigr)
= \frac{1}{N}(\sigma_p^2)^D .
$$

Per dimension, plain MC multiplies the variance by \(\sigma_p^2\), whereas IS multiplies by \(\kappa(\sigma_p^2,\sigma_q^2)\).

If we pick \(\sigma_p^2 > 1\) and \(\sigma_q^2\) such that

$$
\kappa(\sigma_p^2,\sigma_q^2) < 1,
$$

then as \(D\) increases

- plain MC variance explodes like \((\sigma_p^2)^D\);
- IS variance shrinks like \(\kappa(\sigma_p^2,\sigma_q^2)^D\).

For example, with \(\sigma_p^2 = 1.2\) and \(\sigma_q^2 = 2\) one gets

$$
\kappa(1.2, 2) \approx 0.94 < 1,
$$

so the IS variance improves exponentially with the dimension, while plain MC gets exponentially worse.

This obviously does not mean that importance sampling magically fixes real high dimensional problems. It only says that the curse is not a universal law that applies regardless of how you sample. With the wrong proposal you can easily do strictly worse than plain MC; with a very carefully chosen proposal you can do absurdly well.

## Visualising \(\kappa\)

Here is a simple Python script that plots \(\kappa(\sigma_p^2,\sigma_q^2)\) over a grid and marks the existence boundary \(2\sigma_q^2 = \sigma_p^2\) and the contour \(\kappa = 1\).


<div id="disqus_thread"></div>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>


<p>Cited as:</p>
<pre tabindex="0"><code>@article{branchini2025isdimension,
  title   = Curse of dimension in Monte Carlo and importance sampling to the rescue: a contrived toy example,
  author  = Branchini, Nicola,
  journal = https://www.branchini.fun,
  year    = 2025,
}


<!-- 
```python
import numpy as np
import matplotlib.pyplot as plt

def kappa(sig_p2, sig_q2):
    """
    κ(σ_p^2, σ_q^2) = σ_p * σ_q^4 / (2σ_q^2 - σ_p^2)^(3/2)
    Defined only where 2σ_q^2 > σ_p^2 and σ_p^2, σ_q^2 > 0.
    sig_p2, sig_q2 can be scalars or numpy arrays.
    """
    sig_p = np.sqrt(sig_p2)
    denom = 2 * sig_q2 - sig_p2

    out = np.empty_like(sig_p2, dtype=float)
    out[:] = np.nan

    mask = (denom > 0) & (sig_p2 > 0) & (sig_q2 > 0)
    out[mask] = sig_p[mask] * sig_q2[mask]**2 / denom[mask]**1.5
    return out

# Grid over (σ_p^2, σ_q^2)
sig_p2_vals = np.linspace(0.5, 5.0, 200)   # vertical axis
sig_q2_vals = np.linspace(0.5, 10.0, 400)  # horizontal axis
Sp2, Sq2 = np.meshgrid(sig_p2_vals, sig_q2_vals, indexing='ij')

K = kappa(Sp2, Sq2)

fig, ax = plt.subplots(figsize=(7, 5))

cs = ax.contourf(Sq2, Sp2, K, levels=50)
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label(r'$\kappa(\sigma_p^2,\sigma_q^2)$')

# Existence boundary: 2 σ_q^2 = σ_p^2
boundary_p2 = np.linspace(sig_p2_vals.min(), sig_p2_vals.max(), 200)
boundary_q2 = 0.5 * boundary_p2
ax.plot(boundary_q2, boundary_p2, linestyle='--')

# Contour where κ = 1
cs2 = ax.contour(Sq2, Sp2, K, levels=[1.0])
ax.clabel(cs2, fmt={1.0: r'$\kappa=1$'})

ax.set_xlabel(r'$\sigma_q^2$')
ax.set_ylabel(r'$\sigma_p^2$')
ax.set_title(r'$\kappa(\sigma_p^2,\sigma_q^2)$ in the region $2\sigma_q^2 > \sigma_p^2$')

plt.tight_layout()
plt.show() -->
