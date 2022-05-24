---
title: "(WORK IN PROGRESS) Importance Sampling: more than numerical integration"
---

The general ideas that an average approximates an expectation is pervasive. Similarly so is the idea that observations (data, samples, etc.) can be modelled as realizations of an underlying random variable.  

Traditionally, Monte Carlo methods come up in the context of numerical integration. Here, the problem to solve is simply to approximate the value of an integral:
$$
\int h(\mathbf{x}) \mathrm{d}\mathbf{x}
\tag{1}\label{eq1}
$$
 Simple deterministic algorithms like Simpson's rule or the trapezoid method scale terribly with the dimension of the integration variable. Most of these methods rely on dividing the space into grids, which is not a good idea when the dimension increases.
 Monte Carlo provides a framework to develop randomized algorithms that are more efficient, theoretically and practically, in many contexts of interest. Often, the integral of interest is already in the form of an expectation:
 $$
 \int h(\mathbf{x}) \mathrm{d}\mathbf{x} = \int f(\mathbf{x}) \cdot \pi(\mathbf{x}) \mathrm{d}\mathbf{x} .
 \tag{2}\label{eq2}
 $$
 In this case, it is natural to think of generating points distributed according to $\pi(\mathbf{x})$: this leads to approximating \eqref{eq2} with an average (convenient), and many things can be proved about this solution (also convenient). The first thing immediately obvious about this solution is that it is unbiased. When the integral of interest is not an expectation (as in the more general \eqref{eq1}), things become more interesting.

# Numerical integration
  The idea underlying IS kicks in in this setting. Having freed ourselves from the blinding of some $\pi$ within the integrand, we can focus on what matters. That is: we just want to generate points randomly (instead of deterministically - this is where the Monte Carlo comes in) such that the integrand is well approximated. To achieve this, we want these points to be in regions where the integrand has large values.
 This is maybe a bit abstract.  But so is the standard presentation, which goes as the following.

    by multiplying the integrand in \eqref{eq1} by $1 = \frac{q(\mathbf{x})}{q(\mathbf{x}}$, and then it is seen as an expectation over $q(\mathbf{x})$. I prefer to view it in the more general "generating samples randomly where the integrand has large values" - way. This is because the standard way can mislead you to think that your only option to form an estimator is simply to repeat what you did in plain Monte Carlo. Two quick comments about the restrictiveness of this view: (1) we need not restrict to generate points that follow a known (analytical) distribution (see e.g. )
