---
title: "(WORK IN PROGRESS) Importance Sampling: more than numerical integration"
---
*Disclaimer for the Importance Sampling - expert reader: I will be using the term in a very broad sense.*

This post is not a generic introduction to Importance Sampling. It is an overview of many of the places where the key ideas behind the methodology are used.

## Numerical integration
The general ideas that an average approximates an expectation is pervasive. Similarly so, is the idea that observations (data, samples, etc.) can be modelled as realizations of an underlying random variable. The idea of Monte Carlo is related to both.
Traditionally, Monte Carlo methods come up in the context of numerical integration. Here, the problem to solve is simply to approximate the value of an integral:
$$
\int h(\mathbf{x}) \mathrm{d}\mathbf{x}
\tag{1}\label{eq1}
$$
 Simple deterministic algorithms like *[Simpson's](https://en.wikipedia.org/wiki/Simpson%27s_rule)* or the *[trapezoid](https://en.wikipedia.org/wiki/Trapezoidal_rule)* rules scale terribly with the dimension of the integration variable. These rely on dividing the space into grids, which is not a good idea when the dimension increases.
 Monte Carlo provides a framework to develop *randomized* algorithms that are more efficient, theoretically and practically. Often, the integral of interest is already in the form of an expectation:
 $$
 \int h(\mathbf{x}) \mathrm{d}\mathbf{x} = \int f(\mathbf{x}) \cdot \pi(\mathbf{x}) \mathrm{d}\mathbf{x} .
 \tag{2}\label{eq2}
 $$
 Probabilities are also special cases of expectations. In these cases, it is natural to think of generating points distributed according to $\pi(\mathbf{x})$: this leads to approximating \eqref{eq2} with an arithmetic average (convenient), and many things can be proved about this solution (also convenient). The first obvious fact about this solution is that it is unbiased. When the integral of interest is *not* an expectation (as in the more general \eqref{eq1}), things become more interesting.

## Numerical integration
  The idea underlying IS arises naturally in this setting.  The key idea is: we want to generate (or obtain from someone else) points *randomly* (instead of deterministically - this is where the Monte Carlo comes in) such that the integral is well approximated. To achieve this, these points ought to be in regions where the integrand has large values.  
  The IS-savvy reader may say that the generated points need to follow a distribution of some known form. That a density should be available. Maybe the Radon-Nykodim derivative needs to exist, and absolute continuity conditions, etc. I want to take a broader view, that allows me to consider "IS" a method even if some of these conditions are relaxed. For example, the IS weights may not be computable. Some may say, then it's not IS - I'm fine with this too, let's not dabble around semantics too much. [^1]

  [^1]: This is the first footnote.

  I could have just said I am thinking of a randomized algorithm for numerical integration. By the way, numerical integration is cool and fancy: some argue for studying the performance of the popular neural network ensembles (training a neural network multiple times, and averaging the results) *[via the lens of numerical integration](https://cims.nyu.edu/~andrewgw/deepensembles/)*.
