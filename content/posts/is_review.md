---
title: "(WORK IN PROGRESS) Importance Sampling: (much!) more than numerical integration"
date: 2022-05-26T22:10:06Z
type: page
draft: true
---
*Disclaimer for the Importance Sampling - expert reader: I will be using the term in a very broad sense.*

This post is **not** a generic introduction to Importance Sampling. It is an overview of many of the places where the key ideas behind the methodology are used.

### Numerical integration
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

### Importance Sampling as a randomized algorithm for numerical integration
The idea underlying IS arises naturally in this setting. That is: we want to generate (or obtain from someone else) points *randomly* (as opposed to deterministically - this is where the Monte Carlo comes in) such that the integral is well approximated. To achieve this, these points ought to be in regions where the integrand has large values.  
The IS-savvy reader may say that the generated points need to follow a distribution of some known form. That a density should be available. Maybe the Radon-Nykodim derivative needs to exist, and absolute continuity conditions need to hold, etc. etc. I want to take a broader view, that allows me to consider "IS" a method even if some of these conditions are relaxed. The only one I don't want to relax is that points need to be generated randomly. For example, the IS weights may not be computable. Some may say, then it's not IS - I'm fine with this too, let's not dabble around semantics too much.   

### Beyond (explicit) numerical integration

Let me start going through the applications, beyond explicit numerical integration, where the IS idea is key.
A classic example is Reinforcement Learning, where the objective function (that needs to be maximized, and not estimated) is an expectation w.r.t. , among other things, something that can be controlled by the algorithm (i.e. , the *policy* of the agent). Here, the IS idea is used to either estimate gradients of this objective function (see e.g. [Tang \& Abbeel 2010, NeurIPS](https://proceedings.neurips.cc/paper/2010/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html))
