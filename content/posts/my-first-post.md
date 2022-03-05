---
title: "My First Post"
date: 2022-02-06T20:42:28+01:00
---

In this post, my aims are:

- Introduce Bayesian inference in state space models
- Introduce approximate inference using importance sampling, in state space models. I will try to present and compare different ways of deriving the algorithms that I found in the literature, trying to unify them in a way that I have not seen explicitly elsewhere.
- Finally, describe the Auxiliary Particle Filter, its diverse intepretations and the recent Improved Auxiliary Particle Filter by Elvira et al [1]. I will illustrate the IAPF by reproducing the results of Elvira et al. [3].

The somewhat comprehensive "tutorial" and introduction to the topic arose from my feeling that the only content online that covers particle filters is either very theoretical material , or programming-oriented tutorials that completely lack motivation for how and why the algorithms are constructed as they are.

1. [Brief introduction to sequential inference](#introduction)
    1. [General Bayesian Filtering](#generalfilter)
    2. [Recursive Formulations](#recursive)
2. [Particle Filtering](#pf)
    1. [Basics of Monte Carlo and Importance Sampling](#basics)
    2. [Choice of proposal and variance of importance weights](#isproposal)
    3. [Sequential Importance Sampling](#sis)
    4. [Resampling](#resampling)
3. [Propagating particles by incoporating the current measurement](#apf)
    1. [The effect of using the locally optimal proposal](#optimalproposal)
    2. [The Auxiliary Particle Filter](#apf2)
        1. [A first intepretation: a standard SMC algorithm with a different $\gamma$](#firstapf)
        2. [The original intepretation of APF and Marginal Particle Filters](#marginalpf)
4. [The Multiple Importance Sampling Interpretation of PF](#mis)
    1. [The Improved Auxiliary Particle Filter](#iapf)

    ## Brief introduction to sequential inference <a name="introduction"></a>

    In Bayesian inference we want to update our beliefs on the state of some random variables, which could represent parameters of a parametric statistical model or represent some unobserved data generating process. Focussing on the "updating" perspective, the step to using Bayesian methods to represent dynamical systems is quite natural. The field of statistical signal processing has been using the rules of probabilities to model object tracking, navigation and even.. spread of infectious diseases.
    The probabilistic evolution of a dynamical system is often called a *state space model*. This is just an abstraction of how we think the state of the system evolves over time. Imagine we are tracking a robot's position (x,y coordinates) and bearing: these constitute a three dimensional vector. At some specific timestep, we can have a belief, i.e. a probability distribution that represents how likely we think the robot is currently assuming a certain bearing etc. If we start with a prior, and define some likelihood function/ sampling process that we believe generates what we observe, we can update our belief over the system's state with the rules of probabilty.
    Let the (uknown) state of the system at time $t$ be the vector valued random variable $\mathbf{s}_{t}$.

    We observe this state through a (noisy) measurement $\mathbf{v}_{t}$ (where v stands for visible).

    Now we have to start making more assumptions. What does our belief on $\mathbf{s}_{t}$ depend on ?

    Suprisingly to me, it turns out for **a lot** of applications it just needs to depend on the $\mathbf{s}$ tate at the previous timestep.
    In other words, we can say that $\mathbf{s}_{t}$ is sampled from some density $f$ conditional on $\mathbf{s}_{t-1}$:
