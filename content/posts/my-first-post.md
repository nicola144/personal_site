---
title: "My First Post"
date: 2022-02-06T20:42:28+01:00
---

In this post, my aims are:
- Introduce Bayesian inference in state space models
- Introduce approximate inference using importance sampling, in state space models. I will try to present and compare different ways of deriving the algorithms that I found in the literature, trying to unify them in a way that I have not seen explicitly elsewhere.
- Finally, describe the Auxiliary Particle Filter, its diverse intepretations and the recent Improved Auxiliary Particle Filter by Elvira et al [ 1 ]. I will illustrate the IAPF by reproducing the results of Elvira et al. [3].

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

Suprisingly to me, it turns out for **a lot** of applications it just needs to depend on the $\mathbf{s}$tate at the previous timestep.
In other words, we can say that \( \mathbf{s}_{t} \) is sampled from some density \(f\) conditional on \( \mathbf{s}_{t-1} \):
\[ \color{blue}{\text{Transition density}}: \qquad \mathbf{s}_{t} \sim \color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1})
\tag{1} \label{eq1} \]
Further, usually the observation or $\mathbf{v}$isible is sampled according to the current state:
\[ \color{green}{\text{Observation density}}: \mathbf{v}_{t} \sim \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}) \tag{2}\label{eq2} \]

It is reasonable to assume this: if we take a measurement, we don't expect its outcome to be dependent on previous states of the system, just the current one ($\color{blue}{f}$ and $\color{green}{g}$ seem arbitrary but they are common in the literature). For example, a classic Gaussian likelihood for would imply that the belief over $\mathbf{v}_{t}$ is a Normal , with the mean being a linear combination of the state's coordinates.

This collection of random variables and densities defines the state space model completely. It is worth, if you see this for the first time, reflecting on the particular assumptions we are making. How the belief on $\mathbf{s}$ evolves with time could depend on many previous states; the measurement could depend on previous measurements, if we had a sensor that degrades over time, etc... I am not great at giving practical examples, but if you are reading this, you should be able to see that this can be generalized in several ways.
Note that a lot of the structure comes from assuming some variables are (conditionally) independent from others. The field of probabilistic graphical models is dedicated to representing statistical independencies in the form of graphs (nodes and edges). One benefit of the graphical representation is that it makes immediately clear how flexible we could be. I am showing the graphical model for the described state space model in Figure 1, below.


![hhm]({{ '/assets/images/tikz2.svg' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1: Graphical model for the typical, first order Markov state space. Shaded nodes represent random variables whose value has been observed. In this model, each observation or "visible" is generated by some unobserved state at each time step.*

In short, when the transition density and the observation densities are linear combination of their inputs with additive, i.i.d. Gaussian noise, then the state space model is often called a Linear Dynamical System (LDS). When variables are discrete, it is often called Hidden Markov Model (HMM). These are just labels.

There are several tasks that we can perform on the state space model described above. Each of these has a fancy name, but one should of course recall that technically all we are doing is applying the sum and product rules. These tasks are associated to a *target* distribution which is the object of interest that we want to compute. Listing some of these:

<p>
  <b>Filtering</b>: The target distributions are of the form: $p\left(\mathbf{s}_{t} | \mathbf{v}_{1: t}\right), \quad t=1, \ldots, T$. This represents what we have learnt about the system's state at time $t$, after observations up to $t$.<br>

  <b>Smoothing</b>:  The target distributions are of the form: $p\left(\mathbf{s}_{t} | \mathbf{v}_{1: T}\right), \quad t=1, \ldots, T$. This represent what we have learnt about the system's state after observing the *complete* sequence of measurements, and revised the previous beliefs obtained by filtering. <br>

<b>Parameter Estimation</b>: The target distributions are of the form: $p\left(\boldsymbol{\theta} | \mathbf{v}_{1: T}\right)= \int p\left(\mathbf{s}_{0: T}, \boldsymbol{\theta} | \mathbf{v}_{1: T}\right) \mathrm{d} \mathbf{s}_{0: T}$. The parameters $\boldsymbol{\theta}$ represent all the parameters of any parametric densities in the state space model. In the case that the transition and/or observation densities are parametric, and parameters are unknown, we can learn them from data by choosing those that both explain the observations well and also agree with our prior beliefs. Parameter estimation is sometimes referred to as <i>learning</i>, because parameters describe properties of sensors that can be estimated from data with machine learning methods. In other words, it is called learning just because it is cool. <br>
</p>
