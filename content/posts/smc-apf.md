---
title: "Sequential Monte Carlo and Improved Auxiliary Particle Filters"
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
Let the (uknown) state of the system at time $t$ be the vector valued random variable $\mathbf{s}\_{t}$.

We observe this state through a (noisy) measurement $\mathbf{v}\_{t}$ (where v stands for visible).

Now we have to start making more assumptions. What does our belief on $\mathbf{s}\_{t}$ depend on ?

Suprisingly to me, it turns out for **a lot** of applications it just needs to depend on the $\mathbf{s}$ tate at the previous timestep.

In other words, we can say that $\mathbf{s}\_{t}$ is sampled from some density $\color{cyan}{f}$ conditional on
:

$$ \color{cyan}{\text{Transition density}}: \qquad \mathbf{s}\_{t} \sim f(\mathbf{s}\_{t} \mid \mathbf{s}\_{t-1}) \tag{1}\label{eq1}
 $$

 Further, usually the observation or $\mathbf{v}$isible is sampled according to the current state:

 $$
 \color{LimeGreen}{\text{Observation density}}: \mathbf{v}\_{t} \sim \color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t}) \tag{2}\label{eq2}
 $$

 It is reasonable to assume this: if we take a measurement, we don't expect its outcome to be dependent on previous states of the system, just the current one ($\color{cyan}{f}$ and $\color{LimeGreen}{g}$ seem arbitrary but they are common in the literature). For example, a classic Gaussian likelihood for would imply that the belief over $\mathbf{v}\_{t}$ is a Normal , with the mean being a linear combination of the state's coordinates.

 This collection of random variables and densities defines the state space model completely. It is worth, if you see this for the first time, reflecting on the particular assumptions we are making. How the belief on $\mathbf{s}$ evolves with time could depend on many previous states; the measurement could depend on previous measurements, if we had a sensor that degrades over time, etc... I am not great at giving practical examples, but if you are reading this, you should be able to see that this can be generalized in several ways.
 Note that a lot of the structure comes from assuming some variables are (conditionally) independent from others. The field of probabilistic graphical models is dedicated to representing statistical independencies in the form of graphs (nodes and edges). One benefit of the graphical representation is that it makes immediately clear how flexible we could be. I am showing the graphical model for the described state space model in Figure 1, below.

 ![hmm](/hmm.svg)
*Fig. 1: Graphical model for the typical, first order Markov state space. Shaded nodes represent random variables whose value has been observed. In this model, each observation or "visible" is generated by some unobserved state at each time step.*

In short, when the transition density and the observation densities are linear combination of their inputs with additive, i.i.d. Gaussian noise, then the state space model is often called a Linear Dynamical System (LDS). When variables are discrete, it is often called Hidden Markov Model (HMM). These are just labels.

There are several tasks that we can perform on the state space model described above. Each of these has a fancy name, but one should of course recall that technically all we are doing is applying the sum and product rules. These tasks are associated to a *target* distribution which is the object of interest that we want to compute. Listing some of these:

  - **Filtering**: The target distributions are of the form: $p\left(\mathbf{s}\_{t} | \mathbf{v}\_{1: t}\right), \quad t=1, \ldots, T$. This represents what we have learnt about the system's state at time $t$, after observations up to $t$.<br>

  - **Smoothing**:  The target distributions are of the form: $p\left(\mathbf{s}\_{t} | \mathbf{v}\_{1: T}\right), \quad t=1, \ldots, T$. This represent what we have learnt about the system's state after observing the *complete* sequence of measurements, and revised the previous beliefs obtained by filtering. <br>

  - **Parameter Estimation**: The target distributions are of the form: $p\left(\boldsymbol{\theta} | \mathbf{v}\_{1: T}\right)= \int p\left(\mathbf{s}\_{0:T}, \boldsymbol{\theta} | \mathbf{v}\_{1: T}\right) \mathrm{d} \mathbf{s}\_{0: T}$. The parameters $\boldsymbol{\theta}$ represent all the parameters of any parametric densities in the state space model. In the case that the transition and/or observation densities are parametric, and parameters are unknown, we can learn them from data by choosing those that both explain the observations well and also agree with our prior beliefs. Parameter estimation is sometimes referred to as <i>learning</i>, because parameters describe properties of sensors that can be estimated from data with machine learning methods. In other words, it is called learning just because it is cool.

  ### General Bayesian Filtering <a name="generalfilter"></a>

  #### Some notation/terminology
  - As common in this field, I use the overloaded term of "distribution" to refer to densities, mass functions and distributions. Moreover the same notation is used for random variables and their realization ie. $$p(\mathbf{X} = \mathbf{x} \mid \mathbf{Z} = \mathbf{z}) = p(\mathbf{x} \mid \mathbf{z})$$
  - The notation $$\mathbf{v}\_{1:t}$$ means a collection of vectors $$ \left \{ \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_t \right \}$$
  - Therefore, $$ p\left ( \mathbf{v}\_{1:t} \right )$$ is a joint distribution: $$p\left ( \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_t \right ) $$
  - Integrating $$ \int p(\mathbf{x}\_{1:t}) \mathrm{d}\mathbf{x}\_{i:j}$$ means $$ \underbrace{\int \dots \int}\_{j-i+1} p(\mathbf{x}\_{1:t}) \mathrm{d}\mathbf{x}\_{i} \mathrm{d}\mathbf{x}\_{i+1} \dots \mathrm{d}\mathbf{x}\_{j} $$
  - The symbol $$:=$$ denotes a definition.

  In this post, I am only concerned with filtering, and will always assume that any parameters of <span style="color:blue">transition</span> or <span style="color:green">observation</span> densities are known in advance. There are classes of algorithms that learn the parameters and perform inference at the same time, such as Particle Markov Chain Monte Carlo or SMC2.

  Let's start by deriving the filtering distribution in the state space model described without many assumption on the distributions.
  Recall that the aim is to compute: $$ p\left(\mathbf{s}\_{t} | \mathbf{v}\_{1: t}\right)$$. Apply Bayes rule:  

  $$
  \require{cancel}
  p\left(\mathbf{s}\_{t} | \mathbf{v}\_{1:t}\right) = \frac{ \overbrace{p \left( \mathbf{v}\_{t} \mid \mathbf{s}\_{t}, \cancel{\mathbf{v}\_{1:t-1}} \right )}^{\mathbf{v}_t ~ \text{only dep. on} ~ \mathbf{s}_t} p\left( \mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1} \right ) }{p\left( \mathbf{v}_t \mid \mathbf{v}\_{1:t-1} \right )} = \frac{  \color{green}{g}\left( \mathbf{v}\_{t} \mid \mathbf{s}\_{t} \right ) p\left( \mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1} \right ) }{p\left( \mathbf{v}_t \mid \mathbf{v}\_{1:t-1} \right )} \tag{3}\label{eq3}
  $$

  If this equation is confusing, think of the previous measurements $$\mathbf{v}\_{1:t-1}$$ as just a "context", that is always on the conditioning side, a required "input" to all densities involved, with Bayes rule being applied to $$\mathbf{s}\_{t}$$ and $$\mathbf{v}\_{t}$$.
  We know the current measurements only depends on the state, therefore $$p \left( \mathbf{v}\_{t} \mid \mathbf{s}\_{t}, \mathbf{v}\_{1:t-1} \right ) = p \left( \mathbf{v}\_{t} \mid \mathbf{s}\_{t} \right ) = \color{green}{g}( \mathbf{v}\_{t} \mid \mathbf{s}\_{t} )$$, and only the term on the right side of the numerator is left to compute. This term is a marginal of $$ \mathbf{s}_t$$, which means we have to integrate out anything else. If we were doing this very naively, each time we would integrate out all previous states, but by caching results a.k.a. Dynamic Programming, we only need to marginalize the previous state:

  $$
    p\left( \mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1} \right ) = \int p\left( \mathbf{s}\_{t}, \mathbf{s}\_{t-1} \mid \mathbf{v}\_{1:t-1} \right ) \mathrm{d}\mathbf{s}\_{t-1}
  $$

  Continuing, we split the joint with the product rule and exploit that the states are independent of previous measurements:

  $$
  \begin{equation}\begin{aligned}
    &= \int p\left( \mathbf{s}\_{t} \mid  \mathbf{s}\_{t-1}, \cancel{\mathbf{v}\_{1:t-1}} \right ) p(\mathbf{s}\_{t-1} \mid \mathbf{v}\_{1:t-1}) \mathrm{d}\mathbf{s}\_{t-1} \\
    &= \int p\left( \mathbf{s}\_{t} \mid  \mathbf{s}\_{t-1} \right ) p(\mathbf{s}\_{t-1} \mid \mathbf{v}\_{1:t-1}) \mathrm{d}\mathbf{s}\_{t-1} \\
    &= \int \color{blue}{f}\left( \mathbf{s}\_{t} \mid  \mathbf{s}\_{t-1} \right ) p(\mathbf{s}\_{t-1} \mid \mathbf{v}\_{1:t-1}) \mathrm{d}\mathbf{s}\_{t-1}
  \end{aligned}\end{equation}\tag{4}\label{eq4}$$


  And we are done, if you notice that the right side term in the integral is the filtering distribution at $$t-1$$, which we have already computed recursively.
  In the literature names are given to the step that requires computing $$ p\left( \mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1} \right )$$ called *prediction*, because it's our belief on $$ \mathbf{s}\_{t}$$ before observing the currrent measurement, and *correction* is the name given to the step $$ p\left(\mathbf{s}\_{t} | \mathbf{v}\_{1:t}\right) \propto \color{green}{g}\left( \mathbf{v}\_{t} \mid \mathbf{s}\_{t} \right ) \cdot p\left( \mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1} \right )$$, because we "correct" our prediction by taking into account the measurement.

  In a LDS all computations have closed form solutions, and this algorithm instantiates into the *Kalman Filter*. For discrete  valued random variables, if the dimensionalities are small we can also do exact computations and the label this time is *Forward-Backward* algorithm for HMMs.
  When variables are non-Gaussian and/or transition/observation densities are nonlinear function of their inputs, we have to perform approximate inference.
  By far the most popular method is to use Monte Carlo approximations, and more specifically importance sampling. When we use importance sampling to approximate the filtering distribution, this is called *particle filtering*.
