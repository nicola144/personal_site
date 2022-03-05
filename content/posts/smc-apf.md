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
  - As common in this field, I use the overloaded term of "distribution" to refer to densities, mass functions and distributions. Moreover the same notation is used for random variables and their realization ie. $p(\mathbf{X} = \mathbf{x} \mid \mathbf{Z} = \mathbf{z}) = p(\mathbf{x} \mid \mathbf{z})$
  - The notation $\mathbf{v}\_{1:t}$ means a collection of vectors $ \left \{ \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_t \right \}$
  - Therefore, $ p\left ( \mathbf{v}\_{1:t} \right )$ is a joint distribution: $p\left ( \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_t \right ) $
  - Integrating $ \int p(\mathbf{x}\_{1:t}) \mathrm{d}\mathbf{x}\_{i:j}$ means $ \underbrace{\int \dots \int}\_{j-i+1} p(\mathbf{x}\_{1:t}) \mathrm{d}\mathbf{x}\_{i} \mathrm{d}\mathbf{x}\_{i+1} \dots \mathrm{d}\mathbf{x}\_{j} $
  - The symbol $:=$ denotes a definition.

  In this post, I am only concerned with filtering, and will always assume that any parameters of <span style="color:cyan">transition</span> or <span style="color:LimeGreen">observation</span> densities are known in advance. There are classes of algorithms that learn the parameters and perform inference at the same time, such as Particle Markov Chain Monte Carlo or SMC2.

  Let's start by deriving the filtering distribution in the state space model described without many assumption on the distributions.
  Recall that the aim is to compute: $ p\left(\mathbf{s}\_{t} | \mathbf{v}\_{1: t}\right)$. Apply Bayes rule:  

  $$
  \require{cancel}
  p\left(\mathbf{s}\_{t} | \mathbf{v}\_{1:t}\right) = \frac{ \overbrace{p \left( \mathbf{v}\_{t} \mid \mathbf{s}\_{t}, \cancel{\mathbf{v}\_{1:t-1}} \right )}^{\mathbf{v}_t ~ \text{only dep. on} ~ \mathbf{s}_t} p\left( \mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1} \right ) }{p\left( \mathbf{v}_t \mid \mathbf{v}\_{1:t-1} \right )} = \frac{  \color{LimeGreen}{g}\left( \mathbf{v}\_{t} \mid \mathbf{s}\_{t} \right ) p\left( \mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1} \right ) }{p\left( \mathbf{v}_t \mid \mathbf{v}\_{1:t-1} \right )} \tag{3}\label{eq3}
  $$

  If this equation is confusing, think of the previous measurements $\mathbf{v}\_{1:t-1}$ as just a "context", that is always on the conditioning side, a required "input" to all densities involved, with Bayes rule being applied to $\mathbf{s}\_{t}$ and $\mathbf{v}\_{t}$.
  We know the current measurements only depends on the state, therefore $p \left( \mathbf{v}\_{t} \mid \mathbf{s}\_{t}, \mathbf{v}\_{1:t-1} \right ) = p \left( \mathbf{v}\_{t} \mid \mathbf{s}\_{t} \right ) = \color{LimeGreen}{g}( \mathbf{v}\_{t} \mid \mathbf{s}\_{t} )$, and only the term on the right side of the numerator is left to compute. This term is a marginal of $ \mathbf{s}_t$, which means we have to integrate out anything else. If we were doing this very naively, each time we would integrate out all previous states, but by caching results a.k.a. Dynamic Programming, we only need to marginalize the previous state:

  $$
    p\left( \mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1} \right ) = \int p\left( \mathbf{s}\_{t}, \mathbf{s}\_{t-1} \mid \mathbf{v}\_{1:t-1} \right ) \mathrm{d}\mathbf{s}\_{t-1}
  $$

  Continuing, we split the joint with the product rule and exploit that the states are independent of previous measurements:

  $$
  \begin{equation}\begin{aligned}
    &= \int p\left( \mathbf{s}\_{t} \mid  \mathbf{s}\_{t-1}, \cancel{\mathbf{v}\_{1:t-1}} \right ) p(\mathbf{s}\_{t-1} \mid \mathbf{v}\_{1:t-1}) \mathrm{d}\mathbf{s}\_{t-1} \\\\\\
    &= \int p\left( \mathbf{s}\_{t} \mid  \mathbf{s}\_{t-1} \right ) p(\mathbf{s}\_{t-1} \mid \mathbf{v}\_{1:t-1}) \mathrm{d}\mathbf{s}\_{t-1} \\\\\\
    &= \int \color{cyan}{f}\left( \mathbf{s}\_{t} \mid  \mathbf{s}\_{t-1} \right ) p(\mathbf{s}\_{t-1} \mid \mathbf{v}\_{1:t-1}) \mathrm{d}\mathbf{s}\_{t-1}
  \end{aligned}\end{equation}\tag{4}\label{eq4}$$


  And we are done, if you notice that the right side term in the integral is the filtering distribution at $t-1$, which we have already computed recursively.
  In the literature names are given to the step that requires computing $ p\left( \mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1} \right )$ called *prediction*, because it's our belief on $ \mathbf{s}\_{t}$ before observing the currrent measurement, and *correction* is the name given to the step $ p\left(\mathbf{s}\_{t} | \mathbf{v}\_{1:t}\right) \propto \color{LimeGreen}{g}\left( \mathbf{v}\_{t} \mid \mathbf{s}\_{t} \right ) \cdot p\left( \mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1} \right )$, because we "correct" our prediction by taking into account the measurement.

  In a LDS all computations have closed form solutions, and this algorithm instantiates into the *Kalman Filter*. For discrete  valued random variables, if the dimensionalities are small we can also do exact computations and the label this time is *Forward-Backward* algorithm for HMMs.
  When variables are non-Gaussian and/or transition/observation densities are nonlinear function of their inputs, we have to perform approximate inference.
  By far the most popular method is to use Monte Carlo approximations, and more specifically importance sampling. When we use importance sampling to approximate the filtering distribution, this is called *particle filtering*.

### Recursive formulations <a name="recursive"></a>

There is another way to derive generic computation steps to obtain the filtering distribution. It is common in the particle filtering literature to consider the sequential estimation of a different distribution to the filtering, namely:

$$\begin{equation}\begin{aligned}
p(\mathbf{s}\_{1:t} \mid \mathbf{v}\_{1:t}) \propto p(\mathbf{s}\_{1:t}, \mathbf{v}\_{1:t})
\end{aligned}\end{equation}\tag{5}\label{eq5}$$

Let's take its unnormalized version for simplicity. Applying Bayes' rule gives the following recursive relationship:

$$\begin{equation}\begin{aligned}
p(\mathbf{s}\_{1:t}, \mathbf{v}\_{1:t}) = p(\mathbf{s}\_{1:t-1}, \mathbf{v}\_{1:t-1}) \color{cyan}{f}(\mathbf{s}\_{t} \mid \mathbf{s}\_{t-1}) \color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t})
\end{aligned}\end{equation}\tag{6}\label{eq6}$$

If you can't see why this holds, consider this simple example/subcase:

$$\begin{equation}\begin{aligned}
p(\mathbf{s}\_{1}, \mathbf{s}\_{2}, \mathbf{v}\_{1}, \mathbf{v}\_{2}) =   p(\mathbf{s}\_{1}, \mathbf{v}\_{1}) p(\mathbf{s}_2, \mathbf{v}_2 \mid \mathbf{s}_1, \mathbf{v}_1) = p(\mathbf{s}\_{1}, \mathbf{v}\_{1}) \color{cyan}{f}(\mathbf{s}\_{2} \mid \mathbf{s}\_{1}) \color{LimeGreen}{g}(\mathbf{v}\_{2} \mid \mathbf{s}\_{2})
\end{aligned}\end{equation}$$

Hopefully this convinces you that \eqref{eq6} is true. Then, let's return to the task of sequentially estimating $p(\mathbf{s}\_{1:t} \mid \mathbf{v}\_{1:t})$:


$$\begin{equation}\begin{aligned}
p(\mathbf{s}\_{1:t} \mid \mathbf{v}\_{1:t}) &= \frac{p(\mathbf{s}\_{1:t}, \mathbf{v}\_{1:t})}{p(\mathbf{v}\_{1:t})} \\\\\\
&= \frac{p(\mathbf{s}\_{1:t-1}, \mathbf{v}\_{1:t-1}) \color{cyan}{f}(\mathbf{s}\_{t} \mid \mathbf{s}\_{t-1}) \color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t})}{p(\mathbf{v}\_{1:t})} \\\\\\
&= \frac{p(\mathbf{s}\_{1:t-1}, \mathbf{v}\_{1:t-1})}{p(\mathbf{v}\_{1:t-1})} \frac{\color{cyan}{f}(\mathbf{s}\_{t} \mid \mathbf{s}\_{t-1}) \color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t})}{p(\mathbf{v}\_{t} \mid \mathbf{v}\_{1:t-1})} \\\\\\
&= p(\mathbf{s}\_{1:t-1} \mid \mathbf{v}\_{1:t-1}) \frac{\color{cyan}{f}(\mathbf{s}\_{t} \mid \mathbf{s}\_{t-1}) \color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t})}{p(\mathbf{v}\_{t} \mid \mathbf{v}\_{1:t-1})}
\end{aligned}\end{equation}\tag{7}\label{eq7}$$

Now that we've gone through all this, we are ready to show how to get the filtering distribution by simple marginalization of the expression we just found:

$$\begin{equation}\begin{aligned}
p(\mathbf{s}\_{t} \mid \mathbf{v}\_{1:t}) &= \int p(\mathbf{s}\_{1:t} \mid \mathbf{v}\_{1:t}) \mathrm{d} \mathbf{s}\_{1:t-1} \\\\\\
&= \int p(\mathbf{s}\_{1:t-1} \mid \mathbf{v}\_{1:t-1}) \frac{\color{cyan}{f}(\mathbf{s}\_{t} \mid \mathbf{s}\_{t-1}) \color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t})}{p(\mathbf{v}\_{t} \mid \mathbf{v}\_{1:t-1})} \mathrm{d} \mathbf{s}\_{1:t-1}\\\\\\
&= \frac{\color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t})}{p(\mathbf{v}\_{t} \mid \mathbf{v}\_{1:t-1})} \int p(\mathbf{s}\_{1:\color{red}{t-1}} \mid \mathbf{v}\_{1:t-1}) \color{cyan}{f}(\mathbf{s}\_{t} \mid \mathbf{s}\_{t-1}) \mathrm{d} \mathbf{s}\_{1:t-1} \\\\\\
&= \frac{\color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t})}{p(\mathbf{v}\_{t} \mid \mathbf{v}\_{1:t-1})} \overbrace{\int p(\mathbf{s}\_{1:\color{red}{t}} \mid \mathbf{v}\_{1:t-1}) \mathrm{d} \mathbf{s}\_{1:t-1}}^{= p(\mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1}) ~ \text{by marginalization}} \\\\\\
&= \frac{p(\mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1}) \color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t})}{p(\mathbf{v}\_{t} \mid \mathbf{v}\_{1:t-1})}
\end{aligned}\end{equation}\tag{8}\label{eq8}$$

Which is the indeed same result that we got through the prediction and correction steps.

The two perspectives, namely the prediction-correction equations or the recursive formulations, can be both used to derive concrete algorithms in slightly different ways. Let's highlight the two most important equations for particle filtering:

<div id="example1">

$$\begin{equation}\begin{aligned}
 p(\mathbf{s}\_{1:t} \mid \mathbf{v}\_{1:t}) =  p(\mathbf{s}\_{1:t-1} \mid \mathbf{v}\_{1:t-1}) \frac{\color{cyan}{f}(\mathbf{s}\_{t} \mid \mathbf{s}\_{t-1}) \color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t})}{p(\mathbf{v}\_{t} \mid \mathbf{v}\_{1:t-1})}
\end{aligned}\end{equation}\tag{9}\label{eq9}$$
</div>
<br>

We can call this "Trajectory Filtering Distribution" (TFD), since it considers the sequential estimation of the whole trajectory of states. Similarly,

<div id="example1">
$$\begin{equation}\begin{aligned}
p(\mathbf{s}\_{t} \mid \mathbf{v}\_{1:t}) = \frac{p(\mathbf{s}\_{t} \mid \mathbf{v}\_{1:t-1}) \color{LimeGreen}{g}(\mathbf{v}\_{t} \mid \mathbf{s}\_{t})}{p(\mathbf{v}\_{t} \mid \mathbf{v}\_{1:t-1})}
\end{aligned}\end{equation}\tag{10}\label{eq10}$$
</div>
<br>

this can be called "State Filtering Distribution" (SFD).

## Particle filtering <a name="pf"></a>

### Basics of Monte Carlo and Importance Sampling <a name="basics"></a>
Most things aren't linear and/or Gaussian, so we need approximate inference. Specifically in the filtering/sequential Bayes literature, importance sampling based methods are more popular than deterministic approximations such as Laplace's method, Variational Bayes and Expectation Propagation.
Recall that the Monte Carlo method is a general tool to approximate integrals, expectations, probabilities with random samples:

$$
\mathbb{E}\_{p(\mathbf{x})}[f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x} \approx \frac{1}{N} \sum\_{n=1}^{N} f(\mathbf{x}\_{n}) \qquad \mathbf{x}_n \sim p(\mathbf{x})
\tag{11}\label{eq11}$$

Where $f(\mathbf{x})$ is some generic function of $\mathbf{x}$. Monte Carlo approximations of this kind are very appealing since unbiased and consistent, and it is easy to show that the variance of the estimate is $ \mathcal{O}(n^{-1})$ *regardless* of the dimensionality of the vector $\mathbf{x}$. Another simple idea that we will use extensively in particle filtering is that these samples can not only be used to approximate integrals with respect to the target distribution $p(\mathbf{x})$, but also to approximate the target itself:

$$
p(\mathbf{x}) \approx \frac{1}{N}\sum\_{n=1}^{N} \delta\_{\mathbf{x}}(\mathbf{x}\_{n})
$$

Where $  \delta\_{\mathbf{x}}(\mathbf{x}\_{n})$ is the Dirac delta mass evaluated at point $\mathbf{x}\_{n}$. This is a function that is $\infty$ at its evaluation point and $0$ everywhere else, and satisfies $ \int \delta\_{\mathbf{x}} \mathrm{d}\mathbf{x} = 1$, so that approximating the distribution itself recovers the previous result for expectations:

$$
\mathbb{E}\_{p(\mathbf{x})}[f(\mathbf{x})]  \approx \int  f(\mathbf{x})  \frac{1}{N}\sum\_{n=1}^{N} \delta\_{\mathbf{x}}(\mathbf{x}\_{n}) \mathrm{d}\mathbf{x} =   \frac{1}{N} \sum\_{n=1}^{N} \int \delta\_{\mathbf{x}}(\mathbf{x}\_{n}) f(\mathbf{x}) \mathrm{d}\mathbf{x} = \frac{1}{N} \sum\_{n=1}^{N} f(\mathbf{x}_n)
$$

This should be interpreted as an approximation to the underlying distribution and not of the density function.
Often it is not possible to sample from the distribution of interest. Therefore we can use importance sampling, which is a technique based on the simple observation that we can sample from another, known distribution and assign a weight to the samples to represent their "importance" under the real target:

$$\begin{equation}\begin{aligned}
\mathbb{E}\_{p(\mathbf{x})}[f(\mathbf{x})] &= \int f(\mathbf{x}) \cdot p(\mathbf{x}) \mathrm{d}\mathbf{x} \\\\\\
&= \int \frac{f(\mathbf{x}) \cdot p(\mathbf{x})}{q(\mathbf{x})} \cdot q(\mathbf{x}) \mathrm{d} \mathbf{x} \\\\\\
&= \mathbb{E}\_{q(\mathbf{x})} \left [ f(\mathbf{x}) \cdot \frac{p(\mathbf{x})}{q(\mathbf{x})} \right ] \\\\\\
&= \mathbb{E}\_{q(\mathbf{x})} \left [ f(\mathbf{x}) \cdot w(\mathbf{x}) \right ]
\end{aligned}\end{equation}\tag{12}\label{eq12}$$

Under certain conditions, namely that $ f(\mathbf{x}) \cdot p(\mathbf{x}) > 0 \Rightarrow q(\mathbf{x}) > 0$, we have rewritten the expectation under a distribution of choice $q(\mathbf{x})$ called $\color{#FF8000}{\text{proposal}}$ which we know how to sample from. Note that it is not possible to have $ q(\mathbf{x}) = 0$, as we will never sample any $\mathbf{x}\_{i}$ from $q$ such that this holds. The weight $w(\mathbf{x})$ can be intepreted as “adjusting” the estimate of the integral by taking into account that the samples were generated from the “wrong” distribution. Notice that Importance Sampling can be used to approximate also generic integrals which are not necessarily expectations.

Let's return in the context of Bayesian inference, where we have a target posterior distribution $ \pi(\mathbf{x}) = p(\mathbf{x} \mid \mathcal{D}) $ where $ \mathcal{D}$ is any observed data.
In our state space model $\mathcal{D} = \mathbf{v}\_{1:t}$, and $\pi(\mathbf{x}) = p(\mathbf{s}\_{1:t} \mid \mathbf{v}\_{1:t})$. Consider an integral of some function of $ \mathbf{x}$ under the posterior, often the ultimate object of interest:

$$\begin{equation}\begin{aligned}
\mathcal{I} = \mathbb{E}\_{\pi(\mathbf{x})}[f(\mathbf{x})] = \int f(\mathbf{x}) \pi(\mathbf{x})
\end{aligned}\end{equation}\tag{13}\label{eq13}$$

We can estimate this integral in two main ways with IS: the former which assumes that we know the normalizing constant of the posterior $ \pi(\mathbf{x})$, and the latter estimates the normalizing constant too by IS, with the same set of samples. Since in Bayesian inference we usually can only evaluate $\pi(\mathbf{x})$ up to a normalizing constant, let's examine the latter option, called *self-normalized* IS estimator:

$$\begin{equation}\begin{aligned}
\mathbb{E}\_{\pi} \left [ f(\mathbf{x}) \right ] &= \int f(\mathbf{x}) \pi(\mathbf{x}) \mathrm{d} \mathbf{x} \\\\\\
&= \int  f(\mathbf{x})\frac{\pi(\mathbf{x})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\\\\\
&= \int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{p(\mathcal{D})q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\\\\\
&=  \frac{1}{p(\mathcal{D})}\int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\\\\\
&=  \frac{1}{\int p(\mathbf{x}, \mathcal{D}) \mathrm{d} \mathbf{x}}\int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\\\\\
&=  \frac{1}{\int \frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})}  q(\mathbf{x}) \mathrm{d} \mathbf{x}}\int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\\\\\
&= \frac{1}{\mathbb{E}\_{q}\left [ \frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} \right ]}
\cdot \mathbb{E}\_{q}\left [ f(\mathbf{x}) \frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} \right ] \\\\\\
&\approx \frac{1}{\cancel{\frac{1}{N}}\sum\_{n=1}^{N} \frac{p(\mathbf{x}_n , \mathcal{D})}{q(\mathbf{x}_n)}}
\cdot ~ \cancel{\frac{1}{N}} \sum\_{n=1}^{N} f(\mathbf{x}_n) \frac{p(\mathbf{x}_n, \mathcal{D})}{q(\mathbf{x}_n)} := \widehat{\mathcal{I}}\_{SN}
\end{aligned}\end{equation}\tag{14}\label{eq14}$$

Where the ratio $ \frac{p(\mathbf{x}_n, \mathcal{D})}{q(\mathbf{x}_n)} := \tilde{w}(\mathbf{x}_n)$ plays the role of the (unnormalized) importance weight. The estimator $\widehat{\mathcal{I}}\_{SN}$ can be shown to be biased. An important observation that is useful in particle filtering is that the normalizing constant estimate $ Z \approx  \widehat{Z} = \frac{1}{N} \sum\_{n=1}^{N} \frac{p(\mathbf{x}_n , \mathcal{D})}{q(\mathbf{x}_n)} = \frac{1}{N} \sum\_{n=1}^{N} \tilde{w}(\mathbf{x}_n) $ is unbiased. Even more importantly, the approximate posterior is :

$$
\pi(\mathbf{x}) \approx \sum\_{n=1}^{N} w(\mathbf{x}_n)\delta\_{\mathbf{x}_n}(\mathbf{x}) \qquad w(\mathbf{x}_n) = \frac{\tilde{w}(\mathbf{x}_n)}{\sum\_{k=1}^{N} \tilde{w}(\mathbf{x}_k)}
$$

Using normalized weights. If the normalizing constant was known exactly, then we could build a *non-normalized* IS estimator which is actually unbiased (with an almost equivalent derivation, omitted):  

$$
\widehat{\mathcal{I}}\_{NN} := \frac{1}{N} \cdot \frac{1}{Z} \sum\_{n=1}^{N}  f(\mathbf{x}_n) \frac{p(\mathbf{x}_n, \mathcal{D})}{q(\mathbf{x}_n)} = \frac{1}{N} \sum\_{n=1}^{N}  f(\mathbf{x}_n) \frac{\pi(\mathbf{x}_n)}{q(\mathbf{x}_n)}
$$

Where $Z$ is the normalizing constant of the posterior distribution $\pi(\mathbf{x})$. In this post we are only concerned with self-normalized estimators which, while biased, turn out to have lower variance in several settings.
From now on, an importance weight as function of sample $ \mathbf{x}_n $ is abbreviated as $w_n$; it is important however to keep in mind that weights are a function of inputs/samples.

### Choice of proposal and variance of importance weights <a name="isproposal"></a>

It is pretty intuitive that our IS estimates can only be as good as our proposal. In general, we should seek a proposal that minimizes the variance of our estimators. This follows from the fact that the variance of a MC estimate (which is a sample average) is the expected square error from the true value of the integral. Let us see this by considering, for simplicity, the variance of the non-normalized estimator:

$$
\mathbb{V}\_{q} [ \widehat{\mathcal{I}}\_{NN} ] =  \mathbb{E}\_{q} \left [ \left ( \widehat{\mathcal{I}}\_{NN} - \mathbb{E}\_{q} \left [  \frac{f(\mathbf{x}) \pi(\mathbf{x})}{q(\mathbf{x})} \right ] \right )^{2} \right ] = \mathbb{E}\_{q} \left [ \left ( \widehat{\mathcal{I}}\_{NN} - \mathcal{I} \right )^2  \right ]
$$

Which follows by simply applying the definition of variance (recalling that our samples are obtained through the proposal $q$ so that all expectations are under $q$). In order to derive the proposal that minimizes the variance, it is easier to inspect a different expression for the variance $\mathbb{V}\_{q} [ \widehat{\mathcal{I}}\_{NN} ] $, which uses the identity that variance equals second moment minus first moment squared, instead of the definition:

$$
\mathbb{V}\_{q} [ \widehat{\mathcal{I}}\_{NN} ] = \frac{1}{N} \mathbb{V}\_{q} \left [ \frac{f(\mathbf{x})\pi(\mathbf{x})}{q(\mathbf{x})} \right ] = \frac{1}{N} \mathbb{E}_q \left [ \left ( \frac{f(\mathbf{x})\pi(\mathbf{x})}{q(\mathbf{x})} \right )^2 \right ] - \frac{1}{N}  \underbrace{\left (  \mathbb{E}_q \left [ \frac{f(\mathbf{x})\pi(\mathbf{x})}{q(\mathbf{x})} \right ] \right )^2}\_{=(\mathcal{I})^2}
$$

Notice that the term on the right in this expression is just $\mathcal{I}^2$ and thus does not involve $q$. We only need to minimize the first term with respect to $q$. Expanding this term on the left builds some intuition on what the form of the minimizing proposal looks like:

$$\begin{equation}\begin{aligned}
\mathbb{E}_q \left [ \left ( \frac{f(\mathbf{x})\pi(\mathbf{x})}{q(\mathbf{x})} \right )^2 \right ]  &=  \int \left ( \frac{f(\mathbf{x})\pi(\mathbf{x})}{q(\mathbf{x})} \right )^2 q(\mathbf{x}) \mathrm{d} \mathbf{x} \\\\\\
&=  \int \frac{(f(\mathbf{x})\pi(\mathbf{x}))^2}{q(\mathbf{x})} \mathrm{d}\mathbf{x} \\\\\\
&= \int  \left | f(\mathbf{x})\pi(\mathbf{x})  \right | \frac{\left | f(\mathbf{x})\pi(\mathbf{x})  \right |}{q(\mathbf{x})} \mathrm{d} \mathbf{x}
\end{aligned}\end{equation}\tag{15}\label{eq15}$$

Clearly, if we want this quantity to be small, then whenever the numerator is high $q$ should be at least as high. That is, when $  \vert f(\mathbf{x}) \pi(\mathbf{x})  \vert $ is high, then $q(\mathbf{x})$ should be high, or at least it should definitely not be small. That is, we need  $  \vert f(\mathbf{x}) \pi(\mathbf{x})  \vert $  large   $\Rightarrow q(\mathbf{x})$ large. In fact, the proposal that minimizes the variance turns out to be a normalized version of $  \vert f(\mathbf{x}) \pi(\mathbf{x})  \vert $  :

$$
q^{*}(\mathbf{x}) = \frac{\left | f(\mathbf{x})\pi(\mathbf{x})  \right | }{\int \left | f(\mathbf{x})\pi(\mathbf{x})  \right |  \mathrm{d}\mathbf{x}}
$$

Unfortunately deriving this from scratch, by explicitly minimizing the second moment turns out to require some (basic) functional analysis. However, given knowledge of the optimal solution, we can verify that it is indeed minimizing the variance by making use of the following inequality:

$$
\mathbb{E}[x^2] \geq \mathbb{E}[ \left | x \right |]^{2}
$$

wich follows from the Cauchy-Schwartz inequality. We show that this bound is tight when using the optimal proposal. Plugging in the optimal proposal gives:

$$\begin{equation}\begin{aligned}
 \mathbb{E}\_{q^{*}} \left [ \left ( \frac{f(\mathbf{x})\pi(\mathbf{x})}{q^{*}(\mathbf{x})} \right )^2 \right ] &=  \int   \left | f(\mathbf{x})\pi(\mathbf{x})  \right | \frac{\left | f(\mathbf{x})\pi(\mathbf{x})  \right |}{q^{*}(\mathbf{x})} \mathrm{d} \mathbf{x} \\\\\\
 &= \int  \left | f(\mathbf{x})\pi(\mathbf{x})  \right |  \mathrm{d}\mathbf{x} ~ \cdot ~ \int  \left | f(\mathbf{x})\pi(\mathbf{x})  \right |^2 \frac{1}{ \left | f(\mathbf{x})\pi(\mathbf{x})  \right | } \mathrm{d} \mathbf{x}
\end{aligned}\end{equation}\tag{16}\label{eq16}$$
