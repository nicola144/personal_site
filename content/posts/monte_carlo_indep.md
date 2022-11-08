---
title: "Simple Monte Carlo is independent of dimension. Or is it ?"
date: 2022-05-26T22:10:06Z
type: page
draft: true
---

It is essentially the status quo to claim that the error (Mean Squared Eror = MSE) of Monte Carlo integration is "independent of the dimension" of the variable being integrated, or some equivalent variant of this statement. I will provide evidence and citations later. In this post, I provide a simple reasoning for why this can be misleading, especially for the novice that approaches the Monte Carlo literature with the aim to learn how to chooose among estimators in practice. 
Defining the notation needed for the post, let the integral (expectation) to be estimated as 

$$\begin{equation}\begin{aligned}
\mu = \mathbb{E}\_{p}[f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x} , 
\end{aligned}\end{equation}\tag{1}\label{eq1}$$

where $p(\mathbf{x})$ is a density, $\mathbf{x} \in \mathbb{R}^{D}$, the test function $f(\mathbf{x})$ is normally considered $f: \mathbb{R}^D \rightarrow \mathbb{R}$ and the corresponding Monte Carlo estimator as 

$$\begin{equation}\begin{aligned}
\widehat{\mu}\_{\text{MC}} = \frac{1}{N} \sum\_{n=1}^{N} f(\mathbf{x}^{(n)}) , ~~ \mathbf{x}^{(n)} \sim p(\mathbf{x}) ,
\end{aligned}\end{equation}\tag{2}\label{eq2}$$

with samples being i.i.d. 

At this point, many (if not most) authoritative sources proceed with some variant of the following: because the variance (hence the MSE) of $\widehat{\mu}\_{\text{MC}}$ is given simply by

$$\begin{equation}\begin{aligned}
\mathbb{E}\_p[(\widehat{\mu}\_{\text{MC}} - \mu)^2] = \mathbb{V}\_p[\widehat{\mu}\_{\text{MC}}] = \frac{\mathbb{V}\_p[f(\mathbf{x})]}{N} , 
\end{aligned}\end{equation}\tag{3}\label{eq3}$$

which is readily seen to be $\mathcal{O}(1/N)$, then the error is independent of $D$. For example, Art B. Owen in his book [1] (my personal favourite source on Monte Carlo) states (Chapter 2, page 17): "*A striking feature about the formula [equivalent of our Eq. 3] is that the dimension* $D$ *does not appear in it anywhere.* " .
Another example, Doucet and Johansen in their (great) tutorial on particle filtering [2], mention: "*The main advantage of Monte Carlo methods over standard approximation techniques is that the variance of the approximation error decreases at a rate of*  $\mathcal{O}(1/N)$ *regardless of the dimension of the space*".  And more, this well-cited tutorial from the respected "Acta Numerica" [3] begins with "*Its convergence rate,* $\mathcal{O}(1/\sqrt{N})$, *[here referring to RMSE] independent of dimension*". 

In fact, even proponents of "competitor" methods to Monte Carlo, seem to list this "independence of dimensionality" property as a potential advantage, see e.g. from the Probabilistic Numerics textbook (page 110): "*A defender of Monte Carlo might argue that its most truly desirable characteristic is the fact that its convergence (see Lemma 9.2) does not depend on the dimension of the problem. Performing well even in high dimension is a laudable goal.*". They later propose alternative reasons to what I will explain for why this can be misleading; however, they also mention " *rather than being equally good for any number of dimensions, Monte Carlo is perhaps better thought of as being equally bad.*"; I am not sure I agree with this: as I will show shortly, even for simple integrands it is clear that a higher dimension is **worse**, not equally bad. Iain Murray, in his PhD thesis on MCMC [4], seems to take a more careful stance (although not elaborating on it): " *Monte Carlo is usually simple and its* $\mathcal{O}(1/\sqrt{N})$ *scaling of error bars “independent of dimensionality” may be good enough* "; the quotes added here indeed seem to suggest us that there is more to the story.  

And indeed, there is more to the story. Is $\mathbb{V}\_p[f(\mathbf{x})]$ really independent of the dimension of $\mathbf{x}$, namely $D$ ? Maybe for some choices of $f$, but clearly also not for many others. What is a simple, perhaps the simplest, way to see that the statements previously mentioned can be very misleading?  
Suppose that the function $f$ is non-negative. This is not very restrictive: it includes e.g. normalizing constant estimation, and much of the particle physics literature uses an adaptive Monte Carlo method, "VEGAS", which requires this assumption. More general functions will be the subject of a future blogpost. Then, the MSE in \eqref{eq3} can be written as 

$$\begin{equation}\begin{aligned}
\mathbb{E}\_p[(\widehat{\mu}\_{\text{MC}} - \mu)^2] = \frac{\mu^2}{N} ~ \chi^{2}(p \cdot f \mid \mid p) , 
\end{aligned}\end{equation}\tag{4}\label{eq4}$$

where we $p \cdot f$ denotes the **normalized version** of the product $ f(\mathbf{x}) p(\mathbf{x}) $, i.e. 


$$\begin{equation}\begin{aligned}
p \cdot f = \frac{f(\mathbf{x}) p(\mathbf{x})}{\int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x}} =  \frac{f(\mathbf{x}) p(\mathbf{x})}{\mu}
\end{aligned}\end{equation}\tag{5}\label{eq5}$$

(which is a density) and I have used the chi-squared divergence between densities $p$ and $q$:

$$\begin{equation}\begin{aligned}
\chi^{2}(p \mid \mid q) = \int \frac{p(\mathbf{x})^2}{q(\mathbf{x})} \mathrm{d}\mathbf{x} - 1. 
\end{aligned}\end{equation}\tag{6}\label{eq6}$$

What has changed ? Nothing, essentially, but perhaps people familiar with divergences already start having some intuition, since they usually do not behave well with dimension and, clearly, in general $p \cdot f$ and $p$ are different densities. And, I personally like this chi-squared view, among other reasons, because it makes apparent that simple Monte Carlo is a special case of importance sampling. More on this in following posts, likely. 

Now, we take a concrete (and very simple!) example to show how the divergence in \eqref{eq5} can easily scale badly with $D. Let $f(\mathbf{x})$ be (the square root of) a Gaussian density $f(\mathbf{x}) = \sqrt{\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}\_{1},  \boldsymbol{\Sigma}\_{1})}$, and let $p(\mathbf{x})$ be an actual Gaussian, $p(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}\_{2},  \boldsymbol{\Sigma}\_{2})$. 
We can then expand the MSE of the Monte Carlo estimator as 

$$\begin{equation}\begin{aligned}
\frac{\mu^2}{N} ~ \chi^{2}(p \cdot f \mid \mid p) &= \frac{\mu^2}{N}  \left ( \int f(\mathbf{x})^2 p(\mathbf{x}) \mathrm{d}\mathbf{x} - 1 \right )  \\\\\\
&= \frac{\mu^2}{N}  \left ( \int \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}\_{1},  \boldsymbol{\Sigma}\_{1}) \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}\_{2},  \boldsymbol{\Sigma}\_{2})  \mathrm{d}\mathbf{x} - 1 \right ) \\\\\\
&=  \frac{\mu^2}{N}  \left ( \mathcal{O}((2 \pi)^{D/2}) - 1 \right )
\end{aligned}\end{equation}\tag{6}\label{eq6}$$



## References
1. Art B. Owen. Monte Carlo theory, methods and examples. 2013
2. Doucet, A. and Johansen, A.M., 2009. A tutorial on particle filtering and smoothing: Fifteen years later. Handbook of nonlinear filtering, 12(656-704), p.3.
3. Caflisch, R.E., 1998. Monte carlo and quasi-monte carlo methods. Acta numerica, 7, pp.1-49.
4. Philipp Hennig, Michael A. Osborne, Hans Kersting. Probabilistic Numerics: Computation as Machine Learning. Cambridge University Press, 2022.
5. Murray, I., 2007. Advances in Markov chain Monte Carlo methods. University of London, University College London (United Kingdom).
6. H. S. Battey, D. R. Cox. "Some Perspectives on Inference in High Dimensions." Statistical Science, 37(1) 110-122 February 2022.
7. Wainwright, M.J., 2019. High-dimensional statistics: A non-asymptotic viewpoint (Vol. 48). Cambridge University Press.