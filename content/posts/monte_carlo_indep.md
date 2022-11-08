---
title: "Simple Monte Carlo is independent of dimension. Or is it ?"
date: 2022-05-26T22:10:06Z
type: page
draft: true
---

It is essentially the status quo to claim that the error (Mean Squared Eror = MSE) of Monte Carlo integration is "independent of the dimension" of the variable being integrated, or some equivalent variant of this statement. We will provide evidence and citations later. In this post, we provide a simple reasoning for why this can be misleading, especially for the novice that approaches the Monte Carlo literature with the aim to learn choosing among estimators in practice. 
Defining the notation needed for the post, we have the integral (expectation) to be estimated as 

$$\begin{equation}\begin{aligned}
\mu = \mathbb{E}\_{p}[f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x} , 
\end{aligned}\end{equation}\tag{1}\label{eq1}$$

where $p(\mathbf{x})$ is a density, $\mathbf{x} \in \mathbb{R}^{D}$, and the corresponding Monte Carlo estimator as 

$$\begin{equation}\begin{aligned}
\widehat{\mu}\_{\text{MC}} = \frac{1}{N} \sum\_{n=1}^{N} f(\mathbf{x}^{(n)}) , ~~ \mathbf{x}^{(n)} \sim p(\mathbf{x}) ,
\end{aligned}\end{equation}\tag{2}\label{eq2}$$

with samples being i.i.d. 
At this point, many (if not most) authoritative sources state some variant of the following: because the variance (hence the MSE) of $\widehat{\mu}\_{\text{MC}}$ is given simply by 
$$\begin{equation}\begin{aligned}
\mathbb{E}\_p[(\widehat{\mu}\_{\text{MC}} - \mu)^2] = \mathbb{V}\_p[\widehat{\mu}\_{\text{MC}}] = \frac{\mathbb{V}\_p[f(\mathbf{x})]}{N} , 
\end{aligned}\end{equation}\tag{3}\label{eq3}$$
which is readily seen to be $\mathcal{O}(1/N)$. For example, Owen ([1]) (my personal favourite source on Monte Carlo, by a large margin) states (Chapter 2, page 17): "A striking feature about the formula [equivalent of our Eq. 3] is that the dimension $D$ does not appear in it anywhere. " .
Another example, Doucet and Johansen in their (great) tutorial on particle filtering [2], mention: "The main advantage of Monte Carlo methods over standard approximation techniques is that the variance of the approximation error decreases at a rate of  $\mathcal{O}(1/N)$ regardless of the dimension of the space".  And more, this well-cited tutorial from the respected "Acta Numerica" [3]: "Its convergence rate, $\mathcal{O}(1/\sqrt{N})$, [here referring to RMSE] independent of dimension". Iain Murray, in his PhD thesis on MCMC, seems to take a more careful stance (although not elaborating on it): " Monte Carlo is usually simple and its $\mathcal{O}(1/\sqrt{N})$ scaling of error bars “independent of dimensionality” may be good enough "; the quotes added here indeed seem to suggest us that there is more to the story.  



## References
1. Art B. Owen. Monte Carlo theory, methods and examples. 2013
2. Doucet, A. and Johansen, A.M., 2009. A tutorial on particle filtering and smoothing: Fifteen years later. Handbook of nonlinear filtering, 12(656-704), p.3.
3. Caflisch, R.E., 1998. Monte carlo and quasi-monte carlo methods. Acta numerica, 7, pp.1-49.
1. H. S. Battey, D. R. Cox. "Some Perspectives on Inference in High Dimensions." Statistical Science, 37(1) 110-122 February 2022.
2. Wainwright, M.J., 2019. High-dimensional statistics: A non-asymptotic viewpoint (Vol. 48). Cambridge University Press.