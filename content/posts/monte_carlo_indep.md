---
title: "Monte Carlo is independent of dimension. Or is it ?"
date: 2022-05-26T22:10:06Z
type: page
draft: true
---

It is essentially the status quo to claim that the error (Mean Squared Eror = MSE) of Monte Carlo integration is "independent of the dimension" of the variable being integrated, or some equivalent variant of this statement . In this post, we provide a simple reasoning for why this can be misleading, especially for the novice that approaches the Monte Carlo literature with the aim to learn choosing among estimators in practice. 
Defining the notation needed for the post, we have the integral (expectation) to be estimated as 
$$\begin{equation}\begin{aligned}
\mu = \mathbb{E}_{p}[f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x} , 
\end{aligned}\end{equation}\tag{1}\label{eq1}$$
where $p(\mathbf{x})$ is a density, and the corresponding Monte Carlo estimator as 
$$\begin{equation}\begin{aligned}
\widehat{\mu}_{\text{MC}} = \frac{1}{N} \sum_{n=1}^{N} f(\mathbf{x}^{(n)}) , \qquad \mathbf{x}^{(n)} \sim p(\mathbf{x}) ,
\end{aligned}\end{equation}\tag{2}\label{eq2}$$
with samples being i.i.d. 
At this point, many (if not most) authoritative sources state some variant of the following: because the variance (hence the MSE) of $\widehat{\mu}_{\text{MC}}$ is given simply by 
$$\begin{equation}\begin{aligned}
\mathbb{E}_p[(\widehat{\mu}_{\text{MC}} - \mu)^2] = \mathbb{V}_q
\end{aligned}\end{equation}\tag{3}\label{eq3}$$
