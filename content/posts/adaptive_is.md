---
title: "What is adaptive importance sampling?"
date: 2023-02-23T15:34:02Z 
tags: ["Tutorial"]
---

In this post, I will introduce the concept of **adaptive importance sampling (AIS)**.

I assume a machine-learning type audience. You will be at least vaguely familiar with Bayesian inference and probabilistic methods.

So AIS is a meta-algorithm / class of methods, similarly to Markov Chain Monte Carlo. AIS is motivated by the problem of approximating integrals / expectations. 


<div id="disqus_thread"></div>
<script>
    /**
    *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
    *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables    */

    var disqus_config = function () {
    this.page.url = "https://www.branchini.fun/posts/sampling_and_estimating";  
    this.page.identifier = "samplingandestimating"; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
    };

    (function() { // DON'T EDIT BELOW THIS LINE
    var d = document, s = d.createElement('script');
    s.src = 'https://personal-website-g7y0elzvjn.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>


<p>Cited as:</p>
<pre tabindex="0"><code>@article{branchini2023why,
  title   = "Why do sampling and estimating the normalizing constant avoid each other?",
  author  = Branchini, Nicola,
  journal = https://www.branchini.fun,
  year    = 2023,
}
