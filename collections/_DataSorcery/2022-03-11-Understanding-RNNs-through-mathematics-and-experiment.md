---
layout: single
title: "Understanding RNNs through mathematical analysis and experiment"
collection: DataSorcery
---

Recently I've become interested in RNN's (recurrent neural networks). These are neural networks which utilize a sort of memory state. Neural networks learn functions of the form 

$$nn:\mathbb{R}^{d_{input}} \rightarrow \mathbb{R}^{d_{output}}.$$

However recurrent neural networks learn a function of the form 

$$rnn:\mathbb{R}^{d_{input} \times d_{output}} \rightarrow \mathbb{R}^{d_{output}}$$ 

such that it can learn sequences of the form $$h_n = rnn(h_{n-1},x_n).$$ Here $$h_n$$ can be further processed to an output $$y_n$$, for example with another neural network as $$y_n = nn(h_n)$$. 

These types of neural networks are surprisingly strong and capable. Andrej Karpathy wrote a beautiful [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) on the ridiculous effectiveness of the RNN in character based text generation.

It is thus an interesting question whether we can deduce which type of problem the RNN is good at solving. To analyze this wee will use both analytical and experimental tools. 

# Formulation

First, what is the unique strength of the RNN? RNN can be appied to sequences of arbitrary length, thus they estimate values of the form 

$$ Y_T | X_T,\ldots,X_1 $$

given that we know prior information $$X_T,\ldots,X_1$$ predict what $$Y_T$$ is going to be. Here $$Y_T$$ is typically dependent on $$X_T,\ldots,X_1$$ is some shape or fashion. So the task that an RNN needs to complete is extract some information from the $$X_T$$ and process this information to a suitable predictor.

This is the general setup for the task of the RNN, but it is actually too broad to describe what RNN do. RNN process information step-wise, applying each time the same function. Thus we should perhaps think more in steps about them. 

$$\begin{align}
H_0 =&\ initialize() \\
H_1 =&\ f(H_0;X_1) \\
H_2 =&\ f(H_1;X_2) \\
H_3 =&\ f(H_2;X_3) \\
\vdots&
\end{align}
$$

This differs slightly from the previous idea, because it becomes clear now that information is being processed several times. Any information extracted from $$X_1$$ needs to be first processed to $$H_1$$ and later again processed to $$H_2$$ before it can be processed again to $$H_3$$. Moreover, in the mean time it also must serve as predictors for $$Y_1=g(H_1)$$ and $$Y_2=g(H_2)$$ before it can be used as predictor for $$Y_3 = g(H_3)$$. Thus the information from $$X_1$$ serves as predictor for three values, or more, all while being processed several times by the same function, and new information is also being injected. The information from $$X_1$$ needs to survive for two applications of $$f$$ whereas the information from $$X_2$$ only needs to survive one application from $$f$$. Intuitively, this suggests that the information decline for $$X_1$$ is exponential as it must make room for more recent information being injected.

# Models

There are several models that have been proposed to serve as RNN. I am considering here those that are readily accessible through their pytorch implementations, which are the Elman-type RNN, the LSTM (long short-term memory) and the GRU (gated recurrent unit). All of these implementation follow the same model as above, but the way they go about computing the update is different. 

### Elman RNN

The Elman-type RNN is the simplest of the three. It computes the update by concatenating input $x_n$ with $h_{n-1}$ and applying a single layer neural network, i.e. 

$$ h_n = nn(h_{n-1},x_n) = \sigma ( W_{ih} x_n + W_{hh} h_{n-1} + b_h ).$$

Here $$\sigma$$ can be either the $$tanh$$ or the $$ReLU$$ for the pytorch implementation.

### LSTM

The LSTM is a well known and quite possibly the most popular RNN architecture. It uses two hidden state vector, one of which is called the cell state and will be denoted by $$c_n$$. Given input $$x_n$$ and $$h_{n-1}$$ the LSTM first computes four single-layer neural nets based on $$h_{n-1}$$ and $$x_n$$: 

$$
\begin{align}
i &= \sigma( W_{ii} x_n + W_{ih} h_{n-1} + b_i)\\
f &= \sigma( W_{fi} x_n + W_{fh} h_{n-1} + b_f)\\
g &= tanh( W_{gi} x_n + W_{gh} h_{n-1} + b_g)\\
o &= \sigma( W_{oi} x_n + W_{oh} h_{n-1} + b_o)\\
\end{align}
$$

With these four values it computes the update of the cell state as 

$$ c_n = f \odot c_{n-1} + i \odot g.$$

Here $$\odot$$ denotes the element-wise product of two vectors. Finally, from $$o$$ and $$c_n$$ the update for the hidden state $$h_n$$ is again computed: 

$$ h_n = o \odot tanh(c_n).$$

These update rules can be read as follows. The value $$f$$ computes a forget value which determines to what extend information in $$c_{n-1}$$ is preserved, note that the sigmoid function $$\sigma(x) = e^x/(1+e^x)$$ only takes value between $$0$$ and $$1$$. The values $$i$$ and $$g$$ then compute new information to be added to $$c_{n-1}$$ based on the hidden state and the input. 
Finally, $$o$$ computes an ignore value which determines what output the hidden state will ignore and what it will return from the cell state. Of note is that here $$c_n$$ can become a very large value, in theory it has no upper bound. This might make up for a potential weakness, which is that $$h_n$$ is computed with just the activation function instead of a linear combination and the activation function.

It is worth to spend a little more time on interpretation of the update rules for the $$c_n$$ state. Traditionally, $$f$$ is read as a forget value, determining which part of $$c_{n-1}$$ is forgotten and by how much. It is noteworthy that the sigmoid activation function $$\sigma$$ has the property $$\sigma(-x) = 1-\sigma(x)$$. This means that $$f$$ and $$i$$ could learn to work as a mixture of the form $$f$$ and $$i=1-f$$. In this case $$c_n$$ functions like an online mean of updates from $$g$$. However, $$g$$ is always bounded between $$-1$$ and $$1$$ and thus $$c_n$$ would always be bounded between $$-1$$ and $$1$$ causing $$tanh(c_n)$$ to be limited to a small expression range and poorly capable of utilizing the non-linearity in $$tanh$$. Instead $$c_n$$ must achieve scale to uitilize the non-linear parts of $$tanh$$. This is only possible through accumulated addition of $$i \odot g$$. This means that early in the sequence the LSTM is denied some of the non-linear curvature in the $$tanh$$ and later in the sequence it must struggle against the accumulated addition to get access to linear-like curvature of the $$tanh$$. Of course, with `no free lunch' in the mixture it might just be that this is not at all a problem for the LSTM.

### GRU

The GRU is another popular RNN architecture. It uses only one hidden state just like the Elman RNN and is less complex than the LSTM, though don't let this simplicity fool you; it is quite powerful. The GRU first computes two single-layer neural nets from the inputs $$h_{n-1}$$ and $$x_n$$ as follows:

$$
\begin{align}
r &= \sigma ( W_{ri} x_n + W_{rh} h_{n-1} + b_r )\\
z &= \sigma ( W_{zi} x_n + W_{zh} h_{n-1} + b_z )\\
\end{align}
$$

With $$r$$ it computes another value $$n$$ as follows:

$$n = tanh(W_{ni} x_n + b_i + r \odot (W_{nh} h_{n-1} + b_h) ).$$

And finally, this is processed into the value for $$h_{n}$$ as 

$$ h_n = (1-z) \odot n + z \odot h_{n-1}.$$

These update rules are particularly interesting. First, because they are very interpretable. And second, because my interpretations might be wrong as we will see later. I'll start with the last update step first. We see here that the new $$h_n$$ is a mixture of the value $$n$$ and the previous value for $$h_{n-1}$$. The mixture is a convex mixture and we have seen this before in online mean computation. This is what inspired me to write this blog post: to compare the performance of the different RNN based on online mean computation and see how well each performs.

When I look at these update rules, the first that catches my eye is the mixture. Moreover, what I notice is that the mixture is always bounded by values $$\alpha$$ and $$\beta$$ such that $$0<\alpha<z<\beta<1$$. This means that the mixture cannot preserve information forever, as it will always decline by at least $$\beta$$. In particular, it is not capable of computing an online uniform mean, as outlined in a previous [blog post](https://baerververgaert.github.io/Mathemagics/2022-01-11-Mean-Computing-Online-Algorithms.html). It also cannot perfectly set the new information $$n$$ to the hidden state, because $$\alpha$$ guarantees some leakage from the previous hidden state.

It is clear that $$n$$ computes new information to be added to the old information. Exactly, how $$n$$ is computed though is interesting. Instead of using a regular single-layer neural net, it does something special. First it processes the input linearly. Then, it processes the previous state linearly, but it filters some of that informaton based on the output of another single-layer neural net. We can think of this as $$n$$ first extracting all information it could possibly want from the previous hidden state and then selecting only what it thinks it will need.

The new information is then mixed with the old information in accordance to the extend the GRU thinks is more important.


# The Big Guns - Deep Mathematics

Now let's bring out the big guns. We have seen how we fomulate the problems that RNN solve, and we have seen how RNN architectures do computation. Thus far we've concluded that RNN compute information from sequences of variable length to sequences of variable length. We've seen that information may struggle to remain important over many iterations, or that the network may struggle to utilize its features. To fully appreciate and understand the workings of the RNN let's look at some related mathematical fields and see what it tells us about the behaviour of RNN. 

### Dynamical Systems

First we will look at the RNN, but the arguments and concepts involved hold also for the LSTM and the GRU. This is because we will be reasoning about functions in general and not just specific RNN architectures. 

The regular RNN we can understand through its update rule 

$$h_n = \sigma ( W_{hi} x_n + W_{hh} h_{n-1} + b_h).$$

Let's first see what happens if we set all input to zero, i.e. $$0=x_1=x_2=\ldots=x_n$$ and $$h_0=0$$. Then we have that $$h_1 = \sigma( b_h)$$ and $$h_2 = \sigma( W_{hh} \sigma(b_h) + b_h)$$. We can clearly see that the $$h_n$$ are just going to be the repeated application of the same function, namely 

$$\sigma(W_{hh} \cdot + b_h).$$

Analysis of repeated functions are the domain of dynamical systems. In dynamical systems we study how a function $$f$$ behaves when we repeatedly apply it to itself. Thus given initial value $$x$$ we are interested in the sequence 

$$
\begin{align}
h_1 =&\ f(h_0) = f(h_0)\\
h_2 =&\ f(h_1) = f(f(h_0))\\
h_3 =&\ f(h_2) = f(f(f(h_0)))\\
\vdots&
\end{align}
$$

This looks familiar, right? It is very similar to how we first described the problem that RNN solve. However, we are missing here the input $$x_n$$. We will get to that later. For now we will keep to this formulation of dynamical systems. 

Note that if the $$x_n$$ are all the same constant than the RNN again is a dynamical system, because $$x$$ never changes the update rule $$h_n = rnn(h_{n-1},x)$$ can be seen as applying a function $$f$$ over and over again to the $$h_n$$, i.e. $$h_n = f(h_{n-1})$$.

Note further that if the $$x_n$$ are periodic with period $$k$$, i.e. $$x_1 = x_{k+1}, x_2 = x_{k+2}, \ldots $$ than the RNN again describes a dynamical system every $$k$$ steps with the function 

$$h_1 = f(h_0) = rnn(\ldots rnn(rnn(h_0,x_1),x_2),\ldots,x_k).$$

Finally, if the $$x_n$$ follow themselves a dynamical system, say $$x_0=x, x_1=f(x_0), x_2=f(x_1),\ldots$$ then the RNN can again be described as a dynamical system with two sets of variables 

$$
\begin{pmatrix} x_n \\ h_n \end{pmatrix}
=
\begin{pmatrix} f(x_{n-1}) \\ rnn(h_n,f(x_{n-1})) \end{pmatrix}.
$$

Thus we see that the analysis tools of dynamical systems cover a wide variety of input patterns, and are suitable for discussing the behaviour of the RNN. 

### Bifurcation Theory

One of these tools is the study of bifurcation theory which is interested in attractors of a dynamical system subject to a parameter $$\gamma$$. Attractors are points or states that a system converges towards. For the RNN we are interested of attractors and repellants of the RNN with a given input $$x$$ as this tells us which information the RNN is gravitating towards and which information it wants to forget.

To give a bit more detail, atrractors are stable points, meaning they are points of the form $$h=f(h)$$, which also attract nearby points closer to itself. Conversely, repellants are also stable points $$h=f(h)$$ which repel nearby points away from itself. There are also saddle points which are again stable points that attract some nearby points but repel others. Finally, we also have orbits. Instead of stable individual points, orbits are collections of points that are stable. So a point $$h$$ in the orbit maps to another point in the orbit but never outside the orbit. Orbits can also be attractors, repellants or saddle-like. It is easiest to think about orbits as going around in a chain, i.e. $$h_1\rightarrow h_2 \rightarrow h_3 \rightarrow h_1 \rightarrow h_2 \ldots$$. In the case the orbit is an attractor then nearby points are pulled into the orbit and start cycling as well.

Bifurcation theory is interested in the behaviour of the attractors of dynamical systems with an extra parameter, so they would be interested in systems of the form

$$f_\gamma(h) = g(h,\gamma).$$

Thus they are interested in sequences of the form $$h_1 = g(h_0,\gamma), h_2=g(h_1,\gamma), h_3=g(h_2,\gamma), \ldots$$. This should look familiar because it is the form of the RNN with static input. 

From bifurcation theory we know that even simple one-dimensional systems can have many chaotic attractors. The prime example of this is the dynamical system 

$$g(h,\gamma) = \gamma h (1-h).$$

If the value $$\gamma$$ is between $$3$$ and $$4$$ then this dynamical system has many chaotic attractors. Moreover, approximating multiplication is well within the range of capabilities for neural networks (this was part of my thesis :D). Thus chaotic attractors are an interesting approach to RNN. On the one hand, chaotic attractors means that the RNN can achieve many different attractors which might help explain why it is so remarkably capable. Simultaneously, the chaos in the attractors means that even with small pertubation the attractors can change quite drastically. This implies that RNN might find it difficult to fine-tune within a field of chaotic attractors as these are constantly jumping around. Diminishing the learning rate with a learning rate scheduler might help the RNN to converge on the right set of attractors. It might also prove beneficial to develop the tools which can recognize that a RNN is displaying and utilizing chaotic behaviour. Not to avoid the field of chaos, because it can be very useful to the RNN, but to help it find its optimum within it at the right speed.

Related to attractors we can view Andrej Karpathy's results, particularly the xml generator, is that it learn orbits. Correctly opening and closing xml tags is something it could have learned by constructing the orbit that creates the tag along with the orbit that closes the tag. If there is a sensitivity to attractors there well might also be a sensitiviy to orbits (attractors are just a special case of orbits). Since the network can control the number of attractors, it is not too far fetched that it can also control the number of orbits. Needing only to construct a small number of orbits my be beneficial as learned parameters can be in a field of less chaos. Similarly, once it is done with an a tag, the next character could have that orbit as a repellant such that it doesn't continue looping, thus making use of the input $$x_n$$ as a method of gear shifting.

### Differential Equations

Finally, there is a last mathematical tool closely related to dynamical systems that may prove insightful, namely differential equations. Where dynamical systems studies where a function is going after repeated application, a differential equation describes the change in a system. Particularly, differential equations are often studied in terms of that function, i.e. often we have equations of the form 

$$\frac{dh}{dt}(t) =  f(h(t)).$$

Particularly well known examples are 

$$\frac{dh}{dt} (t) = \alpha h(t),\ \frac{dh}{dt}(t) = k h(t) (L - h(t)).$$

Differential equations are related to dynamical systems through the Taylor approximation which states that 

$$ h(t) \approx h(t_0) + h'(t_0)(t-t_0) + h''(t_0)(t-t_0)^2/2 + \ldots = \sum_{k=0}^\infty \frac{h^{(k)}(t_0)}{k!}(t-t_0)^k.$$ 

In particular we can view one step of the RNN with constant input as the approximation 

$$ h(t) \approx \sum_{k=0}^\infty \frac{h^{(k)}(t-1)}{k!}.$$ In particular, if the differential equation is called autonomous we have precisely 

$$\frac{dh}{dt}(t) = f(h(t))$$

whereas non-autonomous have the general equation 

$$\frac{dh}{dt}(t) = f(h(t),t).$$

Such a differential equation can be easily applied to compute all the higher-order derivatives by chain rule. Particularly, if $$h$$ is autonomous, then each higher-order derivative $$h^{(k)}(t)$$ is some function $$g_k$$ of $$h(t)$$ of the form 

$$h^{(k)}(t) = g_k(h(t)).$$

In this case the Taylor approximation now reads as 

$$ h(t) = \sum_{k=0}^\infty \frac{g_k(h(t-1))}{k!} $$

which is really just a functon of $$h(t-1)$$. This function can be approximated with a neural network as 

$$h_n = nn(h_{n-1}).$$

However, RNN also receive input $$x_n$$ which suggest a differential equation of the form 

$$\frac{dh}{dt}(t) = g(h(t),x(t)).$$

How we now analyze the behaviour of the differential equation depends upon our interpretation of $$x_n$$. These can either be viewed as completely time-independent and possibly stochastic values. In which case we read the Taylor approximation as 

$$ h(t) = \sum_{k=0}^\infty \frac{ g_k(h(t-1),x(t))}{k!}.$$

However, it is rarely the case that $$x_n$$ contains no time information. For example, in text generation the very first character in a sentence is capitalized which is strings is a different symbol from the lower case. When inserting characters, the $$x_n$$ can depend on $$n$$, and thus our analysis is incorrect. 

Instead we can consider the posssibly non-autonomous differential equation which also describes the change in $$x(t)$$ as 

$$\frac{dx}{dt}(t) = g_x ( x(t), t).$$

This means that the corresponding Taylor approximation is 

$$ h(t) = \sum_{k=0}^\infty \frac{g_k(h(t-1),x(t),t)}{k!}.$$

This differs slightly from the RNN as this is in fact approximated by an RNN with not just $$x_n$$ as input, but also $$n$$ itself as 

$$ h_n = rnn(h_{n-1},x_n,n).$$

This analysis hints that positional information can be not just beneficial to transformer nets but also to RNN networks. Note that the inclusion of $$x_n$$ is not redudant when $$n$$ is also added as $$x_n$$ may be a stochastically dependent variable.

# Experiment

This concludes the theory for this portion. We will now do a couple of experiments and see what we can learn from these. You can find their notebooks at my github here. 

