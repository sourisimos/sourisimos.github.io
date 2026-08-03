---
title: 'A (not so) short and (yet very) intuitive view of our article Where Rectified Flow Leaks'
date: 2026-08-03
permalink: /posts/2026/04/WhereRFLeak/
excerpt: "The original paper can be a bit off-putting, yet I believe the ideas & intuitions behind it are (humbly) crazy cool. So here is a blog post that I hope is fun to read, with everything we did in our paper :)"
tags:
  - Generative Models
  - Flow Matching
  - Memorization
  - ICML
---

This post is meant to be a nice walkthrough of our [ICML 2026 paper](https://openreview.net/forum?id=Ty5X41WbJw), written with Gabriel Meseguer-Brocal and Geoffroy Peeters. The paper itself is fairly abstract and mathematical, and I somehow managed to avoid spelling out most of the intuitions we actually had along the way... Yet beyond the love of maths and CS, the love of *understanding* is probably even more attractive, right? So here it is: hopefully a really vivid and pleasant blog post to explain those intuitions (and there are a lot of them, from a rope metaphor to spaghetti).

## What do we want to do?

First things first, let's start with what we wanted to do.

Generative models are trained on enormous piles of data, a lot of it copyrighted, and that has made one question suddenly quite pressing. It has been showing up in lawsuits from [Getty against Stability](https://www.courtlistener.com/) to the [major record labels against Suno and Udio](https://www.musicbusinessworldwide.com/as-suno-and-udio-admit-training-ai-with-unlicensed-music-record-industry-says-theres-nothing-fair-about-stealing-an-artists-lifes-work/): what does a model actually keep about what it was trained on?

There is a whole field, with many cool figures, trying to tackle this question: *memorization*. Importantly, there isn't just one kind of memorization. The obvious one (and the main focus for now) is verbatim memorization, the model handing back a training image or a melody note for note, [like this one from Carlini et al.](https://arxiv.org/pdf/2301.13188). It is very impressive, but it is also rather rare, since it only happens to a handful of examples among tens of thousands. Instead of catching memorization at generation time, you can also look for it *inside* the model, where it takes a subtler form. A model can treat its training data differently from data it has never seen, reconstructing it a little more faithfully, behaving a little differently near it, following a trajectory a bit more specific to it, all without ever reproducing it.

Since "memorization" usually points to the verbatim kind, and to keep things clearly separated, we named this measurable asymmetry the **membership signal**: any trace you can read off a model that tells you whether a given sample was likely in its training set. I insist on one thing: it is not a fixed mathematical definition but a *concept*, anything that leaks information about whether a data point belonged to the training set.

Something often misunderstood, and what makes the whole memorization story (verbatim and membership signal alike) so tricky is that you can train a model that has clearly absorbed a lot about its data while its loss curves look perfectly healthy, with no overfitting, validation dropping smoothly, nothing to see. So the question is: if it isn't on the loss curve, is there a *where* in the model's behaviour that could be hiding this information?

## Just a bit of explanation on Rectified Flow / Flow Matching

Memorization phenomena are so subtle that techniques are usually tailored to a specific type of model: DDIM/DDPM, GANs, [Rectified Flows](https://arxiv.org/abs/2209.03003) / [Flow Matching](https://arxiv.org/abs/2210.02747)... Those last ones are the ones we focused on.

> From now on I'll just say Flow Matching for both Rectified Flow and Flow Matching. They are not *eeexactly* the same, but I'm quite sure it'll be clear enough, and clearly less verbose. One other thing: here I am **not** talking about the Reflow procedure.

Before getting to the core, let me quickly recall the learning paradigm of Flow Matching. You are probably aware that It is a superstar (... even if its time may be coming, with one-step generation paradigms like MeanFlow or Shortcut models) behind systems like [Stable Audio](https://arxiv.org/abs/2407.14358), [FLUX](https://blackforestlabs.ai/), and [Stable Diffusion 3](https://arxiv.org/abs/2403.03206).
Here, we focus on the *linear* interpolation version. Pick a noise point $$x_0$$ and a data point $$x_1$$, draw the straight line between them, and a point on that line is $$x_\lambda = (1-\lambda) x_0 + \lambda x_1$$. Here $$\lambda$$ says how far along you are: $$\lambda = 0$$ is pure noise, $$\lambda = 1$$ is the data. The model's whole job is to look at a point on this line and predict the *direction* that carries it toward the data. At generation time you start from noise and move along the path with small explicit Euler steps, and the model hands you the direction at each one.

Basically, imagine you are a Noise, at home (i.e. in your so-cosy Gaussian-Distribution City) and you want to go to your bakery (i.e. in Latent Representation City, a very famous one) to buy a baguette (you are a French Noise). But your GPS only gives you the DIRECTION you should head in. So what you do is check the GPS, walk a bit, check again to see if it changed, and so on. And this GPS didn't come from nowhere: you and your Noise friends trained it beforehand by going from many places to many others, and each time, while on your way, you always fed it the straightest direction toward your destination.

Well, congratulations: if you were an extra-dimensional point, you just trained your own Flow Matching GPS using the linear interpolation path!

![The interpolation path](/images/rf-leak/01_interpolation_path.svg)

I actually did a whole blog post on RF / FM, so make sure to check it if you need a reminder. ;)

## Where did we look for information?

Finally (and I promise, after this we're done with the setup), let me describe the probe itself.

Take one data point, from the training set or the held-out set, mix it linearly with noise to land at some position $$\lambda$$ on the path, let the model predict the direction from there, take one step, and reconstruct where it thinks the data should be. Compare that to the true point: the distance is the model's error at that position. That is just the Flow Matching loss itself, read at a single $\$lambda$$ instead of averaged over all of them:

$$
\mathcal{L}(\lambda) = \mathbb{E}\big[\, \lVert v_\theta(x_\lambda, \lambda) - v \rVert^2 \,\big]
$$

Now, it's worth pausing on what this loss actually contains, because it's the key to everything after. A little algebra splits it into three parts:

* the **approximation error**: how far the model is from the best possible predictor,
* an **irreducible noise** floor that no model, however good, could ever predict,
* and a **cross-term** that couples the two, measuring how much the model's error lines up with that noise.

The first two are, in a sense, "innocent": every model pays them, on training and held-out data alike. The third one is the interesting one, and it's the only place where being a member of the training set could leave a mark. (Keep it in mind, because we'll come back to it and make it fully precise near the end.)

So how do we get to it? The trick is a subtraction (what a trick!). We run the probe on many samples, separately for the training set and the held-out set, and look at the **average difference** in error between the two, as a function of where we started. Since the two "innocent" terms behave the same on seen and unseen data, they cancel out, and what survives is essentially that lonely cross-term, the one place a training bias could hide.

That those two terms behave the same on training and held-out data does rely on two mild assumptions, which just describe a properly trained model:

**Assumption 1 (Uniform approximation error).** The model is no closer to the best predictor on its training points than on the population at large. In other words, it hasn't done anything special on the samples it saw, which is what you get when the model hasn't overfit in the classical sense (early stopping helps here). Note that this does *not* forbid a train-test gap in the loss; it only forces that gap to go through the cross-term.

> *Wait, isn't that assuming the whole thing away?* It's the natural objection: if the model isn't any closer to ideal on its training data, where could the leak even be? The trick is that "closer" is about the *length* of the model's error, while the leak lives in its *direction*. A model can be exactly as far from ideal on a member and a non-member, yet be wrong in a telltale direction on the one it saw. So this assumption doesn't rule out memorization; it just rules out the crude kind (being plainly more accurate on your own training set), which is what lets us isolate the subtle kind. This will click into place when we get to the triangle near the end, so for now just file it away and trust me!

**Assumption 2 (Representative sample).** The empirical noise floor matches its true value, which is essentially just the law of large numbers for a large enough dataset.

And that's the whole idea: the train-test gap is a subtraction *designed* to isolate the one thing we want, a bias (if it exists) toward the training data, el famoso membership signal.

## The observation that started it all

So now we had a clean setup and a way to measure this bias, but we genuinely didn't know what it would look like. Maybe a flat line, if there's no bias. Maybe a bump near $$\lambda = 1$$, close to the data, where you'd naively expect memorization to live. Maybe noise.

Instead, in every single configuration we tried, the gap traced the same curve: a **bell**. Flat and near-zero at both ends of the path, rising to one clean peak somewhere in the middle.

![The bell-shaped train-test gap](/images/rf-leak/02_bell_curve.svg)

And it wasn't specific to one dataset or one model. It came back for audio and for images, for transformers and for UNets, across latent spaces and noise levels. That kind of stubbornness is usually the sign of something structural underneath, and it's exactly what pushed us from "we found a nice measurement" to "we need to explain *why*". *Why* a bell at all, and *Why* does it peak in the middle?

Even crazier: this doesn't appear randomly at the end of training. It grows steadily during training, well before any visible overfitting.

## The Hourglass

Now that we've run the experiments and laid out the setup, let's get to the real explanation.

Picture the Flow Matching path in 3D: two flat planes facing each other, the noise distribution on the left and the data distribution on the right, with the interpolation paths running between them. Every sample is a single straight strand, tied from its noise point on the left to its data point on the right, so the whole thing is a bundle of straight spaghetti stretched across the gap.

Now pull the two clouds apart and look at the bundle from the side. Because each strand joins a scattered noise point to a scattered data point, the bundle isn't a uniform tube. It forms an **hourglass**: wide at both ends where the points are spread out, and pinched in the middle where all the strands funnel through a narrow waist.

And this bundle isn't only a picture of the data. It's essentially the flow field the model ends up learning: at each point in space, the strands passing through are exactly what tell it which way to push. So the shape of the bundle *is* the shape of the model's job, and what an interesting coincidence that the pinch of the hourglass seems to sit at the same place as the peak of the error, no?

![The hourglass](/images/rf-leak/03_hourglass.svg)

Now think about the model's job at each point: given your position, predict your direction.

Near the ends, this is easy, and for the same reason at both. At the start, out in Gaussian-Distribution City, your exact position sits on essentially one sensible trip: you're far from the bakeries, they're all clustered off in one general direction, so where you stand points you cleanly toward where to go. And at the very end, near the data (i.e. your bakery), it's the mirror image: you're right next to your destination, only one trip realistically ends here, so again your position tells you your direction almost for free. Either way, few contradictory routes pass through where you're standing, so a simple rule nails it.

The middle is where it breaks. There, your position sits in the crowded central square, the spot where trips coming from every home and heading to every bakery all cross. The same location is compatible with dozens of different directions, and no simple rule can tell which one is yours. Should I take this road or that one? Genuinely hard to say, and it's exactly here that knowing a shortcut, or the one right answer, would save you the most.


That "simple rule" is the important part, but let me keep the precise meaning of it for the next section. For now, the picture is enough: there is a region on the path, not a single point, where position becomes a much worse guide to direction. In the clean Gaussian case, we can even compute where that region sits *in closed form*, from the variances of the noise and the data alone, with nothing about the model in it.

> The signal peaks where the model has the least useful guidance from position alone.

That's the part I find genuinely surprising, and it's worth sitting with, because the naive expectation could easily be different. You might expect the model to leak most where it has had the clearest opportunity to gain an advantage over time, or maybe right at the beginning, where everything looks like noise. But near the noise end, the direction is actually quite structured: the data cloud sits roughly in one place, so the model still has a fairly good idea of where to go. The more confusing region is not the pure-noise end, but the crowded middle.

Still, I've been a little sloppy on purpose. I kept saying the model runs out of *information*, when what really runs out is one particular kind of it. Fixing that sloppiness is another really cool piece of the paper, so it deserves its own section.

## Why the leak has to exist

Okay, as I just said, I lied a little (purely for pedagogical purposes, you know...). Near the pinch, the model doesn't run out of *all* information, only the easiest kind to learn: the linear one, that "read your direction straight off your position" rule we just watched collapse in the crowded square.

> One quick disambiguation, because *linear* is doing double duty here. The *path* is linear: the straight segment from noise to data, the spaghetti strands, and that never changes. What changes along the path is whether the *direction* can be read off your *position* by a simple linear rule. Two different "linears", then: the road is always straight, but the guidance is only linear near the ends. It's the second one that collapses at the pinch, and the leak lives in that collapse.

So what does *nonlinear* actually mean, back in the streets? A linear rule is a smooth one: shift your position a little and the right direction shifts a little too, in proportion. Step a bit east, aim a touch more west, nothing dramatic. That's fine near the ends. But in the crowded square it breaks, because two spots barely a step apart can call for completely opposite directions: this walker peels off east toward their bakery, that one veers west toward theirs, and they're standing right next to each other. No smooth, gentle rule can do that. You need a twistier one, a rule that turns sharply depending on the fine details of exactly where you stand.

And notice *why* the smooth rule could never do the memorizing for us, however much data it saw. A linear rule is a single instruction that everyone follows the same way: "lean east in proportion to where you stand." There is only one such rule for the whole city, so it has nowhere to store "you, specifically, third door on the left." To single out one trip, the rule would have to change depending on which trip you are, and a rule that changes from spot to spot is exactly what we just called twisty. That's not a coincidence: bending the rule to fit one trip *is* the definition of going nonlinear. So memorization doesn't just happen to be easier in the nonlinear regime; it can *only* live there.

And here is where it gets interesting, because being that precise cuts both ways. Ask what the sharp turn is actually *for*. Part of it is steering you toward the right *neighbourhood*, a genuine rule of the city ("at this kind of junction, bear east") that helps any trip heading that way. That part generalizes. But part of it is steering you toward *one exact door*, the precise way this single trip happened to end, which helps nobody else. That part is memorization.

Here's the trap, and it's (one of) the hearts of the paper. A single French Noise doesn't *know* about neighbourhoods. It has never seen the city from above; all it knows is its own door. So when it learns its turn, it learns the whole thing, the general "bear east" and the private "then the third door on the left" fused into one gesture, with no way to tell which part was a real rule of the city and which was just where it personally ended up. The neighbourhood only exists in hindsight, once you look across thousands of trips and notice what they share. But the model never learns from neighbourhoods. It learns from doors, one at a time, and inside any single door the shared rule and the private quirk are welded together.

So when the model reaches for the useful turns, the ones that find the right neighbourhood, it can't help but follow some of them all the way to the door. Fitting that last, useless bit of precision is the *price* of learning the real city, and a door that belongs to a single trip is exactly a trace of that trip having been walked, sitting only on the samples the model actually saw. That is the membership signal.

This is also why I would not describe it as ordinary overfitting. It is not just a training pathology that disappear once you add a bit more regularization; it is tied to the way the model learns the transport problem itself. It's baked into what learning nonlinear structure means at all, and it concentrates at the pinch, because that is the one place the model has no linear alternative to fall back on.

## Why nothing shows up on the dashboard

A bit more explanation here, but hang in there, we're getting close to the end.

So the signal is always there, we understand why it looks like this and why it has to appear. And yet no standard metric catches it. Why? There are two separate reasons, and they compound. This is exactly why we needed a dedicated gap probe: the leak is not a big visible degradation of the loss, it is a small, localized, directional bias. If you only look at the usual averaged training and validation curves, you're looking in the wrong place with the wrong instrument.

**Dilution.** The signal lives in a narrow band near the peak, but training monitors the loss *averaged over the whole path*, and spreading one sharp spike across the entire interval barely moves the average.

**Masking.** This one is subtler. Early in training the model is still learning the generalizable structure, and that lowers *both* the training and the validation loss together. Underneath, on the training side, the leak term is quietly growing, but it's buried under the much larger gains from generalization. On the validation side the leak term is absent, since the model never saw those sample-specific residuals. So validation keeps dropping and looks perfectly healthy, while the membership signal accumulates silently.

![Masking](/images/rf-leak/05_masking.svg)

To see this cleanly, let's finally cash in that cross-term I asked you to keep in mind.

Flow Matching is trained by least squares, so the loss is a squared distance, and any squared distance splits into a triangle. Draw the model's prediction, the optimal prediction, and the true target as its three corners:

![The triangle of the three loss terms](/images/rf-leak/06_triangle.svg)

Its sides are our three terms: the **approximation error** (how far the model is from the optimal predictor), the **irreducible noise** (the sample-specific residual that no predictor can infer from position alone), and the **cross-term** (the alignment between them).

That word, *alignment*, is the key. The membership signal is not simply the model being more wrong on its training data. Standard metrics mostly see magnitudes: how large the error is on average. But the signal lives in a *correlation* between the model's error and the sample-specific residual. Two errors can have almost the same size while differing in orientation. The leak is not a louder error; it is a biased direction. Same-length arrow, different angle, and nobody was looking at the angle.

Now compare training points to held-out points. On held-out data, the model has never seen those sample-specific residuals, so there is no systematic reason for its error to align with them: the cross-term is essentially zero. On training data, however, the model was fitted on those very residuals, so a small alignment appear. Meanwhile the other two sides of the triangle behave the same on seen and unseen points (this is where Assumption 1 finally pays off), so they cancel in the difference.

> Take the train/held-out difference, and everything cancels except the term that carries membership. The gap *is* the membership signal.

Once the gap reduces to this cross-term, the rest becomes a question of location: where along the path should that term be largest?

This is where the Gaussian calculation comes in. In the clean Gaussian case, the optimal predictor is provably linear, so studying a linear model isn't a simplifying approximation, it's the exact object the network is trying to reach. The membership signal then turns out to be proportional to the *irreducible variance*, the part of the velocity that no linear rule can predict from position. And that variance is largest exactly where the linear information vanishes: at the pinch of the hourglass. The bell shape, the peak location, and the $$1/n$$ decay all drop out of that one proportionality.

**Why a Gaussian story for a real transformer, though?** This is the move that needs justifying. Two reasons. First, spectral bias: the network learns the linear part first, so it really does defer the nonlinear scramble to where the linear signal dies. Second, we work in the latent space of an autoencoder, and those latents are *built* to be roughly Gaussian and isotropic, whether through an explicit KL penalty or through bounded activations like tanh. So the assumption our theory needs is approximately satisfied by construction, which is why the closed-form peak actually lands on real data.

## How the theory survives contact with experiments

A theory that predicts a specific location is a theory you can try to break, so we did, in both directions.

First we moved the peak on purpose. Our formula says the location depends only on the covariances at the two ends of the path, so we pushed on each end: scale up the noise variance and the peak shifts exactly as predicted, swap the training dataset for one with different structure and the peak shifts again, prediction still matching.

Then we did the opposite and varied everything the theory says should *not* matter, like the architecture (transformer versus UNet), the model size, and the sampling scheduler. The peak didn't move; only its height changed.

> Data geometry sets the location, model choices set the magnitude, and the bell shape itself is universal.

The most useful stress test is the case where the closed-form location does *not* work perfectly. On image latents from a VAE that is genuinely far from Gaussian, with heavier tails and stronger correlations, the formula no longer predicts the peak accurately. But this is not the leak disappearing. The gap is still there, and the bell-shaped profile is still visible. What fails is the Gaussian shortcut for *locating* the maximum, not the mechanism underneath. And that distinction matters: the theory never claimed every real latent space is perfectly Gaussian. It says the signal should concentrate where the shared, predictable part of the velocity is weakest and the sample-specific residual strongest. Gaussianity just gives a clean closed-form version of that story; non-Gaussian latents make the location harder to compute, without removing the phenomenon. A story that only ever succeeds is a bit suspicious. One that predicts its own failure mode is telling you it caught something real.

## Turning it into an attack

If the signal is real, it should be exploitable, so as a proof of concept we made it into one. Feed the full per-$$\lambda$$ error profile of a sample into a small classifier and ask it: member or not? Because the profile encodes the whole shape of the bell, it's far more informative than any single measurement, and as far as we know it's the first membership inference attack designed specifically for Flow Matching, comfortably beating attacks ported over from the diffusion literature ([SecMI](https://arxiv.org/abs/2302.01316), [PIA](https://arxiv.org/abs/2308.06405)).

The point isn't the attack itself, but the confirmation that the structure we derived on a whiteboard translates into a practical, measurable risk.

## Where this goes

There's also a practical reason this middle region matters so much. In modern Flow Matching systems, the center of the path is often treated as especially informative, and empirical weighting schemes tend to put a lot of emphasis there, as discussed for instance in the Stable Diffusion 3 paper. This isn't an accident. Near the noise end, the target is dominated by randomness; near the data end, the model is close to reconstruction. In between, it has to solve the real transport problem: connect global structure to noisy coordinates. That is precisely where the hourglass pinches. The model is asked to learn the most useful part of the mapping at the exact place where the sample-specific residual is hardest to ignore. So the center isn't only where the theory predicts the leak to peak; it's also where practical training recipes already ask the model to pay the most attention, which quietly makes it a privacy/efficiency trade-off.

And that's basically it! Well, almost, because plenty is still open.

We only glanced at reflow, and the early signs are that the bell survives but flattens out, which hints that reflow might double as a natural defence. Text-conditioned generation, stronger threat models, larger deployed systems: all still to do.

But the core of it is small, and I think a little beautiful. A generative model leaves a structured, predictable trace of its training data, sitting at a spot you can compute ahead of time from the data alone, growing silently while every dashboard says the model is fine. And it's there not because the model failed but because it *succeeded*, since absorbing a little of each sample's fingerprint is the unavoidable cost of learning the structure that generalizes.

The model's fingerprint hides where it knows the least, and it just can't help but leave it there.

----------

Thanks for reading! If you have questions, disagreements, or spot something I got wrong, please do reach out, I would genuinely love to hear it.

Everything here reflects my own understanding, which is certainly partial. The figures are sketches of the intuitions rather than reproductions of the plots in the paper, so take them as drawings on a whiteboard, not as results. :)