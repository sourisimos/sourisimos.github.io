---
title: 'AI in music: An historic review'
date: 2024-11-05
permalink: /posts/2024/11/blog-post-1/
tags:
  - Music
  - Review
  - AI
---

This post is partly inspired by the lecture "AI vs AI: Artificial Intelligence vs Artistic Intelligence" by Jean-François Zygel, presented at the Halle aux Grains in Toulouse, France. With the author’s permission, we will explore the evolution of algorithmic music generation, tracing its origins to contemporary methods and future perspectives in AI for music.

Here, I'll provide a quick overview, but many links are available if you'd like to dive deeper into any topic. I hope you'll forgive the occasional use of Wikipedia; it often proves to be one of the best resources for gathering extensive, accessible, and free information. <br>
Finaly, since music is fundamentally about listening, I've also included many videos with musical examples and, in some cases, explanatory commentary. Just click on 🎶. 

Also, you can find a relatively up-to-date state of the art on music-related algorithms [here](https://carlosholivan.github.io/DeepLearningMusicGeneration/#figaro-generating-symbolic-music-with-fine-grained-artistic-control).

## Music as Algorithm: The Beginnings of Algorithmic Composition

Since Antiquity, music has followed mathematical and structured principles, explaining why it was part of the [mathematical quadrivium](https://en.wikipedia.org/wiki/Quadrivium) alongside arithmetic, geometry, and astronomy. However, modern algorithmic composition (or generative music) goes beyond following explicit rules: it involves works whose final creation “escapes” human intervention to some extent. 
We will also exclude strictly automatic compositions, in which a machine executes an entirely pre-programmed piece without human intervention, as well as purely random compositions.


<div style="float: right; margin-left: 10px;">
    <img src="https://gate.unigre.it/mediawiki/images/thumb/b/b9/7700_3271_3580-016_944.jpg/300px-7700_3271_3580-016_944.jpg" alt="Arca Musarithmica" width="150"/>
    <p><em>Arca Musarithmica</em></p>
</div>



[Athanasius Kircher](https://en.wikipedia.org/wiki/Athanasius_Kircher) (1602-1680) is often considered the pioneer of algorithmic composition with his invention of the [*Arca Musarithmica*](https://larkfall.wordpress.com/2014/06/06/kircher-schotts-computer-music-of-the-baroque/) (~1650).
This "music box" contains cards and tables combining predefined notes and rhythms that the user assembles following specific rules. Although theorists before him explored musical systematization through combinatory tables, Kircher was the first to physically implement a machine capable of autonomously generating melodies. [🎶](https://www.youtube.com/watch?v=zpfq2L5X6yU)



In the 18th century, composers like [Mozart](https://en.wikipedia.org/wiki/Wolfgang_Amadeus_Mozart), [Haydn](https://en.wikipedia.org/wiki/Joseph_Haydn), and [C.P.E. Bach](https://en.wikipedia.org/wiki/Carl_Philipp_Emanuel_Bach) developed musical games that used chance to create melodies. By rolling dice, each throw selected a measure from a predefined series, generating unique pieces. The most famous example is Mozart's "musical dice game" (1787), which demonstrates how composers of the era explored preliminary forms of musical generation without direct intervention on the piece’s structure. [🎶](https://www.youtube.com/watch?v=9Zdg6Ec4mVw)

## The Computer Age and the Rise of Modern Algorithmic Music

Although theorists like [J.F. Rameau](https://en.wikipedia.org/wiki/Jean-Philippe_Rameau) explored ways to [systematize music](https://en.wikipedia.org/wiki/New_System_of_Musical_Theory) in the 19th century, algorithmic composition experienced a major revival in the 20th century with [Iannis Xenakis](https://en.wikipedia.org/wiki/Iannis_Xenakis) (1922-2001), a mathematician and architect turned composer, often regarded as one of the fathers of modern algorithmic composition and stochastic music (which, while we won’t explore it in detail here, remains a particularly fascinating subject!).
Lacking formal classical musical training, Xenakis applied mathematical and probabilistic principles to music, deeply influencing contemporary music. He created works like Achorripsis (1956-57), where each sound event is generated by probability calculations, allowing the structure to develop semi-randomly.[🎶](https://www.youtube.com/watch?v=WasFTDq0dJI)

<div style="float: right; margin-left: 10px;">
    <img src="https://i0.wp.com/120years.net/wp-content/uploads/upic1-e1705496773502.jpg" alt="UPIC Program" width="220"/>
    <p><em>UPIC Program</em></p>
</div>

His [UPIC program](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=3425fc400cd2c4cf1aa9ff7231ef2a541e234c62) was a major innovation: it allowed visual forms to be translated into music, thus introducing a visual interaction in algorithmic composition. The graphic gesture became central, opening possibilities for generative and AI-assisted music and laying the foundation for composition tools where scientific models structure the work. [🎶](https://www.youtube.com/watch?v=nvH2KYYJg-o)


In the 1950s, [Lejaren Hiller](https://distributedmuseum.illinois.edu/exhibit/lejaren-hiller/) and [Leonard Isaacson](https://en.wikipedia.org/wiki/Leonard_Isaacson) published the [first book on computer-based composition](https://archive.org/details/experimentalmusi00hill/page/n5/mode/2up), based on the Illiac computer they developed [ILLIAC](https://en.wikipedia.org/wiki/ILLIAC), which generated a musical piece, the Illiac Suite [🎶](https://www.youtube.com/watch?v=fojKZ1ymZlo). Though coherent, the result was limited in originality, highlighting the early limitations of musical AI. Their approach relied on strict rules and probabilities without aesthetic depth, underscoring the first boundaries of musical AI.


## From Probabilistic Models to Machine Learning Algorithms

From 1990 to 2015, algorithmic music underwent a pivotal shift, evolving from simple probabilistic models to more sophisticated approaches such as recurrent neural networks (RNNs) and style-learning models. This shift laid the groundwork for the impressive advances in algorithmic music generation that followed. During this period, many machine learning technologies were adapted specifically for the purpose of music creation, marking a significant transformation in the field.

Starting in the late 1980s, [David Cope](https://en.wikipedia.org/wiki/David_Cope) advanced musical AI with [Experiments in Musical Intelligence (EMI)](https://quod.lib.umich.edu/cgi/p/pod/dod-idx/experiments-in-music-intelligence-emi.pdf?c=icmc;idno=bbp2372.1987.025;format=pdf), a program capable of generating new works in the style of famous composers. EMI take par of Latent Semantic Analysis to go beyond following musical rules ; it analyzes stylistic and structural features of composers like Bach, Beethoven, or Mozart, enabling the creation of pieces that convincingly imitate their writing styles. [🎶](https://www.youtube.com/watch?v=2kuY3BrmTfQ&list=PLNaK-WAWTgwudpV2B5xe7WUG8vRtEHnsI&index=1)


A few years later, in 1996, [John A. Biles](https://genjam.org/al-biles/genjam/biography/) developed [GenJam](https://igm.rit.edu/~jabics/BilesICMC94.pdf), a genetic algorithm to create and improve jazz solos in real time, able to "duo" with a musician. [🎶](https://www.youtube.com/watch?v=RDgJw2kiuWU)


In the early 2000s, Cope developed [Emily Howell](https://en.wikipedia.org/wiki/Emily_Howell), a system that take advantages of the first algorithms developped by cope but also enable the user to "dialogues" musically, integrating their preferences and demonstrating the evolution toward collaborative AI in music. [🎶](https://www.youtube.com/watch?v=QHJqp4SlsoU)


LSTM models (Long Short-Term Memory) enabled complex harmonizations, as demonstrated in projects like [BachBot](https://www.mlmi.eng.cam.ac.uk/files/feynman_liang_8224771_assignsubmission_file_liangfeynmanthesis.pdf) (2016) by Feynman Liang et al. [🎶](https://soundcloud.com/bachbot/sets/bachbot-com?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing) from Mircrosoft research team.


Another major advancement was [DeepBach](https://arxiv.org/pdf/1612.01010) (2017)  by Gaëtan Hadjeres, François Pachet, and Frank Nielsen, capable of creating chorales in Bach's style using Restricted Boltzmann Machines (RBM) combined with a convolutional neural network (CNN). [🎶](https://www.youtube.com/watch?v=QiBM7-5hA6o)


Google’s [Magenta platform](https://magenta.tensorflow.org/), launched in 2016, also introduced neural models to generate melodies and rhythmic patterns, paving the way for modern generative music.


## The Music Transformer: A Breakthrough in Musical Coherence

The [Music Transformer](https://arxiv.org/pdf/1809.04281) by Google Magenta's team Cheng-Zhi Anna Huang et al. (2019) represents a major breakthrough. [🎶](https://magenta.tensorflow.org/music-transformer). Using a modified Transformer architecture with relative attention, this model maintains coherence over long compositions by "remembering" musical motifs across extended structures. This innovation overcomes the limitations of LSTM and RNN models, allowing the Music Transformer to produce works that evolve fluidly over several minutes. Key limitations addressed include difficulty capturing long-term dependencies, sequential data processing (shifted to parallelism), and modeling complex structures. Moreover, it's also the first to adresse symbolic generation with deep neural network. 

## Emergence of New Research Axes

Since then, music generation has become closely intertwined with AI algorithms and deep learning. Numerous advanced models have sparked growing interest in the field, leading to the emergence of several specialized subfields. Of course, many of these models address multiple challenges simultaneously.

* **Multi-instrument complexity**: [MuseNet](https://openai.com/index/musenet/) [🎶](https://www.twitch.tv/videos/416276005) by OpenAI and [AudioLM](https://arxiv.org/pdf/2209.03143) [🎶](https://google-research.github.io/seanet/musiclm/examples/) by Google Research leverage the Transformer architecture to create orchestral compositions by managing harmony and interaction between instruments.
* **Voice and lyrics**: OpenAI's [Jukebox](https://arxiv.org/pdf/2005.00341) [🎶](https://soundcloud.com/openai_audio/pop-in-the-style-of-the-beatles-openai-jukebox?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing) introduces music generation with voice and lyrics, using VQ-VAE networks and Transformers. Jukebox’s VQ-VAE (Vector Quantized Variational Autoencoder) captures music in multiple layers of resolution, making it possible to generate vocal music with timbre and inflection nuances, although lyrical coherence remains a challenge.
* **Real-time synthesis**: [RAVE](https://arxiv.org/pdf/2111.05011) [🎶](https://www.youtube.com/watch?v=dMZs04TzxUI)uses a variational autoencoder (VAE) for fast music generation, opening possibilities for dynamic and interactive music.
* **Text-to-music generation**: Google’s [MusicLM](https://arxiv.org/pdf/2301.11325) [🎶](https://google-research.github.io/seanet/musiclm/examples/) and Meta's [MusicGen](https://arxiv.org/pdf/2306.05284) [🎶](https://ai.honu.io/papers/musicgen/) combines textual descriptions and sound features to generate music based on verbal prompts.
* **Symbolic music generation**: [MuseCoco](https://arxiv.org/pdf/2306.00110) [🎶](https://ai-muzic.github.io/musecoco/) from Microsoft [Muzic](https://www.microsoft.com/en-us/research/project/ai-music/) project focuses, among other things, on the generation of scores.


These can be regarded as major advancements in these fields. However, since 2023, numerous research efforts have delved deeper, leading to a substantial proliferation of models, which I won’t present exhaustively here for the sake of simplicity.

To conclude, the rise of music generation has closely followed the development of deep learning algorithms, leading to significant breakthroughs. However, while AI-generated compositions may seem impressive to non-expert listeners, there remains a long way to go before AI can truly challenge the masterpieces created by legendary human composers over centuries. Although AI is making strides in areas like improvisation and generation, it still struggles to match the fluidity and instinctive nature of human improvisation. One of the biggest challenges lies in capturing the complexity of a unique piece and understanding the intricate connections between its various parts. Furthermore, while AI advances have enabled orchestral music generation, human composers continue to play a crucial role in arranging and adapting these results to ensure they achieve the desired emotional depth and coherence.


----------
Thank you for reading my article! Please don't hesitate to reach out if you have any questions or suggestions.

Everything written here represents my personal views and (likely incomplete) knowledge. It does not reflect the opinions of anyone involved in the creation of the algorithms discussed. If you are one of those individuals and would like me to make any changes—whether by adding or removing content—please feel free to contact me, and I'd be happy to accommodate your request.

*Note: Many of the new algorithms are open-access on GitHub, so don’t hesitate to grab them and experiment on your own!*