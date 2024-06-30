This repository serves as the implementation space for my ongoing Bachelor's Thesis

Title:
"Exploring the Hidden Structures of Attention
Layers in Transformer Models through the Lens
of Gaussian Distributions"
Martin Volker Wertich
July 01, 2024
Version: 1.0


Abstract:
Understanding the internal dynamics of Transformer models is challenging. This work aims to provide insights into why this task is so formidable. 
We theoretically analyze the cornerstone of the surge in Large Language Models:
The attention mechanism, which adds an additional layer of complexity to an already opaque black-box model. Gladly, the embedding of human language provides us with sufficient mathematical geometrical structure, which we approximate with Gaussian distributions throughout this work.

In simple terms, two core components of utilizing Transformer models remain largely unintelligible to humans: the mathematical structure of the data and that of the learned weight matrices. 
We try to combine them in the context of an attention layer by intertwining Linear Algebra, Multivariate Statistics, Information Theory, and Random Matrix Theory.
A key takeaway from this work is that the concept of ’attention’ in Large Language Models can be easily underestimated. This technique is not just a simple token-matching function; rather, it serves as a sophisticated combiner
of marginal probability distributions influenced by their mutual dependencies, allowing the model to manage complex linear combinations through a bilinear form internally. 
Grasping the mathematical foundations behind attention is a significant step toward comprehending the functioning of Large Language Models.

This work is both relatively theoretical and slightly inconvenient in its structure, stemming from the complexity of this emerging field and the lack of consistent formalism. 
It finds its place in the branch of Natural Language Processing Interpretability, which is yet to grow and manifest as a crucial pillar of Deep Learning Theory.

