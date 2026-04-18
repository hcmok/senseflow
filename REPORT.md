## Exploratory Study of Flow-guided Semantic Morphing

$$\text{sense} \rightarrow\ \text{vibe} \rightarrow\ \text{rhythm} \rightarrow\ \text{flow} $$
$$\text{war} \rightarrow\ \text{hostilities} \rightarrow\ \text{truce} \rightarrow\ \text{peace}$$
$$\text{code} \rightarrow\ \text{character} \rightarrow\ \text{text} \rightarrow\ \text{prose} \rightarrow\ \text{poetry}$$

### Abstract

This is an exploratory study on flow-guided semantic morphing in the word embedding space. A vector field is learned from optimal transport-induced displacements between word pairs using Riemannian Conditional Flow Matching. As a geometric prior, this field guides an A\* search algorithm to generate a coherent intermediate word sequence connecting two given words. The results suggest that flow guidance can improve search efficiency and uncover semantically meaningful paths in the embedding space.

### 1. Introduction

Word embeddings are representations of words as continuous vectors in a high-dimensional space. Early techniques such as Word2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) learn static word embeddings that capture the semantics of words. Cosine similarity, which depends on the angle between two word vectors, is a widely used metric of semantic similarity for words. BERT (Devlin et al., 2018) utilizes the transformer architecture and bidirectional context to learn contextual word embeddings, which are more effective in capturing the meanings of words in specific contexts.

Semantically morphing one word into another requires generating a sequence of intermediate words whose meanings shift gradually and meaningfully. An interpretable morphing algorithm is desirable since it can reveal the semantic structure encoded in word embeddings trained on large corpora.

### 2. Related Work

The idea of capturing semantic relationships as vectors was popularized by models like GloVe. A well-known example is the analogy **king - man + woman ≈ queen**. It illustrates how linear operations on word vectors can encode semantic relationships. It motivates the possibility of utilizing the geometric structure of the word embedding space to create word transitions.

Zeldes (2018) used Word2Vec embeddings with an A\* search algorithm to find sequences of intermediate words between two given words. At the sentence level, Huang et al. (2018) proposed _Text Morphing_, which generates a sequence of intermediate sentences between two given sentences. Their approach iteratively generates an "editing vector" that encodes the necessary word deletions, additions, and contextual history (via hidden states). The vector is then fed to a decoder to produce the next sentence in the sequence.

### 3. Methods

#### 3.1 Word Representations

Some words exhibit polysemy (e.g., "chip", "apple"). Representing each word with a single static vector may blur word semantics and interfere with word morphing. To address this, we choose BERT embeddings as word representations, which are context-aware and can handle polysemy. We curate a vocabulary by selecting the most frequent words from a portion of the English subset of the allenai/c4 corpus. For each word, we extract its BERT embeddings from multiple sample sentences in the same corpus.

To enhance computational efficiency in downstream tasks, the raw BERT embeddings are z-score normalized and then reduced in dimensionality by incremental principal component analysis (PCA).

Next, we compute word centroids by applying K-Means clustering to the reduced word embeddings. Polysemous words typically produce multiple centroids, each corresponding to a distinct sense. The centroids are again z-score normalized and serve as the final representations of word senses.

In all downstream operations, both word embeddings and centroids are further $L_2$-normalized, such that the word representations lie on the unit hypersphere:
$$S^{d-1} = \{x \in \mathbb{R}^d : \|x\|_2 = 1\}$$

This defines the geometric domain for our Riemannian vector field and enables interpolation and geodesic pathfinding in the embedding space.

#### 3.2 Optimal Transport

To construct training targets for the vector field, we compute optimal transport (OT) displacements between semantically related word pairs. We use WordNet (Princeton University, 2010) to identify candidate pairs, including synonyms, antonyms, hypernyms, etc. For each pair, we select the most semantically aligned sense pair by comparing the cosine similarities of their centroids. The corresponding BERT embeddings serve as the source and target distributions in the OT problem. Prior to transport, the embeddings undergo z-score normalization followed by $L_2$ normalization before transport. This respects the geometry of the embedding space as the unit hypersphere $\mathbb{S}^{d-1}$. For any two $L_2$-normalized vectors $x_i$ and $y_j$, the squared Euclidean distance is monotonically related to cosine similarity:
$$C_{ij} = \|x_i - y_j\|^2_2 = \|x_i\|^2 + \|y_j\|^2 - 2(x_i \cdot y_j) = 2 - 2cos(\theta)$$
For each pair of distributions, we solve the entropy-regularized optimal transport problem:
$$P^\ast = \text{argmin}_{P \in U(\mu, \nu)} \sum_{i,j} P_{ij} C_{ij} - \epsilon H(P)$$
The destination is obtained via barycentric projection:
$$\tilde{x}_i = \frac{\sum_{j} P_{ij} y_j}{\left\| \sum_{j} P_{ij} y_j \right\|_2}$$
The raw displacement is then:
$$disp_i = \tilde{x}_i  - x_i$$
Since we operate on the sphere, this straight-line displacement must later be projected onto the tangent space to obtain the correct Riemannian geodesic (see Section 3.3).

#### 3.3 Riemannian Conditional Flow Matching

We employ Riemannian Conditional Flow Matching (RCFM) (Chen & Lipman, 2024) to learn a smooth vector field from the OT displacements.

First, each raw displacement is projected onto the tangent plane at the source point $x$:
$$v_{\text{tangent}} = disp - \langle disp, x \rangle x$$
Then, the Riemannian exponential map on the unit sphere reconstructs the endpoint $y$:
$$y = \exp_x(v_{\text{tangent}}) = \cos(\|v_{\text{tangent}}\|_2) \, x + \sin(\|v_{\text{tangent}}\|_2) \frac{v_{\text{tangent}}}{\|v_{\text{tangent}}\|_2}$$

During training, we sample $t \sim \mathcal{U}[0,1]$ and obtain an intermediate point $z_t$ along the geodesic by spherical linear interpolation (SLERP):
$$z_t = \gamma(t) = \operatorname{slerp}(x,y;t) = \frac{sin((1-t)\theta)}{sin\theta}x + \frac{sin(t\theta)}{sin\theta}y$$
where $$\theta = \text{arccos}(\langle x, y \rangle)$$

The target velocity at time $t$ is the time derivative of the geodesic:
$$v_{\text{target}}(t) = \dot{\gamma}(t) = \frac{-\theta \cos((1-t)\theta)}{\sin \theta} x + \frac{\theta \cos(t\theta)}{\sin \theta} y$$

Instead of conditioning on the exact OT destination $y$, we condition the model on the word centroid $g$. This provides a more stable target that guides the flow toward meaningful regions of the manifold.

The loss function is:
$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1], (x,y,g) \sim q} \left[ \lambda_{\text{mse}} \| v_\theta(z_t, g, t) - v_{target}(t) \|_g^2 + \lambda_{\text{ortho}} \langle v_\theta(z_t, g, t), z_t \rangle^2 \right]$$
where $||\cdot||_g^2$ is the squared norm induced by the Riemannian metric, which reduces to the standard Euclidean $L_2$ norm on the tangent space of the hypersphere.
The orthogonality term encourages the predicted velocity to remain on the tangent plane at $z_t$.

#### 3.4 Pathfinding with Flow Guidance

When morphing between two words, we first resolve polysemy by selecting the pair of senses (centroids) with the smallest Euclidean distance. We then employ an A\* search algorithm over the set of all word-sense centroids to find a path. The learned Riemannian vector field guides the search. To accelerate neighbor discovery, a k-d tree over centroids is precomputed. The A\* cost function minimizes:
$$f(n) = g(n) + h(n)$$
where
$$g(n) = g_{\text{parent}} + \lambda_{\text{step}} \|z_{n}-z_{\text{parent}}\|_2 + \lambda_{\text{lemma}} l_c(n) + \underbrace{\lambda_{\text{flow}}\left(1- \left\langle \frac{z_{n}-z_{\text{parent}}}{\|z_{n}-z_{\text{parent}}\|_2}, \frac{v_{\text{pred}}}{\|v_{\text{pred}}\|_2} \right\rangle\right)}_{\text{alignment}}$$
$$h(n) = \lambda_{\text{step}}\|z_{n}-z_{\text{goal}}\|_2(1+\text{alignment})$$

- $z_n$: The centroid of the word sense $n$.
- $l_c(n)$: The lemma repetition cost.
  - $l_c(n) = 1$: If $n$ shares a root lemma (noun, verb, or adjective) with any prior word in the path. Different senses of the same word are not counted as repetitions.
  - $l_c(n) = 0$: Otherwise.
- $v_{\text{pred}}$: The predicted velocity at $z_{\text{parent}}$ by the RCFM model.

Due to the lemma and flow cost terms, the A\* heuristic is not strictly admissible. The flow-alignment weighting in $h(n)$ discourages paths that deviate from the learned vector field even when the Euclidean distance to the goal is small.

### 4. Experiments

For experimentation, we retained the first 100 principal components from PCA to represent the word embeddings. They explained 48.8% of the variance in the original data. Retaining a higher number of principal components may be considered in the future. The full configuration can be found in the [config file](./configs/config.yaml).

We used word pairs from SimLex-999 (Hill et al., 2015) to evaluate the performance of our semantic morphing model, and compared it with a vanilla A\* baseline without flow guidance. We employed stratified sampling to select an equal number of word pairs from three subsets defined by their SimLex-999 scores. The results are shown below. The experiment configuration can be found in the [experiment config file](./configs/experiment_config.yaml).

|                                                              | A\* with flow guidance | A\* vanilla |
| ------------------------------------------------------------ | ---------------------- | ----------- |
| Avg. word cosine similarity within path                      | 0.50                   | 0.51        |
| Avg. path length                                             | 3.81                   | 3.77        |
| Avg. node expansion                                          | 44.01                  | 128.77      |
| Avg. node expansion (words with SimLex999 score 0≤x<3.33)    | 68.78                  | 204.71      |
| Avg. node expansion (words with SimLex999 score 3.33≤x<6.66) | 6.28                   | 15.38       |
| Avg. node expansion (words with SimLex999 score 6.66≤x≤10)   | 3.64                   | 9.30        |
| % of paths reaching target                                   | 100.00                 | 99.33       |
| % of paths with intermediate words                           | 58.67                  | 58.00       |

With flow guidance, the average cosine similarity between consecutive words in a path remained nearly equivalent to the vanilla A\* baseline, while requiring significantly fewer node expansions and achieving faster search. The guided approach successfully reached the target word in 100% of cases within the expansion limit, whereas the vanilla A\* failed in 0.67% of cases.

Notably, many tested word pairs are semantically related, allowing both methods to often reach the target directly. No minimum number of intermediate steps was enforced, as this better reflects the intrinsic structure of the word embedding space.

We also evaluated linear and spherical interpolation methods. Neither approach produced valid paths with intermediate words for any of the tested word pairs, likely due to the sparsity of the high-dimensional embedding space.

The generated results can be found in the [results folder](./results/).

_Note: These results correspond to the data and repository code at the time of writing._

### 5. Conclusion

In this study, we explore flow-guided semantic morphing in a high-dimensional word embedding space. We demonstrate that integrating a vector field learned via Riemannian Conditional Flow Matching (RCFM) into the A\* search cost improves search efficiency. The model effectively navigates the sparse linguistic manifold, where standard geometric interpolation methods fail.

The generated word paths are not only semantically meaningful and coherent, but also exhibit a degree of creativity. This approach has potential applications in brainstorming tools. Furthermore, modeling word embeddings as a Riemannian vector field may facilitate further study of semantic structure in embedding spaces.

### 6. Appendix

#### 6.1 Examples of Generated Word Paths

They are generated using the default configuration, except that the results are deterministic.

- Association
  - king → ruler → queen
  - sunny → humid → windy
  - bread → toast → spoon → cup → coffee
  - animal → mammal → mammalian → human
  - person → man → businessman → company
- Contrast
  - bright → light → dark
  - cold → warm → hot
  - happy → glad → sad
  - strong → stiff → weak
  - lose → spend → play → win
  - war → hostilities → truce → peace
  - ice → avalanche → lava
  - mayor → superintendent → warden → prisoner → slave
- Geographics
  - tunnel → passageway → waterway → sea
  - forest → wilderness → mountainous → arid → desert
  - city → municipality → village
  - america → east → asia
- Transformation
  - ant → ape → elephant
  - leaf → foliage → canopy → jungle
  - disturbance → damage → catastrophe
  - tree → garland → ribbon → paper
  - imagine → think → have → possess
  - birth → fetal → premature → mortality → death
- Abstract
  - wave → cyclone → hydra → magma → atom
  - freedom → immunity → imprisonment → prison
  - justice → fairness → quality → level → scale
  - truth → reality → myth → illusion
  - romance → attraction → warmth → heat → energy → power
  - tale → thriller → horror → nightmare
  - sense → vibe → rhythm → flow
- Weak association
  - tea → cups → spheres → worlds → universe
  - history → biography → profile → face → nose
  - drink → bottle → label → category → adjective → verb
  - python → extension → feature → dock_0 → dock_1 → pond
- Polysemy
  - chip → processor → computer
  - chip → bits → sweets → chocolate
  - window → browser → surf
  - rass → grassy → sandy → beach → surf

### 7. References

Chen, R. T. Q., & Lipman, Y. (2024). Flow matching on general geometries. In *The Twelfth International Conference on Learning Representations*. https://openreview.net/forum?id=g7ohDlTITL

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4171-4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423

Hill, F., Reichart, R., & Korhonen, A. (2015). SimLex-999: Evaluating semantic models with (genuine) similarity estimation. *Computational Linguistics*, 41(4), 665–695. https://doi.org/10.1162/coli_a_00237

Huang, S., Wu, Y., Wei, F., & Zhou, M. (2018). Text morphing [Preprint]. *arXiv*. https://arxiv.org/abs/1810.00341

Mikolov, T., Chen, K., Corrado, G.S., & Dean, J. (2013). Efficient estimation of word representations in vector space. *International Conference on Learning Representations*. https://arxiv.org/abs/1301.3781

Pennington, J., Socher, R., & Manning, C. D. (2014, October). GloVe: Global vectors for word representation. In *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)* (pp. 1532-1543). Association for Computational Linguistics. https://doi.org/10.3115/v1/D14-1162

Princeton University. (2010). *About WordNet*. WordNet. https://wordnet.princeton.edu

Zeldes, Y. (2018, April 28). *Word morphing*. *Towards Data Science*. https://towardsdatascience.com/word-morphing-9f87ee577775
