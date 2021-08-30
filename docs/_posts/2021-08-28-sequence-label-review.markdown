---
layout: post
title: "Note on sequence labeling models"
date: 2021-08-28 16:28:00 +0800
categories: math
---

This note is a summary of the review article [A Survey on Recent Advances in Sequence Labeling from Deep Learning Models](https://arxiv.org/abs/2011.06727) by Zhiyong He et al.

## 1. Background

### 1.1 Classical Sequence Labeling

- Part-of-speed Tagging (POS): a.k.a grammatical tagging, is a process of assigning tags like noun (NN), verb (VB), adjective (JJ), etc, to words of sentences. There are different tag systems and [Penn Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) (PTB) is one of the most widely used tag systems. For example, the sentence "Mr. Jones is an editor of the journal" will be tagged as "NNP NNP VBZ DT NN IN DT NN" in TPB.
- Named Entity Recognition (NER): a.k.a entity identification or entity chunking, is a task to identify named entities from text. Three major of entity categories: entity, time, and numeric. Label of a word in NER is comprised of two parts as "X-Y", where X indicates the position of the labeled word and Y indicates the category. One widely-adopted tag system is BIOES, that is, the word labeled “B” (Begin), “I” (Inside) and “E” (End) means that it is the first, middle or last word of a named entity phrase, respectively. The word labeled “0-” (Outside) means it
does not belong to any named entity phrase and “S-” (Single) indicates it is the only word that represent an entity.
- Text Chunking: divide text into syntactically related non-overlapping groups of words. The sentence "The little dog barked at the cat." can be divided into three phrases as “(The little dog) (barked at) (the cat)”. In BIOES tagging system, the corresponding tags for the sentence are "B-NP I-NP E-NP B-VP E-VP B-NP E-NP". 


### 1.2 Deep Learning based models
Deep learning based models typically consists of three modules by functionality: embedding module, context enoder module, and inference module. 

## 2. Embedding module

### 2.1 Pretrained word embedding
Embedding that generate a single context-independent vector for each word, ignoring the modeling of polysemy problem:
- [Word2Vec](https://arxiv.org/abs/1301.3781) includes two architectures, i.e., continuous bag-of-word and skip-gram. 
- [SENNA](https://arxiv.org/abs/1103.0398) word embedding 
- [GloVe](https://nlp.stanford.edu/projects/glove/) embedding by Stanford

Contextual word representations, that is, representation of each word is dependent on its context. For example, the word “present” in “How many people were present at the meeting?” is different from that in “I’m not at all satisfied with the present situation”.

- [Peters et al.](https://arxiv.org/abs/1705.00108) proposed pretrained contextual embeddings from bidirectional language models. They extend the method to [ELMo](https://arxiv.org/abs/1802.05365) (Embeddings from Language Models) representations.
- [BERT](https://arxiv.org/abs/1810.04805) by Devlin et al.
- [He et al.](https://ojs.aaai.org/index.php/AAAI/article/view/6299) proposed an embedding that is both context-aware and knowledge-aware, which encode the prior knowledge of entities from an external knowledge base.



### 2.2 Character-level representation
Character-level representation can capture the word morphological and shape information which is normally ignored by pretrained word embeddings. 

The two most common architectures to capture character-to-word representations are Convolutional Neural Networks(CNNs) and Recurrent Neural Networks(RNNs).

Embedding models based on CNN architectures:

- [Santos and Zadrozny](http://proceedings.mlr.press/v32/santos14.pdf) proposed a CNN-based approach to learn character-level representations of words.

| ![cnn-character-level-representation](/assets/images/cnn-charecter-level-feature-extraction.PNG){:class="img-responsive"} | 
|:--:| 
| The architecture of CNN-based character-level representation |

- [Xin et al.](https://arxiv.org/abs/1810.12443) propose IntNet, a funnel-shaped wide convolutional neural network for learning character-level representations for sequence labeling.

| ![InetNet-architecture](/assets/images/IntNet-architecture.PNG){:class="img-responsive"} | 
|:--:| 
| The architecture of IntNet |

Embedding models based on RNN architectures:
- [Ling et al.](https://arxiv.org/abs/1508.02096): a compositional character to word (C2W) model that uses bidirectional LSTMs (Bi-LSTM) to build word embeddings by taking the characters as atomic units.

| ![lexical-composition-model](/assets/images/lexical-composition-model.PNG){:class="img-responsive"} | 
|:--:| 
| The architecture of lexical Composition model |

- [Dozat et al.](https://nlp.stanford.edu/pubs/dozat2017stanford.pdf): a RNN based character-level model in which the character embeddings sequence of each word is fed into a unidirectional LSTM followed by an attention mechanism.
- [Kann et al.](https://aclanthology.org/W18-3401/): a character-based recurrent sequence-to-sequence architecture, connects the Bi-LSTM character encoding model to a LSTM based decoder that associated with an auxiliary objective
- [Bohnet et al.](https://aclanthology.org/P18-1246/): a sentence-level character model for learning context sensitive character-based representations of words.


### 2.3 Hand-crafted features

- [Collobert et al.](https://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf) (2011): utilize word suffix, gazetteer and capitalization features as well as cascading features that include tags from related tasks.
- [Wu et al.](https://arxiv.org/abs/1808.09075) (2018): a hybrid neural model which combines a feature auto-encoder loss component to utilize hand-crafted features, and significantly outperforms existing competitive models on the task of NER.
- [Rijhwani et al.](https://aclanthology.org/D11-1141/) (2011): a method of ‘”soft gazetteers” that incorporates information from English knowledge bases through cross-lingual entity linking and create continuousvalued gazetteer features for low-resource languages.
- [Ghaddar et al.](https://arxiv.org/abs/1806.03489) (2018): a novel lexical representation (called Lexical Similarity i.e., (LS) vector) for NER, indicating that robust lexical features are quiet useful and can greatly benefit deep neural network architectures.

### 2.4 Sentence-level Representation

- [Yijin Liu et al.](https://arxiv.org/abs/1906.02437) (2019)
- [Ying Luo et al.](https://ojs.aaai.org/index.php/AAAI/article/view/6363) (2020)

## 3. Context Encoder Module
There are three commonly used model architectures for context encoder module, i.e., RNN, CNN and Transformers.

### 3.1 RNN
- [Huang et al.](https://arxiv.org/abs/1508.01991) (2015): Bi-LSTM to generate contextual representations of every word in their sequence labeling model.
- [Rei](https://arxiv.org/abs/1704.07156) (2017): a multitask learning method that equips the Bi-LSTM context encoder module with a auxiliary training objective, which learns to predict surrounding words for every word in the sentence.
- [Zhang et al.](https://arxiv.org/abs/1711.08231) (2017): a new method called Multi-Order BiLSTM which combines low order and high order LSTMs together in order to learn more tag dependencies.
- [Ma et al.](https://arxiv.org/abs/1709.10191) (2017): a LSTM-based model for jointly training sentence-level classification and sequence labeling tasks, in which a modified LSTM structure is adopted as their context encoder module.
- [Gregoric et al.](https://aclanthology.org/P18-2012/) (2018): a different architecture which employs multiple parallel independent Bi-LSTM units across the same input and promotes diversity among them by employing an inter-model regularization term.
- [Kazi et al.](https://aclanthology.org/P17-2027/) (2017): a novel implicitly-defined neural network architecture by changing implicit hidden layer as:
{% raw %}
\begin{equation}
h_t = f(\xi_{t}, h_{t-1}, h_{t+1})
\end{equation}
{% endraw %}

- [Liu et al.](https://arxiv.org/abs/1906.02437) (2019): Deep transition RNNs which extends conventional RNNs by increasing the transition depth of consecutive hidden states
- [Wei et al.](https://ui.adsabs.harvard.edu/abs/2021PatRe.11007636W/abstract) (2021): employs self-attention to provide complementary context information on the basis of Bi-LSTM.

### 3.2 CNN
- An initial work in this area is proposed by [Collobert et al.](https://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf) (2011). [Santos et al.](http://proceedings.mlr.press/v32/santos14.pdf) (2014) follow their work and use similar structure for context feature extraction.
- [Shen et al.](https://arxiv.org/abs/1707.05928) (2017): a deep active learning based model for NER tasks.
- [Wang et al.](https://www.semanticscholar.org/paper/Named-Entity-Recognition-with-Gated-Convolutional-Wang-Chen/f35ada86f4f1e4e6c5e8aaf57d538a4e1d0584c5) (2017): stacked Gated Convolutional Neural Networks(GCNN) for named entity recognition, which extend the convolutional layer with gating mechanism. A gated convolutional layer can be written as 

{% raw %}
\begin{equation}
F_{\mathrm{gating}}(\mathbf{X)} = (\mathbf{X} * \mathbf{W} + \hat{b})\odot \sigma(\mathbf{X} * \mathbf{V} + \hat{c})
\end{equation}
{% endraw %}
where $*$ denotes row convolution, $\sigma$ is the sigmoid function and represents element-wise product, $\mathbf{X}$ is the input of this layer and the rests are parameters to be learned.

- [Strubell et al.](https://arxiv.org/abs/1702.02098) (2017): Iterated Dilated Convolutional Neural Networks (ID-CNNs) method for the task of NER. [Dilated convolutions](https://arxiv.org/abs/1511.07122) (2015) operate on a sliding window of context like typical CNN layers, but the context need not be consecutive.

| ![dilated-cnn](/assets/images/dilated-CNN.PNG){:class="img-responsive"} | 
|:--:| 
| A dilated CNN block with maximum dilation width 4 and filter width 3. Neurons contributing to a single highlighted neuron in the last layer are also highlighted |

### 3.3 Transformer
The Transformer is widely used in various NLP tasks and has achieved excellent results. However, in sequence labeling tasks, the Transformer encoder has been reported to perform poorly. Both the direction and relative distance information are important in the NER, but these information will lose when the sinusoidal position embedding is used in the vanilla Transformer. [Yan et al.](https://arxiv.org/abs/1911.04474) (2019) proposed TENER to address the issue.

## 4. Inference Module
