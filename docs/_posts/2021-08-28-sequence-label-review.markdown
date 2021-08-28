---
layout: post
title: "Note on sequence labeling models"
date: 2021-08-22 16:28:00 +0800
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

## 4. Inference Module
