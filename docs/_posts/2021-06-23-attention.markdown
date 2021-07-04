---
layout: post
title: "自然语言模型中的Attention机制"
date: 2021-06-23 16:28:00 +0800
categories: NPL
---

很多自然语言模型都基于Transformer架构，比如BERT, GPT等。而Attention机制是Transformer架构重要的组成部分。那什么是Attention机制呢？以下用Scaled Dot-Product Attention为例讨论。

# Scaled Dot-Product Attention
Attention可以被看成是一个映射或者函数。输入参数为querys，keys, 和values，每个参数都是一组向量，可以用是二维矩阵表示，矩阵的每行就是一个向量；输出也是用二维矩阵表示的一组向量。用$Q, K, V$来代表这三个矩阵，$Q_i, K_i, V_i$表示这些矩阵第$i$行的向量。$Q$和$K$的向量维度相同，都为$d_k$，$V$的向量维度为$d_v$。$(K_i, V_i)$是一组key-value pair，因此$K$和$V$有相同个数的向量。Attention机制可以表示为：
{% raw %}
$$
A \equiv \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{Q K^{T}}{\sqrt{d_k}}\right)V
$$
{% endraw %}

若$Q, K, V$有以下的维度（形状）
```python
  Q.shape: (query_len, d_k)
  K.shape: (sequence_len, d_k)
  V.shape: (sequence_len, d_v)
```
则$A$的维度为(query_len, d_v)。所以，Attention的结果可以理解为，对每一个query，返回一个向量$V_i$的线性组合。softmax($\cdot$) 相当于一个线性变换$S$，作用于以$V_i$为基的向量空间：
{% raw %}
$$
A_i = \sum_j S_{ij}  V_j,\quad S_{ij} = \frac{\exp(Q_i \cdot K_j)}{\sum_l \exp(Q_i \cdot K_l)}
$$
{% endraw %}

$A_i$中$V_j$的贡献由$Q_i$和$K_j$的点积决定，$Q_i$和$K_j$的关联越大，$V_j$对$A_i$的贡献越大。

# Attention-Mask
Attention-Mask的作用是过滤$A_i$中某些特定的$V_j$。Attention-Mask可以用一个形状为(query_len, sequence_len)的矩阵M来实现，通过将$M_{ij}$的值设为很大的负数，我们可以实现这样的过滤机制。数学表达式为：

{% raw %}
$$
A_i = \sum_j S_{ij}  V_j,\quad S_{ij} = \frac{\exp(Q_i \cdot K_j + M_{ij})}{\sum_l \exp(Q_i \cdot K_l + + M_{il})}
$$
{% endraw %}


# Multi-Head Attention

单个Attention机制没有可以学习的参数，对给定的输入，其输出结果是不变的。使用Multi-Head Attention可以实现学习的机制。

{% raw %}
$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)W^O \\
\quad \mathrm{head}_{i} = \mathrm{Attention}(Q W^{Q}_{i}, K W^{K}_{i}, V W^{V}_i)
$$
{% endraw %}

其中，$W^Q_i$和$W^K_i$为$d_{\mathrm{model}} \times d_k$的矩阵参数，$W^V_i$为$d_{\mathrm{model}} \times d_k$的矩阵参数。$W^O$为$h d_v \times d_{\mathrm{model}}$的矩阵参数。在BERT的实现中，$h=8$，$d_k = d_v = d_{\mathrm{model}}/h = 64$。

# Self-Attention

在Transformer模型中，长度为sequence_len的输入文本通过Embedding层后，输出形状为(sequence_len, $d_{\mathrm{model}}$)的矩阵$X$，每个词对应于一个$d_{\mathrm{model}}$维的向量。在调用Multi-Head Attention时，将Q, K, V参数都设为$X$，$\mathrm{MultiHead}(X, X, X)$。这就是Self-Attention。

在BERT中，除了Self-Attention过程，词和词之间是没有耦合的，Self-Attention可以理解为学习文本上下文的机制。