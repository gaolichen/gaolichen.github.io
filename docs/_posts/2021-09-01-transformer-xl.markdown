---
layout: post
title: "Transformer-XL模型"
date: 2021-09-01 16:28:00 +0800
categories: NLP
---

BERT模型只能接受长度固定的输入，通常最大的长度为512，这使得它不适合处理超长文本输入。处理文本的第513个token时，BERT不得不丢掉第1个token的信息，因此无法获得完整的上下文。[Transformer-XL](https://arxiv.org/abs/1901.02860)模型改进了这个缺点。Transformer-XL在BERT基础上做了两点优化。

## 1. 重现机制 (Recurrence Mechanism)

假设BERT的最长输入长度为4，那么BERT的对文本的处理如下所示：

| ![vanila-bert-train](/assets/images/vanila-bert-train.PNG){:class="img-responsive"} | 
|:--:| 
|BERT训练阶段|

| ![vanila-bert-train](/assets/images/vanila-bert-eval.PNG){:class="img-responsive"} | 
|:--:| 
|BERT评估阶段|

训练阶段，每个片段(segment)最多包含4个token，注意力机制只作用于同一个片段的4个token，所以$(x_5, x_6, x_7, x_8)$不能获得$(x_1,x_2,x_3,x_4)$的上下文信息。而在评估（evaluation）阶段，每个token只能看到它前面的3个token的信息， $x_5$看不到$x_1$，$x_6$看不到$x_1, x_2$，以此类推。

Transformer-XL引入了重现机制解决这个问题。每个片段的输入长度仍然固定，新增一个相同长度的缓存区保存历史信息（数据），使得处理当前片段时，注意力机制可以“看到”缓存区里的历史信息。下图虚线方框内为缓存区：

| ![vanila-bert-train](/assets/images/tr-xl-train.PNG){:class="img-responsive"} | 
|:--:| 
|Transformer-XL训练阶段|

处理当前片段时，缓存区的数据固定不变，不参与backpropagation的梯度计算。处理下一个片段时，当前片段产生的数据，变成缓存区的数据，以此类推，不断迭代，直到长文本处理完毕。


| ![vanila-bert-train](/assets/images/tr-xl-evaluation.PNG){:class="img-responsive"} | 
|:--:| 
|Transformer-XL评估阶段|

重现机制的数学表达式如下。假设$s_{\tau}, s_{\tau + 1}$为两个连续的片段，$s_{\tau} = \[x_{\tau, 1},...,x_{\tau, L}\]$，$s_{\tau + 1} = \[x_{\tau + 1, 1},...,x_{\tau + 1, L}\]$，L为片段的长度。定义片段$\tau$的第$n$个隐藏层输出为$h ^n _{\tau} \in \mathbb{R}^{L\times d}$，其中$d$为隐藏层的维度。那么，重现机制可表示为：

![tr-xl-equations](/assets/images/tr-xl-equations.PNG){:class="img-responsive"}

其中$\mathrm{SG}(\cdot)$表示stop-gradient，$\[h_u \circ h_v\]$表示两个隐藏序列的拼接。

## 2. 相对位置编码 (Relative Positional Encodings)

BERT模型的embedding层的输出既包含内容的embedding（word embedding）也包含位置编码（position encoding）。位置编码可以用矩阵$\mathbf{U}\in \mathbb{R}^{L_\mathrm{max}\times d}$表示。第$i$含元素$\mathbf{U}_i$表示第i位置的编码。
