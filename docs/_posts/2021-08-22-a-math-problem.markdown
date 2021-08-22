---
layout: post
title: "一道有趣的数学题"
date: 2021-08-22 16:28:00 +0800
categories: math
---

前段时间解决了一道有趣的数学题。题目如下：$a=(a_1,a_2),b=(b_1,b_2)$为$\mathbb{R}^2$空间的向量，定义两者的标积为
{% raw %}
$$
a \cdot b = a_1 b_2 - a_2 b_1
$$
{% endraw %}
注意，这里的标积是反对称的。求解关于函数$f$的方程
{% raw %}
$$
-f(a+b,c,d) e^{a\cdot b} + f(a,b+c,d) e^{b\cdot c} - f(a,b,c+d) e^{c\cdot d} + f(a+d,b,c) e^{-a\cdot d} = 0
$$
{% endraw %}
$a,b,c,d$为$\mathbb{R}^2$上的任意向量，$f(a,b,c)$是关于$a,b,c$的标积的函数，并且可以在原点处泰勒展开。 这个方程存在一组非0的平凡解（相对容易），还有一个非平凡解（较难）。

先放块砖，待续...