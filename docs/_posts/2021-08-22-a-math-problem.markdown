---
layout: post
title: "一道有趣的数学题（一）"
date: 2021-08-22 16:28:00 +0800
categories: math
---

前段时间解决了一道有趣的数学题。题目如下：$a=(a_1,a_2),b=(b_1,b_2)$为$\mathbb{R}^2$空间的向量，定义两者的标积为
{% raw %}
\begin{equation}
a \cdot b = a_1 b_2 - a_2 b_1
\label{eq:dot_prod}
\end{equation}
{% endraw %}
注意，这里的标积是反对称的。求解关于函数$f$的方程
{% raw %}
\begin{equation}
-f(a+b,c,d) e^{a\cdot b} + f(a,b+c,d) e^{b\cdot c} - f(a,b,c+d) e^{c\cdot d} + f(a+d,b,c) e^{d\cdot a} = 0
\label{eq:equation}
\end{equation}
{% endraw %}
$a,b,c,d$为$\mathbb{R}^2$上的任意向量，$f(a,b,c)$是关于$a,b,c$的标积的函数，并且可以在原点处泰勒展开。 这个方程存在一组非0的平凡解（相对容易），还有一个非平凡解（较难）。

<br/>

## 平凡解
平凡解可以用类似“待定系数法”来求解：先猜测解的形式，然后带入方程求解“系数”。方程(\ref{eq:equation})的左边含有诸如$e^{a \cdot b}$的指数函数，因此我们很自然的猜想解$f(a,b,c)$也含有指数函数。最简单的指数函数为$e^{a\cdot b}, e^{b\cdot c}, e^{c\cdot a}$，所以不妨假设解的形式为：
{% raw %}
\begin{equation}
f(a,b,c) = g_1 (a,b,c) e^{a \cdot b} - g_2 (a,b,c) e^{b \cdot c} + g_3 (a,b,c) e^{c \cdot a}
\label{eq:ansatz}
\end{equation}
{% endraw %}

$g_1, g_2, g_3$为三个任意函数。


将(\ref{eq:ansatz})带入方程的左边。如果全部展开，结果会比较繁琐，为简单起见，对方程(\ref{eq:equation})的左边一项一项的处理，结果如下：

{% raw %}
$$
-f(a+b,c,d) e^{a\cdot b} = -g_1 (a+b,c,d) \color{blue}{e^{a \cdot c + b \cdot c + a \cdot b} } + g_2 (a+b,c,d) \color{green}{e^{c \cdot d + a\cdot b}} - g_3 (a+b,c,d) \color{red}{e^{d \cdot a + d \cdot b + a \cdot b}} 
$$

$$
f(a,b+c,d) e^{b\cdot c} = g_1 (a,b+c,d) \color{blue}{e^{a \cdot b + a \cdot c + b \cdot c} } - g_2 (a,b+c,d) \color{cyan}{e^{b \cdot d + c\cdot d + b \cdot c}} + g_3 (a,b+c,d) \color{purple}{e^{d \cdot a + b \cdot c}}
$$

$$
-f(a,b,c+d) e^{c\cdot d} = -g_1 (a,b,c+d) \color{green}{e^{a \cdot b + c \cdot d} } + g_2 (a,b,c+d) \color{cyan}{e^{b \cdot c + b\cdot d + c \cdot d}} - g_3 (a,b,c+d) \color{orange}{e^{c \cdot a + d \cdot a + c\cdot d}}
$$

$$
f(a+d,b,c) e^{d\cdot a} = g_1 (a+d,b,c) \color{red}{e^{a \cdot b + d \cdot b + d\cdot a} } - g_2 (a+d,b,c) \color{purple}{e^{b \cdot c + d\cdot a}} + g_3 (a+d,b,c) \color{orange}{e^{c \cdot a + c \cdot d + d\cdot a}}
$$
{% endraw %}

以上表达式中，相同的指数部分对应相同的颜色。将以上四个表达式相加，按照指数部分合并同类项：
{% raw %}
$$
\begin{aligned}
e^{a\cdot c + b\cdot c+ a\cdot b} \text{ 系数} &= -g_1 (a+b,c,d) + g_1(a,b+c,d) \\
e^{c\cdot d + a\cdot b} \text{ 系数} &= g_2 (a+b,c,d) - g_1(a,b,c+d) \\
e^{d\cdot a + d\cdot b + a \cdot b} \text{ 系数} &= -g_3 (a+b,c,d) + g_1(a+d,b,c) \\
e^{b\cdot d + c\cdot d + b \cdot c} \text{ 系数} &= -g_2 (a,b+c,d) + g_2(a,b,c+d) \\
e^{d\cdot a + b\cdot c} \text{ 系数} &= g_3 (a,b+c,d) - g_2(a+d,b,c) \\
e^{c\cdot a + d\cdot a + c \cdot d} \text{ 系数} &= -g_3 (a,b,c+d) + g_3(a+d,b,c)
\end{aligned}
$$
{% endraw %}

因为$a,b,c,d$是任意的向量，要使方程成立，以上每一个指数函数的系数应该都为0。这里有6个方程，而我们的未知数只有3个，一般情况下是无解的。通过观察，我们发现如果$g_1, g_2, g_3$为以下表达式，这6个方程都能恰好满足，
{% raw %}
$$
\begin{aligned}
g_1(a,b,c) &= g (a+b,c) \\
g_2(a,b,c) &= g (a,b+c) \\
g_3(a,b,c) &= g (a+c,b)
\end{aligned}
$$
{% endraw %}
其中$g(a,b)$为任意以$a\cdot b$为自变量的函数。所以方程的解为

{% raw %}
\begin{equation}
f(a,b,c) = g (a+b,c) e^{a \cdot b} - g (a,b+c) e^{b \cdot c} + g (a+c,b) e^{c \cdot a}
\label{eq:solution}
\end{equation}
{% endraw %}

以上就是求解平凡解的过程。任何可以表示为(\ref{eq:solution})形式的解都为平凡解。

可能有些人会问，如果我们仍用“待定系数”法，但使用与(\ref{eq:ansatz})不同的解的形式，会不会得到另外的解呢？比如让(\ref{eq:ansatz})中的每一项指数都为$a\cdot b, b\cdot c, c\cdot a$的线性组合，会怎么样呢？我们可能会得到看似不一样的结果，但其实都是(\ref{eq:solution})的特殊形式或者等价形式。虽然我无法穷尽所有的解的形式，但一个大胆的猜想如下：任何令方程指数项“恰好”消除的解，都是形如(\ref{eq:solution})的解。

其实，我们求解平凡解的过程只用了标积(\ref{eq:dot_prod})的一个性质，分配律：
{% raw %}
$$
\begin{aligned}
a \cdot (b+c) &= a\cdot b + a\cdot c \\
(a+b) \cdot c &= a\cdot c + b \cdot c
\end{aligned}
$$
{% endraw %}

平凡解的求解过程不需要标积的具体表达式，也不需要“向量在二维空间”这个条件，甚至不需要标积“反对称”这个性质。所以，这个解被称为“平凡解”也就不难理解了。而只有将这些性质用起来，我们才能获得非平凡解。