---
title: "Eigenvalues and Eigenvectors"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
---

Notes on eigenvalues, eigenspaces, generalized eigenspaces, and the ladder of
normal forms that sits on top of them: upper-triangular $\to$ primary
decomposition $\to$ Jordan $\to$ diagonal.

## Notation and Standing Assumptions

Unless stated otherwise, $T \in \mathcal{L}(V)$. Results that genuinely need
$\mathbf{F} = \mathbb{C}$ say so explicitly.

{% include notation-table.md keys="F V n LV PF eigenspace gen_eigenspace geo_mult alg_mult spectrum min_poly min_poly_exp jordan_block partition conj_partition index_nilpotency" %}

## Invariant Subspaces

{: .prompt-info }
> Sums and arbitrary intersections of $T$-invariant subspaces are $T$-invariant.

{: .prompt-info }
> A $T$-invariant subspace is invariant under $p(T)$ for every $p \in \mathcal{P}(\mathbf{F})$.

{: .prompt-proof }
> The operators leaving the subspace invariant are closed under sums, products and scalar multiples, and contain $I$ and $T$, so they contain every $p(T)$. $\blacksquare$

{: .prompt-info }
> $\operatorname{null}p(T)$ and $\operatorname{range}p(T)$ are $T$-invariant for every $p \in \mathcal{P}(\mathbf{F})$. In particular $E(\lambda,T)$ and $G(\lambda,T)$ are $T$-invariant.

{: .prompt-proof }
> $p(T)$ commutes with $T$. If $p(T)v = 0$ then $p(T)(Tv) = T\,p(T)v = 0$; and $T\,p(T)v = p(T)(Tv) \in \operatorname{range}p(T)$. The same argument works for any operator commuting with $T$; see [Commuting Operators]({% post_url 2026-08-01-commuting-operators %}). $\blacksquare$

{: .prompt-info }
> **Invariant direct sums.** Suppose $V = V_1 \oplus \dots \oplus V_m$ with each $V_k$ invariant under $T$, and write $T_k = \left.T\right\rvert_{V_k}$. Then for every $p \in \mathcal{P}(\mathbf{F})$:
>
> (a) $\operatorname{null}p(T) = \bigoplus_k \operatorname{null}p(T_k)$ and $\operatorname{range}p(T) = \bigoplus_k \operatorname{range}p(T_k)$.
>
> (b) The minimal polynomial of $T$ is the lcm of those of the $T_k$.

{: .prompt-proof }
> Each $V_k$ is invariant under $p(T)$ by the box above, and $\left.p(T)\right\rvert_{V_k} = p(T_k)$. So writing $v = \sum_k v_k$ with $v_k \in V_k$, we get $p(T)v = \sum_k p(T_k)v_k$ with $p(T_k)v_k \in V_k$. By directness, $p(T)v = 0$ iff every $p(T_k)v_k = 0$, which is the null-space formula, and the range formula is read off the same expression. For (b), by (a) $p(T) = 0$ iff every $p(T_k) = 0$, i.e. iff the minimal polynomial of every $T_k$ divides $p$, i.e. iff their lcm divides $p$. $\blacksquare$

{: .prompt-info }
> Every subspace of an eigenspace $E(\lambda,T)$ is invariant, since $T$ acts on it as $\lambda I$.

### Every $k$-Dimensional Subspace Invariant

The converse of the last box: if enough subspaces are invariant, $T$ is a scalar. A $1$-dimensional invariant subspace is exactly the span of an eigenvector, so for $k = 1$ the hypothesis says every nonzero vector is an eigenvector.

{: .prompt-info }
> Suppose $ k \in \\{ 1, \dots, n - 1 \\} $ and every subspace of $ V $ of dimension $ k $ is invariant under $ T $. Then $ T $ is a scalar multiple of the identity operator.

{: .prompt-proof }
> **Every nonzero $v$ is an eigenvector.** Suppose not: $v, Tv$ are independent for some $v$. Extend them to a basis $v, Tv, x_3, \dots, x_n$. Since $k + 1 \le n$, the subspace
>
> $$U = \operatorname{span}(v,\ x_3,\ \dots,\ x_{k+1})$$
>
> (just $\operatorname{span}(v)$ when $k = 1$) has dimension $k$ and contains $v$, but not $Tv$. This contradicts the invariance of $U$.
>
> **All the eigenvalues agree.** Write $Tu = \lambda_u u$ for nonzero $u$. If $w = cu$, then $Tw = c\lambda_u u = \lambda_u w$, so $\lambda_w = \lambda_u$. If $u, w$ are independent, then
>
> $$\lambda_{u+w} u + \lambda_{u+w} w = T(u+w) = \lambda_u u + \lambda_w w$$
>
> forces $\lambda_u = \lambda_{u+w} = \lambda_w$. So $T = \lambda I$ for a single $\lambda$. $\blacksquare$

## Null Space and Range Chains

{: .prompt-info }
> Suppose $m$ is a nonnegative integer. Then
>
> $$ \operatorname{null} T^m = \operatorname{null} T^{m + 1} \iff \operatorname{range} T^m = \operatorname{range} T^{m + 1} $$
>
> and once either holds at $m$, it holds at every $m' \ge m$.

{: .prompt-tip }
> So the null-space chain and the range chain stabilize at the *same* step; this common step is the **index** of $T$ (in operator theory, the *ascent* and *descent*, which are equal here). Before it, $\dim\operatorname{null}T^m$ grows by at least $1$ at each step, so the chains have stabilized by $m = n$. This is why $G(\lambda,T) = \operatorname{null}(T-\lambda I)^n$ is a safe uniform definition. The true index of $T - \lambda I$ is usually much smaller, namely $e_\lambda$ (proved in [The Nilpotent Part](#the-nilpotent-part) below).

{: .prompt-info }
> For every $k \ge 1$,
>
> $$V = \operatorname{null}T^k \oplus \operatorname{range}T^k \iff k \ge \text{the index of } T.$$
>
> In particular $V = \operatorname{null}T^n \oplus \operatorname{range}T^n$ always, and $V = \operatorname{null}T \oplus \operatorname{range}T$ iff the index is at most $1$.

{: .prompt-proof }
> First, for any operator $S$: $V = \operatorname{null}S \oplus \operatorname{range}S \iff \operatorname{null}S = \operatorname{null}S^2$.
>
> - *($\Rightarrow$)* If $v \in \operatorname{null}S^2$, then $Sv \in \operatorname{null}S \cap \operatorname{range}S = \\{0\\}$, so $v \in \operatorname{null}S$.
> - *($\Leftarrow$)* If $Su \in \operatorname{null}S$, then $u \in \operatorname{null}S^2 = \operatorname{null}S$, so $Su = 0$. Hence $\operatorname{null}S \cap \operatorname{range}S = \\{0\\}$, and the dimensions add up to $n$ by rank–nullity.
>
> Apply this to $S = T^k$. Since $\operatorname{null}T^k \subseteq \operatorname{null}T^{k+1} \subseteq \operatorname{null}T^{2k}$, the condition $\operatorname{null}T^k = \operatorname{null}T^{2k}$ holds iff the chain has stopped growing by step $k$, i.e. $k \ge$ the index. The index is at most $n$. $\blacksquare$

## Generalized Eigenspaces

Suppose the distinct eigenvalues of $T$ are $ \lambda_1, \dots, \lambda_m $.

|                                  | $G(\lambda_k,T)$                                                                           | $E(\lambda_k,T)$                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Definition                       | $\operatorname{null}(T-\lambda_k I)^{j}$, any $j \ge e_{\lambda_k}$ ($j = n$ always works) | $\operatorname{null}(T-\lambda_k I)$                                        |
| Containment                      | $E \subseteq G$                                                                            | —                                                                           |
| Restriction of $T - \lambda_k I$ | nilpotent, index exactly $e_{\lambda_k}$                                                   | zero                                                                        |
| Sum is direct                    | always                                                                                     | always                                                                      |
| Sum is all of $V$                | iff the minimal polynomial splits over $\mathbf{F}$ (automatic over $\mathbb{C}$)          | iff $T$ diagonalizable                                                      |
| Multiplicity                     | algebraic $d_{\lambda_k} = \dim G$                                                         | geometric $g_{\lambda_k} = \dim E$, $1 \le g_{\lambda_k} \le d_{\lambda_k}$ |
| Min. polynomial                  | $\prod_k (z-\lambda_k)^{e_{\lambda_k}}$                                                    | $\prod_k(z-\lambda_k)$ **iff diagonalizable**                               |
| Char. polynomial                 | $\prod_k (z-\lambda_k)^{d_{\lambda_k}}$                                                    | — (always uses $d_{\lambda_k}$)                                             |

{: .prompt-info }
> An eigenvalue $\lambda$ of $T$ is called **defective** when $\dim E(\lambda, T) \;<\; \dim G(\lambda, T)$, i.e. $g_\lambda < d_\lambda$.

{: .prompt-tip }
> Directness of the sum of eigenspaces says eigenvectors with distinct eigenvalues are linearly independent. Conversely, every linearly independent list $v_1, \dots, v_m$ arises this way: extend it to a basis $v_1, \dots, v_n$ and define $T$ by $Tv_k = kv_k$ for $k \le m$ and $Tv_k = 0$ for $k > m$.

{: .prompt-tip }
> **Blockwise computation.** For $V = V_1 \oplus \dots \oplus V_m$ with each $V_k$ invariant, [invariant direct sums](#invariant-subspaces) with $p = (z - \lambda)^j$ give $\operatorname{null}(T - \lambda I)^j = \bigoplus_k \operatorname{null}(T_k - \lambda I)^j$ for every $j$. Taking $j = 1$ and $j = n$ (the chains of $T_k - \lambda I$ stabilize by $\dim V_k \le n$): $E(\lambda,T) = \bigoplus_k E(\lambda,T_k)$ and $G(\lambda,T) = \bigoplus_k G(\lambda,T_k)$. So $g_\lambda$ and $d_\lambda$ add across the blocks, the index of $T - \lambda I$ is the max of the blockwise ones (so $e_\lambda = \max_k e_\lambda^{(k)}$, matching the lcm), and the characteristic polynomial is the **product** of the blocks'. This is the engine behind both the primary decomposition and the Jordan form: choose the $V_k$ well and each block becomes trivial to read.

### The Nilpotent Part

Every eigenvalue produces a nilpotent operator. If $\lambda$ is an eigenvalue of $T$, then $G(\lambda,T)$ is [invariant](#invariant-subspaces) under $T$, and

$$N := \left.(T - \lambda I)\right\rvert_{G(\lambda,T)}$$

is nilpotent. So everything in [Nilpotent Operators]({% post_url 2026-07-25-nilpotent-operators %}) applies to this $N$, acting on the space $G(\lambda,T)$ of dimension $d_\lambda$: it has a Jordan basis and a diagram, with partition $\mu = (m_1 \ge \dots \ge m_k)$, conjugate partition $\mu'$, index of nilpotency $p = m_1$, and $d_j = \dim\operatorname{null}N^j$. The parts $m_i$ are the sizes of the Jordan blocks of $T$ for $\lambda$.

{: .prompt-warning }
> Two notations meet here. The algebraic multiplicity $d_\lambda = \dim G(\lambda,T)$ is indexed by an eigenvalue; $d_j = \dim\operatorname{null}N^j$ is indexed by a power. They are compatible: $d_j = d_\lambda$ for every $j \ge p$.

{: .prompt-tip }
> Since $\left.T\right\rvert_{G(\lambda,T)} = \lambda I + N$, for every scalar $\alpha \ne \lambda$
>
> $$\left.(T - \alpha I)\right\rvert_{G(\lambda,T)} = (\lambda - \alpha)I + N$$
>
> is invertible, by [Inverse of a Jordan Block]({% post_url 2026-07-25-nilpotent-operators %}#inverse-of-a-jordan-block). So $\lambda$ is the only eigenvalue of $\left.T\right\rvert_{G(\lambda,T)}$, and taking $\alpha = 0$, $\left.T\right\rvert_{G(\lambda,T)}$ is invertible when $\lambda \ne 0$.

At the other extreme, $T$ is nilpotent iff $G(0,T) = \operatorname{null}T^n = V$, since a nilpotent operator satisfies $T^n = 0$.

The three numbers attached to $\lambda$ are read off the diagram of $N$:

|                                   | reads off as                     | on the diagram               |
| --------------------------------- | -------------------------------- | ---------------------------- |
| $d_\lambda$ (char. poly exponent) | $\dim G(\lambda,T) = \sum_i m_i$ | total boxes (area)           |
| $e_\lambda$ (min. poly exponent)  | $p = m_1$                        | height of the tallest column |
| $g_\lambda$                       | $d_1 = \mu'_1 = k$               | size of the bottom level     |

The first row holds because the boxes form a basis of $G(\lambda,T)$, and the third because $g_\lambda = \dim\operatorname{null}N$, since $E(\lambda,T) \subseteq G(\lambda,T)$. By rank–nullity on $N$,

$$\dim\operatorname{range}N = d_\lambda - g_\lambda,$$

the number of boxes above the bottom level. The second row needs a proof.

{: .prompt-info }
> The [index](#null-space-and-range-chains) of $T - \lambda I$, the index of nilpotency $p$ of $N$, and the exponent $e_\lambda$ of $z - \lambda$ in the minimal polynomial of $T$ are all equal.

{: .prompt-proof }
> **Index $= p$.** Since $\operatorname{null}(T - \lambda I)^j \subseteq G(\lambda,T)$ for every $j$, these null spaces are exactly the $\operatorname{null}N^j$, which grow strictly until $j = p$ and are constant after.
>
> **$e_\lambda = p$.** Write $S = T - \lambda I$, $G = G(\lambda,T) = \operatorname{null}S^n$, $R = \operatorname{range}S^n$, and $p_T(z) = (z-\lambda)^{e_\lambda}q(z)$ with $q(\lambda) \ne 0$. Then $V = G \oplus R$ by the [null–range decomposition](#null-space-and-range-chains), both summands are invariant under every polynomial in $T$, and $S\vert_G = N$.
>
> *$p \le e_\lambda$.* Since $(T\vert_G - \lambda I)^p = N^p = 0$, the minimal polynomial of $T\vert_G$ divides $(z-\lambda)^p$, so it is $(z-\lambda)^j$ for some $j \le p$. Then $N^j = (T\vert_G - \lambda I)^j = 0$, which forces $j = p$. So the minimal polynomial of $T\vert_G$ is $(z-\lambda)^p$, and it [divides $p_T$]({% post_url 2026-07-22-polynomials %}#homomorphism-gives-divisibility-injectivity-gives-equality). Write $(z-\lambda)^{e_\lambda}q = h\,(z-\lambda)^p$. If $p > e_\lambda$, cancelling $(z-\lambda)^{e_\lambda}$ gives $q = h\,(z-\lambda)^{p-e_\lambda}$, so $q(\lambda) = 0$, a contradiction. Hence $p \le e_\lambda$.
>
> *$e_\lambda \le p$.* $S\vert_R$ is injective, because $\operatorname{null}S \subseteq G$ and $G \cap R = \\{0\\}$, so it is invertible. Restricting $p_T(T) = 0$ to $R$ gives $(S\vert_R)^{e_\lambda}\,q(T)\vert_R = 0$, so $q(T)\vert_R = 0$. Now $f = (z-\lambda)^pq$ satisfies $f(T)\vert_G = q(T)\vert_G\,N^p = 0$ and $f(T)\vert_R = (S\vert_R)^p\,q(T)\vert_R = 0$, so $f(T) = 0$ on $V = G \oplus R$. The minimal polynomial divides every polynomial that annihilates $T$, so $f = h\,p_T$ for some polynomial $h$, that is, $(z-\lambda)^pq = h\,(z-\lambda)^{e_\lambda}q$. Cancelling $q$ and comparing degrees gives $p = \deg h + e_\lambda \ge e_\lambda$. $\blacksquare$

So $d_\lambda$, $e_\lambda$, $g_\lambda$ are $n$, $p$, $k$ for $N$, and all relations among them are [those of $n$, $p$, $k$]({% post_url 2026-07-25-nilpotent-operators %}#jordan-basis): $1 \le g_\lambda, e_\lambda \le d_\lambda$ and $g_\lambda + e_\lambda - 1 \le d_\lambda$, with no inequality between $g_\lambda$ and $e_\lambda$ alone.

{: .prompt-info }
> *The nilpotent shadow of diagonalizability.* $\lambda$ is not defective ($g_\lambda = d_\lambda$, i.e. $k = n$ for $N$) iff $N$ is the [one-level extreme]({% post_url 2026-07-25-nilpotent-operators %}#one-level-the-zero-operator) $N = 0$, i.e. $G(\lambda,T) = E(\lambda,T)$ and $e_\lambda = 1$. By the [null–range decomposition](#null-space-and-range-chains) this says $V = \operatorname{null}(T - \lambda I) \oplus \operatorname{range}(T - \lambda I)$, which holds trivially when $\lambda \notin \sigma(T)$. So over $\mathbb{C}$,
>
> $$T \text{ is diagonalizable} \iff V = \operatorname{null}(T - \lambda I) \oplus \operatorname{range}(T - \lambda I) \text{ for every } \lambda \in \mathbb{C}.$$
>
> Over $\mathbb{R}$ the right side also holds when $\sigma(T) = \varnothing$ (see [existence](#existence-of-eigenvalues)), so it no longer implies diagonalizability.

### Existence of Eigenvalues

Everything here rests on one fact: $\sigma(T)$ is exactly the set of zeros of $p_T$ in $\mathbf{F}$. So whether $T$ has eigenvalues depends on how $p_T$ factors over $\mathbf{F}$.

{: .prompt-info }
> (a) Over $\mathbb{C}$, $\sigma(T) \ne \varnothing$.
>
> (b) Over $\mathbb{R}$, $\sigma(T) = \varnothing$ exactly when $p_T$ has no real zero, as for rotation by $\pi/2$ on $\mathbb{R}^2$, with $p_T = z^2 + 1$. This cannot happen when $n$ is odd.
>
> (c) Over $\mathbb{R}$, $T$ always has an invariant subspace of dimension $1$ or $2$.

{: .prompt-proof }
> (a) $p_T$ has degree at least $1$ since $V \ne \\{0\\}$, so it has a complex zero.
>
> (b) The first claim is the fact above. For odd $n$, see [Complexification]({% post_url 2026-08-03-complexification %}#application-odd-dimension-forces-a-real-eigenvalue).
>
> (c) Let $q$ be an irreducible factor of $p_T$, so $\deg q \in \\{1, 2\\}$, and write $p_T = qh$. By minimality $h(T) \ne 0$, so $w = h(T)v \ne 0$ for some $v$, and $q(T)w = p_T(T)v = 0$. Then $T^{\deg q}w \in \operatorname{span}(w, \dots, T^{\deg q - 1}w)$, so this span is invariant, nonzero, and of dimension at most $\deg q \le 2$. $\blacksquare$

{: .prompt-tip }
> Over $\mathbb{R}$ the role of an eigenvector is played by an invariant plane on which $T$ acts as a rotation-scaling with a conjugate pair of eigenvalues of $T_{\mathbb{C}}$. This is why the hierarchy of normal forms below assumes the minimal polynomial splits: without it, even the upper-triangular form fails.

### Counting Eigenvalues

{: .prompt-info }
> Let $r = \dim\operatorname{range}T$.
>
> (a) $\sum_{\lambda \in \sigma(T)} d_\lambda \le n$, with equality iff the minimal polynomial of $T$ splits over $\mathbf{F}$ (automatic over $\mathbb{C}$). Over $\mathbb{R}$, $\sum_{\lambda \in \sigma(T)} d_\lambda \equiv n \pmod 2$.
>
> (b) $\sum_{\lambda \in \sigma(T)} g_\lambda \le \sum_{\lambda \in \sigma(T)} d_\lambda$, so $\sum_{\lambda \in \sigma(T)} g_\lambda \le n$, with equality iff $T$ is diagonalizable.
>
> (c) $\sum_{\lambda \in \sigma(T) \setminus \\{0\\}} d_\lambda \le n - d_0 \le r$, where $d_0 = \dim G(0,T)$ (which is $0$ when $0$ is not an eigenvalue).
>
> (d) $\lvert \sigma(T) \rvert \le \min(n,\; 1 + r)$, and over $\mathbb{C}$ also $\lvert \sigma(T) \rvert \ge 1$ by [existence](#existence-of-eigenvalues).

{: .prompt-proof }
> (a) The sum of the $G(\lambda,T)$ is direct and lies in $V$, and it is all of $V$ iff the minimal polynomial [splits](https://en.wikipedia.org/wiki/Splitting_field). Over $\mathbb{C}$ the equality is also visible in an [upper-triangular matrix]({% post_url 2026-07-04-quotient-spaces %}#induction-on-dimension) of $T$, whose $n$ diagonal entries list each eigenvalue $\lambda$ exactly $d_\lambda$ times. Over $\mathbb{R}$, the sum is the part of $n = \sum d_\lambda(T_{\mathbb{C}})$ coming from real eigenvalues; the non-real eigenvalues of $T_{\mathbb{C}}$ come in conjugate pairs with equal multiplicities ([Complexification]({% post_url 2026-08-03-complexification %})), so they contribute an even number.
>
> (b) $E(\lambda,T) \subseteq G(\lambda,T)$ gives $g_\lambda \le d_\lambda$. The sum of the $E(\lambda,T)$ is direct, and it is all of $V$ iff $T$ is diagonalizable.
>
> (c) For $\lambda \ne 0$, $\left.T\right\rvert_{G(\lambda,T)}$ is invertible (see [The Nilpotent Part](#the-nilpotent-part)), so $G(\lambda,T) = T^n\big(G(\lambda,T)\big) \subseteq \operatorname{range}T^n$. Hence $\bigoplus_{\lambda \ne 0} G(\lambda,T) \subseteq \operatorname{range}T^n$. Since $V = \operatorname{null}T^n \oplus \operatorname{range}T^n = G(0,T) \oplus \operatorname{range}T^n$ by the [null–range decomposition](#null-space-and-range-chains), we get $\dim\operatorname{range}T^n = n - d_0$, and $\operatorname{range}T^n \subseteq \operatorname{range}T$.
>
> (d) Each eigenvalue has $d_\lambda \ge 1$, so $\lvert \sigma(T) \rvert \le \sum_{\lambda \in \sigma(T)} d_\lambda \le n$ by (a), and $\lvert \sigma(T) \setminus \\{0\\} \rvert \le \sum_{\lambda \in \sigma(T) \setminus \\{0\\}} d_\lambda \le r$ by (c). The value $0$ adds at most one more. $\blacksquare$

{: .prompt-tip }
> Adding $d_0$ to both sides of (c) gives back (a). So once repetitions are counted, the range bound is never sharper than $n$. The improvement $1 + r$ in (d) comes entirely from $0$ contributing a single distinct value however large $d_0$ is.

{: .prompt-tip }
> **Which bound in (d) applies, and when it is attained** (matrices in some basis).
>
> - **$T$ invertible** ($r = n$): the bound is $n$, attained iff $T = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ with the $\lambda_i$ distinct.
> - **$T$ not invertible** ($r < n$, so $1 + r \le n$): the bound is $1 + r$, attained iff $T = \operatorname{diag}(0, \dots, 0, \lambda_1, \dots, \lambda_r)$ with the $\lambda_i$ distinct and nonzero.

{: .prompt-tip }
> **The gap in $1 + r$.** Over $\mathbb{C}$, when $0$ is an eigenvalue, rank–nullity gives $r = n - g_0$, and $\sum_\lambda d_\lambda = n$, so the gap between the bound and the true count is
>
> $$(1 + r) - \lvert \sigma(T) \rvert = (d_0 - g_0) + \sum_{\lambda \in \sigma(T) \setminus \\{0\\}} (d_\lambda - 1).$$
>
> Each term is a defect, so equality holds exactly when $0$ is not defective and every nonzero eigenvalue is simple, which is the diagonal form above. For nilpotent $T$ the gap is $d_0 - g_0 = n - k$, the number of boxes above the bottom level: $n - 1$ for a single column $J_n(0)$ and $0$ for a single level $T = 0$, the [two extremes]({% post_url 2026-07-25-nilpotent-operators %}#two-extremes) of the diagram.

## Eigenspaces of $p(T)$

{: .prompt-info }
> Suppose $ \mathbf{F} = \mathbb{C}$, $ p \in \mathcal{P}(\mathbb{C}) $, and $ \alpha \in \mathbb{C} $. Then
>
> (a) $G(\alpha,\, p(T)) \;=\; \bigoplus_{\lambda:\, p(\lambda) = \alpha} G(\lambda,\, T).$
>
> (b) $E(\alpha, p(T)) \;\supseteq\; \bigoplus_{\lambda:\,p(\lambda)=\alpha} E(\lambda, T)$, with equality iff $p'(\lambda) \neq 0$ for every defective $\lambda$ in that fiber.

{: .prompt-proof }
> Over $\mathbb{C}$, $V = \bigoplus_{\lambda \in \sigma(T)} G(\lambda,T)$ with each summand $T$-invariant; write $T_\lambda = \left.T\right\rvert_{G(\lambda,T)}$. For each $j$, $(p(T) - \alpha I)^j$ is $q(T)$ with $q = (p - \alpha)^j$, so [invariant direct sums](#invariant-subspaces) gives
>
> $$\operatorname{null}\big(p(T) - \alpha I\big)^j = \bigoplus_{\lambda \in \sigma(T)} \operatorname{null}\big(p(T_\lambda) - \alpha I\big)^j.$$
>
> On each block, $T_\lambda = \lambda I + N$ with $N$ nilpotent. Since $\lambda I$ and $N$ commute, the algebraic Taylor expansion of $p$ around $\lambda$ holds:
>
> $$p(T_\lambda) \;=\; \sum_{t=0}^{\deg p} \frac{p^{(t)}(\lambda)}{t!}\, N^t \;=\; p(\lambda)\,I \;+\; \underbrace{\Big(p'(\lambda)N + \tfrac{p''(\lambda)}{2}N^2 + \dots\Big)}_{=:\,M}.$$
>
> $M$ is a polynomial in $N$ with **zero constant term**, so it is nilpotent. Hence $p(T_\lambda) - \alpha I = (p(\lambda) - \alpha)I + M$ is invertible when $p(\lambda) \ne \alpha$, and equals $M$ when $p(\lambda) = \alpha$. So only the blocks with $p(\lambda) = \alpha$ contribute, and they contribute $\operatorname{null}M^j$.
>
> (a) Take $j = n$: $M^n = 0$, so each such block contributes all of $G(\lambda,T)$.
>
> (b) Take $j = 1$: each such block contributes $\operatorname{null}M$, while $E(\lambda, T) = \operatorname{null}N$. Factor $M = NQ$ with
>
> $$Q = p'(\lambda) I + \tfrac{p''(\lambda)}{2}N + \dots.$$
>
> Two cases:
>
> - **If $p'(\lambda) \neq 0$**, then $Q$ is (nonzero scalar) $+$ (nilpotent), hence invertible, so $\operatorname{null}M = \operatorname{null}N$.
> - **If $p'(\lambda) = 0$**, then $M$ starts at $N^2$ or higher, so $\operatorname{null}M \supseteq \operatorname{null}N^2$; and $\operatorname{null}N^2 \supsetneq \operatorname{null}N$ exactly when $N \ne 0$, i.e. when $\lambda$ is defective.
>
> So the ordinary eigenspace grows strictly exactly when some $\lambda$ in the fiber is **both a critical point of $p$ and a defective eigenvalue of $T$**. $\blacksquare$

## The Hierarchy of Normal Forms

| $T \in \mathcal{L}(V)$ | Basis                                                                                                                                                  | Subspaces                                                                  | Dimensions                                                                                                                                                         | Minimal polynomial                                                                       | Nullspace and range                                                                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Upper-triangularizable | $Tv_k \in \operatorname{span}(v_1,\dots,v_k)$ for each $k$                                                                                             | $\operatorname{span}(v_1,\dots,v_k)$ invariant for each $k$                | $\lambda_k$ occurs $\dim G(\lambda_k,T)$ times on the diagonal                                                                                                     | $(z-\lambda_1)\cdots(z-\lambda_m)$, $\lambda_i \in \mathbf{F}$, repetitions allowed      | —                                                                                                                                                                              |
| Lower-triangularizable | $Tv_k \in \operatorname{span}(v_k,\dots,v_n)$ for each $k$                                                                                             | $\operatorname{span}(v_k,\dots,v_n)$ invariant for each $k$                | same (reverse the basis)                                                                                                                                           | same                                                                                     | —                                                                                                                                                                              |
| Primary decomposition  | basis is a concatenation of bases of the $G(\lambda_k,T)$; inside group $k$, $(T-\lambda_k I)v \in \operatorname{span}$(earlier vectors of that group) | $V = G(\lambda_1,T)\oplus\dots\oplus G(\lambda_m,T)$, $\lambda_k$ distinct | $\sum_k \dim G(\lambda_k,T) = \dim V$; block $k$ has size $d_{\lambda_k}$                                                                                          | $\prod_k (z-\lambda_k)^{e_{\lambda_k}}$, $\lambda_k$ **distinct**, $e_{\lambda_k} \ge 1$ | $G(\lambda,T) = \operatorname{null}(T-\lambda I)^{\dim V}$                                                                                                                     |
| Jordan form            | basis is a disjoint union of chains $v,\,(T{-}\lambda)v,\dots,(T{-}\lambda)^{s-1}v$; equivalently $Tv_k - \lambda v_k \in \\{0,\,v_{k-1}\\}$           | $V = $ direct sum of **indecomposable** invariant subspaces, one per chain | number of blocks for $\lambda$: $\dim E(\lambda,T)$; blocks of size $\ge j$: $\dim\operatorname{null}(T{-}\lambda)^j - \dim\operatorname{null}(T{-}\lambda)^{j-1}$ | $\prod_k (z-\lambda_k)^{e_{\lambda_k}}$, exponent $=$ largest block size                 | $\operatorname{rank}(T-\lambda)^j$ for all $j$ determines the entire diagram                                                                                                   |
| Diagonalizable         | a basis of eigenvectors of $T$                                                                                                                         | $V = E(\lambda_1,T)\oplus\dots\oplus E(\lambda_m,T)$                       | $\sum_k \dim E(\lambda_k,T) = \dim V$                                                                                                                              | $(z-\lambda_1)\cdots(z-\lambda_m)$, $\lambda_k$ **distinct**                             | $V = \operatorname{null}(T{-}\lambda I) \oplus \operatorname{range}(T{-}\lambda I)$ **for every** $\lambda \in \mathbf{F}$ (a characterization when $\mathbf{F} = \mathbb{C}$) |

![Hierarchy of upper-triangular, primary decomposition, and Jordan form](/assets/img/math/upper_triangular_primary_jordan_hierarchy.png)
_Each step refines the previous one._

{: .prompt-tip }
> Each refinement buys finer blocks and pays for it in uniqueness. The $G(\lambda_k,T)$ are determined by $T$ alone, with no choices. The individual Jordan blocks inside a $G$ are not: with $\mu = (2,1)$ there's a whole family of valid choices for the size-2 and size-1 summands, and only the multiset $\\{2,1\\}$ is forced.

![Matrix shapes versus operator classes](/assets/img/math/matrix_shapes_vs_operator_classes.png)
_Matrix shapes against the operator classes that realize them._

## Norms and Eigenvalues

### Size of Eigenvalues

{: .prompt-info }
> Suppose $ \lambda $ is an eigenvalue of $ T $, $ v_1, \dots, v_n $ is any basis of $ V $, and $ A = \mathcal{M}(T, (v_1,\dots,v_n)) $. Then
>
> $$ \lvert \lambda \rvert \;\le\; \max_j \sum_k \lvert A_{j,k} \rvert \;\le\; n \max_{j,k} \lvert A_{j,k} \rvert. $$

{: .prompt-proof }
> Let $x$ be the coordinate vector of an eigenvector for $\lambda$, and pick $j$ with $\lvert x_j \rvert$ maximal, so $x_j \ne 0$. Row $j$ of $Ax = \lambda x$ reads $\lambda x_j = \sum_k A_{j,k}x_k$, so
>
> $$\lvert \lambda \rvert\,\lvert x_j \rvert \le \sum_k \lvert A_{j,k} \rvert\,\lvert x_k \rvert \le \Big(\sum_k \lvert A_{j,k} \rvert\Big)\lvert x_j \rvert.$$
>
> Dividing by $\lvert x_j \rvert$ gives the first bound, and each of the $n$ terms is at most $\max_{j,k}\lvert A_{j,k}\rvert$. $\blacksquare$

{: .prompt-tip }
> The middle quantity is the operator norm of $A$ induced by the max-norm on coordinates, so this is the special case "$\lvert\lambda\rvert \le \lVert T \rVert$ for every induced norm". Running the same argument only on the off-diagonal terms gives the [Gershgorin circle theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem): $\lambda$ lies in some disc $\lvert z - A_{j,j}\rvert \le \sum_{k \ne j}\lvert A_{j,k}\rvert$. The left side does not depend on the basis, so the bound can be minimized over all bases. And since nearby operators have bounded entries, their eigenvalues stay bounded, which is the first step toward eigenvalues depending continuously on $T$.

### Density of Diagonalizable Operators

{: .prompt-info }
> A subset $S \subseteq \mathcal{L}(V)$ is **dense** if for every $ T \in \mathcal{L}(V) $ and every $ \varepsilon > 0 $ there is $ D \in S $ with $ \lVert T - D \rVert < \varepsilon $.

{: .prompt-tip }
> All norms on the finite-dimensional space $\mathcal{L}(V)$ are equivalent, so density does not depend on which norm is chosen.

{: .prompt-info }
> Over $\mathbb{C}$, the diagonalizable operators are dense in $\mathcal{L}(V)$.

{: .prompt-proof }
> Let $T \in \mathcal{L}(V)$. Since $\mathbf{F} = \mathbb{C}$, there is a basis $v_1,\dots,v_n$ with respect to which $\mathcal{M}(T)$ is upper triangular, with diagonal entries $\lambda_1,\dots,\lambda_n$. Given $\varepsilon > 0$, choose $\varepsilon_1,\dots,\varepsilon_n \in \mathbb{C}$ with $\lvert \varepsilon_j \rvert < \varepsilon$ such that $\lambda_1 + \varepsilon_1, \dots, \lambda_n + \varepsilon_n$ are pairwise distinct. This is always possible, since each $\varepsilon_j$ needs only to avoid finitely many values, and any disc is infinite.
>
> Define the perturbation $P \in \mathcal{L}(V)$ by $Pv_j = \varepsilon_j v_j$, and set $D = T + P$. Then $\mathcal{M}(D)$ is upper triangular with the distinct entries $\lambda_j + \varepsilon_j$ on the diagonal. The diagonal of a triangular matrix lists the eigenvalues, so $D$ has $n$ distinct eigenvalues in a space of dimension $n$, hence is diagonalizable. And $D$ is close to $T$: in the operator norm induced by the inner product making $v_1,\dots,v_n$ orthonormal,
>
> $$\lVert T - D \rVert = \lVert P \rVert = \max_j \lvert \varepsilon_j \rvert < \varepsilon. \qquad \blacksquare$$

{: .prompt-warning }
> Density fails over $\mathbb{R}$: a real operator with non-real eigenvalues (e.g. a rotation by $\pi/2$ on $\mathbb{R}^2$) has a whole neighbourhood of operators with no real eigenvalues at all, none of which is diagonalizable over $\mathbb{R}$.
