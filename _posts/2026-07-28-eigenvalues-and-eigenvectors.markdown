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

Unless stated otherwise: $\mathbf{F}$ is $\mathbb{R}$ or $\mathbb{C}$, $V$ is a
**nonzero finite-dimensional** vector space over $\mathbf{F}$ with
$n = \dim V$, and $T \in \mathcal{L}(V)$. Results that genuinely need
$\mathbf{F} = \mathbb{C}$ say so explicitly.

{% include notation-table.md keys="LV PF eigenspace gen_eigenspace geo_mult alg_mult min_poly min_poly_exp jordan_block partition conj_partition index_nilpotency dual_op dual_space" %}

## Invariant Subspaces

{: .prompt-info }
> Suppose $ U $ is a subspace of $ V $ invariant under $ T $. Then $ U $ is invariant under $ p(T) $ for every polynomial $ p \in \mathcal{P}(\mathbf{F}) $.

{: .prompt-info }
> The null space and range of $p(T)$ are invariant under $T$, for every $p \in \mathcal{P}(\mathbf{F})$.

{: .prompt-info }
> Every subspace of an eigenspace is invariant.

{: .prompt-info }
> An arbitrary intersection of $T$-invariant subspaces is $T$-invariant.

### Every Line Invariant

A $1$-dimensional invariant subspace is exactly the span of an eigenvector, so
"every line is invariant" means "every nonzero vector is an eigenvector".

{: .prompt-info }
> Suppose every nonzero vector in $ V $ is an eigenvector of $ T $. Then $ T $ is a scalar multiple of the identity operator.

{: .prompt-proof }
> Write $\lambda_u$ for the eigenvalue attached to a nonzero $u$, so $Tu = \lambda_u u$. Fix nonzero $u, w \in V$.
>
> If $w = cu$ for some $c \ne 0$, then $\lambda_w w = Tw = c\,Tu = c\lambda_u u = \lambda_u w$, so $\lambda_w = \lambda_u$.
>
> If $u, w$ are independent, then $u + w \ne 0$ and
>
> $$\lambda_{u+w} u + \lambda_{u+w} w = T(u+w) = \lambda_u u + \lambda_w w.$$
>
> Independence forces $\lambda_u = \lambda_{u+w} = \lambda_w$. So $\lambda_u$ is the same scalar $\lambda$ for every nonzero $u$, i.e. $T = \lambda I$. $\blacksquare$

### Every $k$-Dimensional Subspace Invariant

{: .prompt-info }
> **Lemma.** Suppose $ k \in \\{ 1, \dots, n - 1 \\} $ and $ v \in V $ is nonzero. Then
>
> $$ \bigcap \{U \subseteq V : \dim U = k,\ v \in U\} = \operatorname{span}(v). $$

{: .prompt-proof }
> The $\supseteq$ direction is trivial ($v$ is in each such $U$). For $\subseteq$, I show that any $w \notin \operatorname{span}(v)$ is *excluded* by some $k$-dimensional subspace through $v$, so $w$ is not in the intersection.
>
> Suppose $w \notin \operatorname{span}(v)$. Then $v, w$ are independent, so extend them to a basis
>
> $$v,\ w,\ x_3,\ \dots,\ x_n.$$
>
> Since $k + 1 \le n$, the vectors $x_3, \dots, x_{k+1}$ exist (an empty list when $k = 1$), and we may set
>
> $$U = \operatorname{span}(v,\ x_3,\ \dots,\ x_{k+1}).$$
>
> Then $\dim U = k$ and $v \in U$. But $w \notin U$, because $v, x_3, \dots, x_{k+1}, w$ are part of a basis, hence independent. $\blacksquare$

{: .prompt-info }
> **Theorem.** Suppose $ k \in \\{ 1, \dots, n - 1 \\} $ and every subspace of $ V $ of dimension $ k $ is invariant under $ T $. Then $ T $ is a scalar multiple of the identity operator.

{: .prompt-proof }
> Let $v \in V$ be nonzero. By the lemma, $\operatorname{span}(v)$ is an intersection of $k$-dimensional subspaces, each invariant by hypothesis, so $\operatorname{span}(v)$ is invariant. Hence $Tv \in \operatorname{span}(v)$, i.e. $v$ is an eigenvector. As every nonzero vector is an eigenvector, $T$ is a scalar multiple of $I$. $\blacksquare$

{: .prompt-tip }
> "Every $k$-dimensional subspace is invariant" collapses all the way down to "every line is invariant", because a line is recoverable as the intersection of the $k$-dimensional subspaces sitting above it.

## Eigenvalues and Eigenvectors

{: .prompt-info }
> Every list of eigenvectors of $ T $ corresponding to distinct eigenvalues of $ T $ is _linearly independent_.

{: .prompt-info }
> Suppose $ v_1, \dots, v_m \in V $. Then
>
> $ v_1, \dots, v_m $ is linearly independent $ \iff \exists\, T \in \mathcal{L}(V) $ such that $ v_1, \dots, v_m $ are eigenvectors of $ T $ corresponding to distinct eigenvalues.

{: .prompt-info }
> Suppose $ \lambda $ is an eigenvalue of $ T $ and $ v_1, \dots, v_n $ is any basis of $ V $. Then
>
> $$ \left\lvert \lambda \right\rvert \le n \max\{\left\lvert \mathcal{M}(T, (v_1,\dots,v_n))_{j,k} \right\rvert : 1 \le j, k \le n \}. $$

### Upper Bound on the Number of Distinct Eigenvalues

{: .prompt-info }
> Tight upper bound on the number of _distinct_ eigenvalues:
>
> $$\#\{\text{distinct eigenvalues of } T\} \;\le\; \min\big(\dim V,\; 1 + \dim \operatorname{range} T\big)$$

{: .prompt-proof }
> The bound $\dim V$ follows from linear independence of eigenvectors with distinct eigenvalues. For the other bound, let $\lambda_1, \dots, \lambda_m$ be the distinct **nonzero** eigenvalues of $T$, with
> corresponding eigenvectors $v_1, \dots, v_m$.
>
> Each $v_i$ lies in $\operatorname{range} T$, since $\lambda_i \ne 0$ lets us write
>
> $$v_i = T\left(\tfrac{1}{\lambda_i} v_i\right).$$
>
> These $v_i$ are linearly independent, so $m \le \dim \operatorname{range} T$. The eigenvalues of $T$ are these $m$ nonzero ones, *plus possibly* $0$, so
>
> $$\#\{\text{distinct eigenvalues}\} \le m + 1 \le 1 + \dim \operatorname{range} T. \qquad \blacksquare$$

{: .prompt-tip }
> **Which of the two bounds is better.** Let $r = \dim \operatorname{range} T$. The comparison
> turns entirely on whether $0$ is an eigenvalue, i.e. whether $T$ fails to be invertible, i.e. whether $r < n$.
>
> - **$0$ is not an eigenvalue ($T$ is invertible).** Then $r = n$ and $1 + r = n + 1 > n$, so $\min$ selects
>   $\dim V$. The range bound is valid but vacuous.
> - **$0$ is an eigenvalue ($T$ is not invertible).** Then $r \le n - 1$ and $1 + r \le n$, so $\min$ selects
>   $1 + r$: a strict improvement whenever $r < n - 1$.

{: .prompt-warning }
> This is about $0$ being **an** eigenvalue, not the **only** one, so it is unrelated to
> nilpotency. $\operatorname{diag}(0,1,2,3)$ is non-invertible and far from nilpotent.
>
> Nilpotency is in fact where the bound is *weakest*: for $J_n(0)$ the true count is $1$
> while $r = n - 1$ gives a bound of $n$. The only nilpotent operator attaining the bound
> is $T = 0$, where $r = 0$ and $1 + r = 1$.

{: .prompt-tip }
> **Equality.**
>
> - $\\#\\{\text{distinct eigenvalues}\\} = 1 + r$ holds iff, in some basis,
>   $T = \operatorname{diag}(0, \dots, 0, \lambda_1, \dots, \lambda_r)$ with $r < n$ and the $\lambda_i$
>   distinct and nonzero. Equivalently: $T$ is diagonalizable, $0$ is an eigenvalue, and
>   every nonzero eigenvalue has geometric multiplicity $1$.
> - $\\#\\{\text{distinct eigenvalues}\\} = \dim V$ holds iff
>   $T = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ in some basis
>   with the $\lambda_i$ distinct. One of them is allowed to be $0$.

### Duals

{: .prompt-info }
> Suppose $ \lambda \in \mathbf{F} $. Then
>
> $\lambda$ is an eigenvalue of $T \iff \lambda$ is an eigenvalue of the dual operator $ T' \in \mathcal{L}(V'). $

## Null Space and Range Chains

{: .prompt-info }
> Suppose $m$ is a nonnegative integer. Then
>
> $$ \operatorname{null} T^m = \operatorname{null} T^{m + 1} \iff \operatorname{range} T^m = \operatorname{range} T^{m + 1} $$
>
> and once either holds at $m$, it holds at every $m' \ge m$.

{: .prompt-tip }
> So the null-space chain and the range chain stabilize at the *same* index; call it the **stabilization index**. Before it, $\dim\operatorname{null}T^m$ grows by at least $1$ at each step, so the chains have stabilized by $m = n$. This is why $G(\lambda,T) = \operatorname{null}(T-\lambda I)^n$ is a safe uniform definition. The true stabilization index of $T - \lambda I$ is usually much smaller, namely $e_\lambda$ (proved in [The Nilpotent Part](#the-nilpotent-part) below).

{: .prompt-info }
> Suppose $V_1,\dots,V_m$ are nonzero subspaces of $V$ with $V = \bigoplus_{k=1}^m V_k$, each $V_k$ invariant under $T$. Then for every $\lambda \in \mathbf{F}$,
>
> $$\operatorname{null}(T-\lambda I)^j = \bigoplus_{k=1}^m \operatorname{null}\big(\left.T\right\rvert_{V_k}-\lambda I\big)^j \qquad \text{for every } j \ge 0,$$
>
> and likewise $\operatorname{range}(T-\lambda I)^j = \bigoplus_k \operatorname{range}(\left.T\right\rvert_{V_k}-\lambda I)^j$.
>
> Taking $j = 1$ gives $E(\lambda, T) = \bigoplus_k E(\lambda, \left.T\right\rvert_{V_k})$; taking $j = n$ gives $G(\lambda, T) = \bigoplus_k G(\lambda, \left.T\right\rvert_{V_k})$.

{: .prompt-tip }
> Everything in this post is therefore computed blockwise. Immediately: $g_\lambda$ and $d_\lambda$ add across the blocks, the stabilization index of $T - \lambda I$ is $\max_k$ of the blockwise ones (so $e_\lambda = \max_k e_\lambda^{(k)}$, i.e. the minimal polynomial is the **lcm** of the blocks'), and the characteristic polynomial is their **product**. This is the engine behind both the primary decomposition and the Jordan form: choose the $V_k$ well and each block becomes trivial to read.

## Generalized Eigenspaces

{: .prompt-info }
> An eigenvalue $\lambda$ of $T$ is called **defective** when $\dim E(\lambda, T) \;<\; \dim G(\lambda, T)$, i.e. $g_\lambda < d_\lambda$.

Suppose the distinct eigenvalues of $T$ are $ \lambda_1, \dots, \lambda_m $.

|                                  | $G(\lambda_k,T)$                                                                           | $E(\lambda_k,T)$                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| Definition                       | $\operatorname{null}(T-\lambda_k I)^{j}$, any $j \ge e_{\lambda_k}$ ($j = n$ always works) | $\operatorname{null}(T-\lambda_k I)$                            |
| Containment                      | $E \subseteq G$                                                                            | —                                                               |
| Invariant                        | yes                                                                                        | yes                                                             |
| Restriction of $T - \lambda_k I$ | nilpotent, index exactly $e_{\lambda_k}$                                                   | zero                                                            |
| Sum is direct                    | always                                                                                     | always                                                          |
| Sum is all of $V$                | iff the minimal polynomial splits over $\mathbf{F}$ (automatic over $\mathbb{C}$)          | iff $T$ diagonalizable                                          |
| Multiplicity                     | algebraic $d_{\lambda_k} = \dim G$                                                         | geometric $g_{\lambda_k} = \dim E$, $1 \le g_{\lambda_k} \le d_{\lambda_k}$ |
| Min. polynomial                  | $\prod_k (z-\lambda_k)^{e_{\lambda_k}}$                                                    | $\prod_k(z-\lambda_k)$ **iff diagonalizable**                   |
| Char. polynomial                 | $\prod_k (z-\lambda_k)^{d_{\lambda_k}}$                                                    | — (always uses $d_{\lambda_k}$)                                 |

Over $\mathbb{C}$, every operator has an upper-triangular matrix with respect to some basis; see [Quotient Spaces]({% post_url 2026-07-04-quotient-spaces %}#induction-on-dimension).

{: .prompt-info }
> In an upper-triangular matrix of $T$,
>
> $$\{\text{distinct diagonal entries}\} = \{\text{zeros of } p_T\} = \{\text{eigenvalues}\},$$
>
> and for each eigenvalue $\lambda$,
>
> $$1 \le e_\lambda \le (\text{number of times } \lambda \text{ appears on the diagonal}) = d_\lambda. $$

### The Nilpotent Part

Every eigenvalue produces a nilpotent operator. If $\lambda$ is an eigenvalue of $T$, then $G(\lambda,T)$ is invariant under $T$, and

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

{: .prompt-info }
> $T$ is nilpotent $\iff G(0,T) = V$.

{: .prompt-proof }
> If $T$ is nilpotent, then $T^n = 0$ by [Nilpotent Operators]({% post_url 2026-07-25-nilpotent-operators %}), so $G(0,T) = \operatorname{null}T^n = V$. Conversely, $G(0,T) = V$ means $T^n = 0$. $\blacksquare$

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
> $e_\lambda = p$: the exponent of $z - \lambda$ in the minimal polynomial of $T$ is the index of nilpotency of $N$.

{: .prompt-proof }
> Write $S = T - \lambda I$, $G = G(\lambda,T) = \operatorname{null}S^n$, $R = \operatorname{range}S^n$, and $p_T(z) = (z-\lambda)^{e_\lambda}q(z)$ with $q(\lambda) \ne 0$. Then $V = G \oplus R$, both summands are invariant under every polynomial in $T$, and $S\vert_G = N$.
>
> **$p \le e_\lambda$.** Since $(T\vert_G - \lambda I)^p = N^p = 0$, the minimal polynomial of $T\vert_G$ divides $(z-\lambda)^p$, so it is $(z-\lambda)^j$ for some $j \le p$. Then $N^j = (T\vert_G - \lambda I)^j = 0$, which forces $j = p$. So the minimal polynomial of $T\vert_G$ is $(z-\lambda)^p$, and it [divides $p_T$]({% post_url 2026-07-22-polynomials %}#homomorphism-gives-divisibility-injectivity-gives-equality). Write $(z-\lambda)^{e_\lambda}q = h\,(z-\lambda)^p$. If $p > e_\lambda$, cancelling $(z-\lambda)^{e_\lambda}$ gives $q = h\,(z-\lambda)^{p-e_\lambda}$, so $q(\lambda) = 0$, a contradiction. Hence $p \le e_\lambda$.
>
> **$e_\lambda \le p$.** $S\vert_R$ is injective, because $\operatorname{null}S \subseteq G$ and $G \cap R = \\{0\\}$, so it is invertible. Restricting $p_T(T) = 0$ to $R$ gives $(S\vert_R)^{e_\lambda}\,q(T)\vert_R = 0$, so $q(T)\vert_R = 0$. Now $f = (z-\lambda)^pq$ satisfies $f(T)\vert_G = q(T)\vert_G\,N^p = 0$ and $f(T)\vert_R = (S\vert_R)^p\,q(T)\vert_R = 0$, so $f(T) = 0$ on $V = G \oplus R$. The minimal polynomial divides every polynomial that annihilates $T$, so $f = h\,p_T$ for some polynomial $h$, that is, $(z-\lambda)^pq = h\,(z-\lambda)^{e_\lambda}q$. Cancelling $q$ and comparing degrees gives $p = \deg h + e_\lambda \ge e_\lambda$. $\blacksquare$

{: .prompt-info }
> The stabilization index of $T - \lambda I$ is $e_\lambda$.

{: .prompt-proof }
> Since $\operatorname{null}(T - \lambda I)^j \subseteq G(\lambda,T)$ for every $j$, these null spaces are exactly the $\operatorname{null}N^j$, which grow strictly until $j = p$ and are constant after. So the [stabilization index](#null-space-and-range-chains) is $p = e_\lambda$. $\blacksquare$

{: .prompt-tip }
> Keep the three numbers apart: $1 \le g_\lambda \le d_\lambda$ and $1 \le e_\lambda \le d_\lambda$, but there is no general inequality between $g_\lambda$ and $e_\lambda$: for $T = 0$ on $\mathbf{F}^3$, $g_0 = 3$ but $e_0 = 1$, while for $J_3(0)$, $g_0 = 1$ but $e_0 = 3$. What does hold is
> $g_\lambda + e_\lambda - 1 \le d_\lambda$, since one Jordan block has size $e_\lambda$ and
> each of the other $g_\lambda - 1$ blocks has size at least $1$.

{: .prompt-info }
> *The nilpotent shadow of diagonalizability.*
>
> $$g_\lambda = d_\lambda \iff \mu = (1,\dots,1) \iff N = 0 \iff G(\lambda,T) = E(\lambda,T).$$
>
> That is, $\lambda$ is not defective exactly when every column has height $1$, and then $T$ acts on $G(\lambda,T)$ as $\lambda I$ rather than through a nontrivial Jordan form. If the generalized eigenspaces of $T$ sum to $V$ (automatic over $\mathbb{C}$) and no eigenvalue is defective, $T$ is diagonalizable.

## Polynomials and Inverses of $T$

### Eigenspaces of $p(T)$

{: .prompt-info }
> Suppose $ \mathbf{F} = \mathbb{C}$, $ p \in \mathcal{P}(\mathbb{C}) $, and $ \alpha \in \mathbb{C} $. Then
>
> (a) $G(\alpha,\, p(T)) \;=\; \bigoplus_{\lambda:\, p(\lambda) = \alpha} G(\lambda,\, T).$
>
> (b) $E(\alpha, p(T)) \;\supseteq\; \bigoplus_{\lambda:\,p(\lambda)=\alpha} E(\lambda, T)$, with equality iff $p'(\lambda) \neq 0$ for every defective $\lambda$ in that fiber.

{: .prompt-proof }
> (a) Each $G(\lambda, T)$ is $T$-invariant, hence $p(T)$-invariant. On $G(\lambda, T)$, write $T = \lambda I + N$ where $N$ is nilpotent. Since $\lambda I$ and $N$ commute, the algebraic Taylor expansion of $p$ around $\lambda$ holds:
>
> $$p(T)\big\rvert_{G(\lambda,T)} \;=\; \sum_{j=0}^{\deg p} \frac{p^{(j)}(\lambda)}{j!}\, N^j \;=\; p(\lambda)\,I \;+\; \underbrace{\Big(p'(\lambda)N + \tfrac{p''(\lambda)}{2}N^2 + \dots\Big)}_{=:\,M}.$$
>
> The remainder $M$ is a polynomial in $N$ with **zero constant term**, so it is nilpotent. So $p(T)$ acts on $G(\lambda, T)$ as $p(\lambda)I + (\text{nilpotent})$, and $G(\lambda,T) \subseteq G(p(\lambda), p(T))$. Hence the right side of (a) is contained in the left for every $\alpha$. Summing over $\alpha$, the right sides add up to $V$ and the left sides form a direct sum inside $V$, so dimensions force equality for each $\alpha$.
>
> (b) By the blockwise result, $E(\alpha, p(T))$ is the direct sum of its intersections with the $G(\lambda,T)$; the intersection is $\\{0\\}$ when $p(\lambda) \ne \alpha$ and $\operatorname{null}M$ when $p(\lambda) = \alpha$, whereas $E(\lambda, T) = \operatorname{null}N$. Factor $M = NQ$ with
>
> $$Q = p'(\lambda) I + \tfrac{p''(\lambda)}{2}N + \dots.$$
>
> Two cases:
>
> - **If $p'(\lambda) \neq 0$**, then $Q$ is (nonzero scalar) $+$ (nilpotent), hence invertible, so $\operatorname{null}M = \operatorname{null}N$.
> - **If $p'(\lambda) = 0$**, then $M$ starts at $N^2$ or higher, so $\operatorname{null}M \supseteq \operatorname{null}N^2$; and $\operatorname{null}N^2 \supsetneq \operatorname{null}N$ exactly when $N \ne 0$, i.e. when $\lambda$ is defective.
>
> So the ordinary eigenspace grows strictly exactly when some $\lambda$ in the fiber is **both a critical point of $p$ and a defective eigenvalue of $T$**. $\blacksquare$

### Inverses

{: .prompt-info }
> Suppose $ T $ is invertible with minimal polynomial $ p_T $ of degree $ m $. Then $ p_T(0) \ne 0 $ and
>
> $$p_{T^{-1}}(z) = \frac{z^m\, p_T(1/z)}{p_T(0)}.$$

{: .prompt-tip }
> $ p_T(0) \ne 0 $ is exactly the statement that $0$ is not an eigenvalue. The right-hand side is the *reversal* of $p_T$ (coefficients read backwards), rescaled to stay monic.

{: .prompt-info }
> Suppose $ T $ is invertible. For all nonzero $\lambda \in \mathbf{F} $,
>
> (a) $ G(\lambda, T) = G(\frac{1}{\lambda}, T^{-1}) $.
>
> (b) $ E(\lambda, T) = E(\frac{1}{\lambda}, T^{-1}) $.

## Diagonalizability

{: .prompt-info }
> Suppose $ \mathbf{F} = \mathbb{C} $ and $\lambda_1, \dots, \lambda_m$ are the distinct eigenvalues of $T$. The following are equivalent.
>
> (a) $T$ is diagonalizable.
>
> (b) $V$ has a basis consisting of eigenvectors of $T$.
>
> (c) $V = E(\lambda_1,T) \oplus \dots \oplus E(\lambda_m,T)$.
>
> (d) $\sum_k \dim E(\lambda_k, T) = \dim V$.
>
> (e) The minimal polynomial of $T$ is $(z-\lambda_1)\cdots(z-\lambda_m)$, i.e. it has no repeated roots.
>
> (f) $ V = \operatorname{null}(T - \lambda I) \oplus \operatorname{range}(T - \lambda I) $ for **every** $ \lambda \in \mathbb{C} $.

{: .prompt-tip }
> Diagonalizable means the eigenspaces are big enough to fill $V$. Conditions (b) to (e) each say "fill $V$" in a different dialect, the min-poly one saying that no eigenvalue needs a repeated factor to be annihilated. A repeated factor is exactly the symptom of an eigenspace that came up short: for
>
> $$T = \begin{pmatrix} 5 & 1 \\\\ 0 & 5 \end{pmatrix},$$
>
> $(T - 5I)$ kills $(1,0)$ but only maps $(0,1) \mapsto (1,0)$; it takes a second application to finish the job, so $(z-5)^2$ is the minimal polynomial and $\dim E(5,T) = 1 < 2 = \dim G(5,T)$.

{: .prompt-warning }
> The hypothesis $\mathbf{F} = \mathbb{C}$ is needed for (f). Rotation by $\pi/2$ on $\mathbb{R}^2$ has no real eigenvalues, so $T - \lambda I$ is invertible and (f) holds trivially for every $\lambda \in \mathbb{R}$, yet $T$ is not diagonalizable over $\mathbb{R}$.

{: .prompt-proof }
> Proof of (a) $\iff$ (f).
>
> *($\Rightarrow$)* A basis of eigenvectors of $T$ is also a basis of eigenvectors of $T - \lambda I$, with eigenvalues $\lambda_j - \lambda$. The basis vectors with eigenvalue $0$ span $\operatorname{null}(T - \lambda I)$, and the rest span $\operatorname{range}(T - \lambda I)$, so $V = \operatorname{null}(T - \lambda I) \oplus \operatorname{range}(T - \lambda I)$.
>
> *($\Leftarrow$)* Fix $\lambda$ and write $S = T - \lambda I$.
>
> **Step 1: $\operatorname{null} S = \operatorname{null} S^2$.** The inclusion $\subseteq$ always holds. Conversely, if $v \in \operatorname{null} S^2$ then $Sv \in \operatorname{null} S \cap \operatorname{range} S = \\{0\\}$, so $Sv = 0$.
>
> **Step 2: $G(\lambda, T) = E(\lambda, T)$.** By the [stabilization](#null-space-and-range-chains) of the null-space chain, Step 1 gives $\operatorname{null} S = \operatorname{null} S^j$ for every $j \geq 1$. Taking $j = n$:
>
> $$G(\lambda, T) = \operatorname{null} S^{n} = \operatorname{null} S = E(\lambda, T).$$
>
> **Step 3: conclude.** Over $\mathbb{C}$, $V = G(\lambda_1, T) \oplus \dots \oplus G(\lambda_m, T)$. By Step 2 each summand equals $E(\lambda_k, T)$, which is condition (c). $\blacksquare$

## The Hierarchy of Normal Forms

| $T \in \mathcal{L}(V)$ | Basis                                                                                                                                                  | Subspaces                                                                  | Dimensions                                                                                                                                                         | Minimal polynomial                                                                       | Nullspace and range                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Upper-triangularizable | $Tv_k \in \operatorname{span}(v_1,\dots,v_k)$ for each $k$                                                                                             | $\operatorname{span}(v_1,\dots,v_k)$ invariant for each $k$                | $\lambda_k$ occurs $\dim G(\lambda_k,T)$ times on the diagonal                                                                                                     | $(z-\lambda_1)\cdots(z-\lambda_m)$, $\lambda_i \in \mathbf{F}$, repetitions allowed      | —                                                                                                                          |
| Lower-triangularizable | $Tv_k \in \operatorname{span}(v_k,\dots,v_n)$ for each $k$                                                                                             | $\operatorname{span}(v_k,\dots,v_n)$ invariant for each $k$                | same (reverse the basis)                                                                                                                                           | same                                                                                     | —                                                                                                                          |
| Primary decomposition  | basis is a concatenation of bases of the $G(\lambda_k,T)$; inside group $k$, $(T-\lambda_k I)v \in \operatorname{span}$(earlier vectors of that group) | $V = G(\lambda_1,T)\oplus\dots\oplus G(\lambda_m,T)$, $\lambda_k$ distinct | $\sum_k \dim G(\lambda_k,T) = \dim V$; block $k$ has size $d_{\lambda_k}$                                                                                          | $\prod_k (z-\lambda_k)^{e_{\lambda_k}}$, $\lambda_k$ **distinct**, $e_{\lambda_k} \ge 1$ | $G(\lambda,T) = \operatorname{null}(T-\lambda I)^{\dim V}$                                                                 |
| Jordan form            | basis is a disjoint union of chains $v,\,(T{-}\lambda)v,\dots,(T{-}\lambda)^{s-1}v$; equivalently $Tv_k - \lambda v_k \in \\{0,\,v_{k-1}\\}$           | $V = $ direct sum of **indecomposable** invariant subspaces, one per chain | number of blocks for $\lambda$: $\dim E(\lambda,T)$; blocks of size $\ge j$: $\dim\operatorname{null}(T{-}\lambda)^j - \dim\operatorname{null}(T{-}\lambda)^{j-1}$ | $\prod_k (z-\lambda_k)^{e_{\lambda_k}}$, exponent $=$ largest block size                 | $\operatorname{rank}(T-\lambda)^j$ for all $j$ determines the entire diagram                                               |
| Diagonalizable         | a basis of eigenvectors of $T$                                                                                                                         | $V = E(\lambda_1,T)\oplus\dots\oplus E(\lambda_m,T)$                       | $\sum_k \dim E(\lambda_k,T) = \dim V$                                                                                                                              | $(z-\lambda_1)\cdots(z-\lambda_m)$, $\lambda_k$ **distinct**                             | $V = \operatorname{null}(T{-}\lambda I) \oplus \operatorname{range}(T{-}\lambda I)$ **for every** $\lambda \in \mathbf{F}$ (a characterization when $\mathbf{F} = \mathbb{C}$) |

![Hierarchy of upper-triangular, primary decomposition, and Jordan form](/assets/img/math/upper_triangular_primary_jordan_hierarchy.png)
_Each step refines the previous one._

{: .prompt-tip }
> Each refinement buys finer blocks and pays for it in uniqueness. The $G(\lambda_k,T)$ are determined by $T$ alone, with no choices. The individual Jordan blocks inside a $G$ are not: with $\mu = (2,1)$ there's a whole family of valid choices for the size-2 and size-1 summands, and only the multiset $\\{2,1\\}$ is forced.

![Matrix shapes versus operator classes](/assets/img/math/matrix_shapes_vs_operator_classes.png)
_Matrix shapes against the operator classes that realize them._

## Density of Diagonalizable Operators

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
