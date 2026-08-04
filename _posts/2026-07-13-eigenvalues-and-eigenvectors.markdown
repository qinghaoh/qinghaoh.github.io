---
title: "Eigenvalues and Eigenvectors"
category: math
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

| Symbol                                                | Meaning                                                                |
| ----------------------------------------------------- | ---------------------------------------------------------------------- |
| $\mathcal{L}(V)$                                      | linear operators $V \to V$                                             |
| $\mathcal{P}(\mathbf{F})$                             | polynomials with coefficients in $\mathbf{F}$                          |
| $E(\lambda,T) = \operatorname{null}(T - \lambda I)$   | eigenspace; $\lambda$ is an eigenvalue $\iff E(\lambda,T) \ne \\{0\\}$ |
| $G(\lambda,T) = \operatorname{null}(T - \lambda I)^n$ | generalized eigenspace                                                 |
| $g_\lambda = \dim E(\lambda,T)$                       | geometric multiplicity                                                 |
| $d_\lambda = \dim G(\lambda,T)$                       | algebraic multiplicity                                                 |
| $e_\lambda$                                           | exponent of $(z-\lambda)$ in the minimal polynomial                    |
| $J_s(\lambda)$                                        | Jordan block of size $s$ for $\lambda$                                 |
| $T'$, $V'$                                            | dual operator, dual space                                              |
| $T_{\mathbb{C}}$, $V_{\mathbb{C}}$                    | complexification (when $\mathbf{F} = \mathbb{R}$)                      |

{: .prompt-tip }
> Three numbers are attached to each eigenvalue and it pays to keep them apart:
> $g_\lambda \le e_\lambda \le d_\lambda$, with $g_\lambda = d_\lambda$ exactly when
> $\lambda$ is not defective.

## Invariant Subspaces

{: .prompt-info }
> Suppose $ U $ is a subspace of $ V $ invariant under $ T $. Then $ U $ is invariant under $ p(T) $ for every polynomial $ p \in \mathcal{P}(\mathbf{F}) $.

{: .prompt-info }
> Every subspace of an eigenspace is invariant.

{: .prompt-info }
> An arbitrary intersection of $T$-invariant subspaces is $T$-invariant.

### Every Line Invariant

The $k = 1$ case is the base of everything below: a $1$-dimensional invariant
subspace is exactly the span of an eigenvector.

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
> The $\supseteq$ direction is trivial ($v$ is in each such $U$). For $\subseteq$, I show any $w \notin \operatorname{span}(v)$ can be *excluded* by some $k$-subspace through $v$ — so $w$ can't be in the intersection.
>
> Suppose $w \notin \operatorname{span}(v)$. Then $v, w$ are independent, so extend them to a basis
>
> $$v,\ w,\ x_3,\ \dots,\ x_n.$$
>
> Now set
>
> $$U = \operatorname{span}(v,\ x_3,\ x_4,\ \dots,\ x_{k+1}) = \operatorname{span}(v) + \operatorname{span}(x_3,\dots,x_{k+1}).$$
>
> That's $v$ together with $k-1$ of the $x_i$'s (an empty list when $k = 1$), so $\dim U = k$, and $v \in U$; the vectors $x_3,\dots,x_{k+1}$ exist because $k + 1 \le n$. But $w \notin U$: the vectors $v, x_3, \dots, x_{k+1}$ are part of a basis that also includes $w$, so $w$ is independent of them and hence not in their span. This $U$ contains $v$, has dimension $k$, and misses $w$. $\blacksquare$

{: .prompt-info }
> **Theorem.** Suppose $ k \in \\{ 1, \dots, n - 1 \\} $ and every subspace of $ V $ of dimension $ k $ is invariant under $ T $. Then $ T $ is a scalar multiple of the identity operator.

{: .prompt-proof }
> Let $v \in V$ be nonzero. By the lemma, $\operatorname{span}(v)$ is an intersection of $k$-dimensional subspaces, each invariant by hypothesis, so $\operatorname{span}(v)$ is invariant. Hence $Tv \in \operatorname{span}(v)$, i.e. $v$ is an eigenvector. As every nonzero vector is an eigenvector, $T$ is a scalar multiple of $I$. $\blacksquare$

{: .prompt-tip }
> "Every $k$-dimensional subspace is invariant" collapses to "every line is invariant" — i.e. all the way down to $k=1$ — because a line is recoverable as the intersection of the $k$-subspaces sitting above it.

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

### Inverses

{: .prompt-info }
> Suppose $ T $ is invertible with minimal polynomial $ p_T $ of degree $ m $. Then $ p_T(0) \ne 0 $ and
>
> $$p_{T^{-1}}(z) = \frac{z^m\, p_T(1/z)}{p_T(0)}.$$

{: .prompt-tip }
> $ p_T(0) \ne 0 $ is exactly the statement that $0$ is not an eigenvalue. The right-hand side is the *reversal* of $p_T$ (coefficients read backwards), rescaled to stay monic.

{: .prompt-info }
> Suppose $ T \in \mathcal{L}(V) $ is invertible. For all $\lambda \in \mathbf{F} $ with $\lambda \ne 0$,
>
> (a) $ G(\lambda, T) = G(\frac{1}{\lambda}, T^{-1}) $.
>
> (b) $ E(\lambda, T) = E(\frac{1}{\lambda}, T^{-1}) $.

### Duals

{: .prompt-info }
> Suppose $ \lambda \in \mathbf{F} $. Then
>
> $\lambda$ is an eigenvalue of $T \iff \lambda$ is an eigenvalue of the dual operator $ T' \in \mathcal{L}(V'). $

### Upper Bound on the Number of Distinct Eigenvalues

{: .prompt-info }
> Tight upper bound on the number of _distinct_ eigenvalues:
>
> $$\#\{\text{distinct eigenvalues of } T\} \;\le\; \min\big(\dim V,\; 1 + \dim \operatorname{range} T\big)$$

{: .prompt-proof }
> Let $\lambda_1, \dots, \lambda_m$ be the distinct **nonzero** eigenvalues of $T$, with
> corresponding eigenvectors $v_1, \dots, v_m$.
>
> Each $v_i$ lies in $\operatorname{range} T$, since $\lambda_i \ne 0$ lets us write
>
> $$v_i = T\left(\tfrac{1}{\lambda_i} v_i\right).$$
>
> Eigenvectors belonging to distinct eigenvalues are linearly independent, so
> $\operatorname{range} T$ contains $m$ linearly independent vectors:
>
> $$m \le \dim \operatorname{range} T.$$
>
> The eigenvalues of $T$ are these $m$ nonzero ones, *plus possibly* $0$. Since $0$ is a
> single value, it adds at most $1$:
>
> $$\#\{\text{distinct eigenvalues}\} \le m + 1 \le 1 + \dim \operatorname{range} T. \qquad \blacksquare$$
>
> The bound $\\#\\{\text{distinct eigenvalues}\\} \le \dim V$ is standard, and the minimum of
> two valid bounds is valid.

{: .prompt-tip }
> **Which of the two bounds is better.** Let $r = \dim \operatorname{range} T$ and
> $n = \dim V$, so rank–nullity gives $\dim \operatorname{null} T = n - r$. The comparison
> turns entirely on whether $0$ is an eigenvalue — equivalently (all the same condition,
> stated in different vocabulary) whether $\operatorname{null} T \ne \\{0\\}$, whether $T$
> fails to be injective, whether $T$ fails to be invertible, whether $r < n$.
>
> - **$0$ is not an eigenvalue ($T$ is invertible).** Then $r = n$ and $1 + r = n + 1 > n$, so $\min$ selects
>   $\dim V$. The range bound is valid but vacuous.
> - **$0$ is an eigenvalue ($T$ is not invertible).** Then $r \le n - 1$ and $1 + r \le n$, so $\min$ selects
>   $1 + r$: a strict improvement whenever $r < n - 1$.

{: .prompt-warning }
> This is about $0$ being **an** eigenvalue, not the **only** one — it is unrelated to
> nilpotency. $\operatorname{diag}(0,1,2,3)$ is non-invertible and far from nilpotent.
>
> Nilpotency is in fact where the bound is *weakest*: for $J_n(0)$ the true count is $1$
> while $r = n - 1$ gives a bound of $n$. The only nilpotent operator attaining the bound
> is $T = 0$, where $r = 0$ and $1 + r = 1$.

{: .prompt-tip }
> **Equality.**
>
> - $\\#\\{\text{distinct eigenvalues}\\} = 1 + r$ holds iff, in some basis,
>   $T = \operatorname{diag}(0, \dots, 0, \lambda_1, \dots, \lambda_r)$ with the $\lambda_i$
>   distinct and nonzero — equivalently: $T$ is diagonalizable, $0$ is an eigenvalue, and
>   every nonzero eigenvalue has geometric multiplicity $1$.
> - $\\#\\{\text{distinct eigenvalues}\\} = \dim V$ holds iff $T$ has $n$ distinct
>   eigenvalues, i.e. $T = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ in some basis
>   with the $\lambda_i$ distinct. One of them is allowed to be $0$.

## Null Space and Range Chains

{: .prompt-info }
> Suppose $m$ is a nonnegative integer. Then
>
> $$ \operatorname{null} T^m = \operatorname{null} T^{m + 1} \iff \operatorname{range} T^m = \operatorname{range} T^{m + 1} $$
>
> and once either holds at $m$, it holds at every $m' \ge m$.

{: .prompt-tip }
> So the null-space chain and the range chain stabilize at the *same* index; call it the **stabilization index**.

{: .prompt-info }
> The stabilization index equals the nilpotency index of $ \left. T \right\rvert_{G(0,T)} $.

{: .prompt-tip }
> The chains always stabilize by $m = n$, which is why $G(\lambda,T) = \operatorname{null}(T-\lambda I)^n$ is a safe uniform definition — but the true stabilization index is usually much smaller, namely $e_\lambda$.

{: .prompt-info }
> Suppose $V_1,\dots,V_m$ are nonzero subspaces of $V$ with $V = \bigoplus_{k=1}^m V_k$, each $V_k$ invariant under $T$, and write $n_k = \dim V_k$. Then for every $\lambda \in \mathbf{F}$,
>
> $$\operatorname{null}(T-\lambda I)^j = \bigoplus_{k=1}^m \operatorname{null}\big(\left.T\right\rvert_{V_k}-\lambda I\big)^j \qquad \text{for every } j \ge 0,$$
>
> and likewise $\operatorname{range}(T-\lambda I)^j = \bigoplus_k \operatorname{range}(\left.T\right\rvert_{V_k}-\lambda I)^j$.
>
> Taking $j = 1$ gives $E(\lambda, T) = \bigoplus_k E(\lambda, \left.T\right\rvert_{V_k})$; taking any $j \ge \max_k n_k$ (in particular $j = n$) gives $G(\lambda, T) = \bigoplus_k G(\lambda, \left.T\right\rvert_{V_k})$.

{: .prompt-tip }
> Everything in this post is therefore computed blockwise. Immediately: $g_\lambda$ and $d_\lambda$ add across the blocks, the stabilization index of $T - \lambda I$ is $\max_k$ of the blockwise ones (so $e_\lambda = \max_k e_\lambda^{(k)}$, i.e. the minimal polynomial is the **lcm** of the blocks'), and the characteristic polynomial is their **product**. This is the engine behind both the primary decomposition and the Jordan form: choose the $V_k$ well and each block becomes trivial to read.

## Generalized Eigenspaces

{: .prompt-info }
> An eigenvalue $\lambda$ of $T$ is called **defective** when $\dim E(\lambda, T) \;<\; \dim G(\lambda, T)$, i.e. $g_\lambda < d_\lambda$.

{: .prompt-info }
> $\operatorname{rank}\big(\left.(T - \lambda I)\right\rvert_{G(\lambda,T)}\big) = \dim G(\lambda,T) - \dim E(\lambda,T)$

For a single eigenvalue $\lambda$, let $\mu = (m_1 \ge \dots \ge m_k)$ be the sizes of its
Jordan blocks (the *nilpotent diagram* of $\left.(T-\lambda I)\right\rvert_{G(\lambda,T)}$),
drawn as columns of boxes:

|                                   | reads off as                     | on the diagram               |
| --------------------------------- | -------------------------------- | ---------------------------- |
| $d_\lambda$ (char. poly exponent) | $\dim G(\lambda,T) = \sum_i m_i$ | total boxes (area)           |
| $e_\lambda$ (min. poly exponent)  | $m_1$                            | height of the tallest column |
| $g_\lambda$                       | $k = \mu'_1$                     | size of the bottom level     |

{: .prompt-tip }
> $g_\lambda = d_\lambda \iff$ every column has height $1 \iff \lambda$ is not defective. An operator with no defective eigenvalues is diagonalizable.

Suppose the distinct eigenvalues of $T$ are $ \lambda_1, \dots, \lambda_m $.

|                                  | $G(\lambda_k,T)$                                                                           | $E(\lambda_k,T)$                              |
| -------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------- |
| Definition                       | $\operatorname{null}(T-\lambda_k I)^{j}$, any $j \ge e_{\lambda_k}$ ($j = n$ always works) | $\operatorname{null}(T-\lambda_k I)$          |
| Containment                      | $E \subseteq G$                                                                            | —                                             |
| Invariant                        | yes                                                                                        | yes                                           |
| Restriction of $T - \lambda_k I$ | nilpotent, index exactly $e_{\lambda_k}$                                                   | zero                                          |
| Sum is direct                    | always                                                                                     | always                                        |
| Sum is all of $V$                | iff char. poly splits over $\mathbf{F}$ (automatic over $\mathbb{C}$)                      | iff $T$ diagonalizable                        |
| Multiplicity                     | algebraic $d_k = \dim G$                                                                   | geometric $g_k = \dim E$, $1 \le g_k \le d_k$ |
| Min. polynomial                  | $\prod_k (z-\lambda_k)^{e_{\lambda_k}}$                                                    | $\prod_k(z-\lambda_k)$ **iff diagonalizable** |
| Char. polynomial                 | $\prod_k (z-\lambda_k)^{d_k}$                                                              | — (always uses $d_k$)                         |

{: .prompt-info }
> In an upper-triangular matrix,
>
> $$\{\text{distinct diagonal entries}\} = \{\text{zeros of min poly}\} = \{\text{eigenvalues}\}.$$
>
> $$1 \le e_\lambda \le (\text{times } \lambda \text{ appears on the diagonal}) = \dim G(\lambda, T). $$

## Eigenspaces of $p(T)$

{: .prompt-info }
> Suppose $ \mathbf{F} = \mathbb{C}$, $ p \in \mathcal{P}(\mathbb{C}) $ is a nonconstant polynomial, and $ \alpha \in \mathbb{C} $. Then
>
> (a) $G(\alpha,\, p(T)) \;=\; \bigoplus_{\lambda:\, p(\lambda) = \alpha} G(\lambda,\, T).$
>
> (b) $E(\alpha, p(T)) \;\supseteq\; \bigoplus_{\lambda:\,p(\lambda)=\alpha} E(\lambda, T)$, with equality iff $p'(\lambda) \neq 0$ for every defective $\lambda$ in that fiber.

{: .prompt-proof }
> (a) Each $G(\lambda, T)$ is $T$-invariant, hence $p(T)$-invariant. On $G(\lambda, T)$, write $T = \lambda I + N$ where $N$ is nilpotent. Since $\lambda I$ and $N$ commute, the algebraic Taylor expansion of $p$ around $\lambda$ holds:
>
> $$p(T)\big\rvert_{G(\lambda,T)} \;=\; \sum_{j=0}^{\deg p} \frac{p^{(j)}(\lambda)}{j!}\, N^j \;=\; p(\lambda)\,I \;+\; \underbrace{\Big(p'(\lambda)N + \tfrac{p''(\lambda)}{2}N^2 + \dots\Big)}_{=:\,M}.$$
>
> The remainder $M$ is a polynomial in $N$ with **zero constant term**, so it's nilpotent. That means $p(T)$ acts on $G(\lambda, T)$ as $p(\lambda)I + (\text{nilpotent})$ — its *only* eigenvalue there is $p(\lambda)$, and the whole block sits inside $G(p(\lambda), p(T))$. Summing over all $\lambda$ with $p(\lambda) = \alpha$ and counting dimensions (both sides decompose $V$) upgrades the inclusion to equality.
>
> (b) Now zoom in from $G$ to $E$ inside a single block. The ordinary eigenspace of $p(T)$ for $\alpha = p(\lambda)$, intersected with $G(\lambda, T)$, is $\operatorname{null}(M)$, whereas $E(\lambda, T) = \operatorname{null}(N)$. Factor $M = N \cdot Q$ with
>
> $$Q = p'(\lambda) I + \tfrac{p''(\lambda)}{2}N + \dots.$$
>
> Two cases:
>
> - **If $p'(\lambda) \neq 0$**, then $Q$ is (nonzero scalar) $+$ (nilpotent), hence invertible, so $\operatorname{null}(M) = \operatorname{null}(N)$. The eigenspace matches exactly.
> - **If $p'(\lambda) = 0$**, then $M$ starts at $N^2$ or higher, so $\operatorname{null}(M) \supseteq \operatorname{null}(N^2)$; and $\operatorname{null}(N^2) \supsetneq \operatorname{null}(N)$ exactly when $N$ has a Jordan block of size $\ge 2$.
>
> So the precise condition for strict growth of the ordinary eigenspace is: **$\lambda$ is a critical point of $p$ (i.e. $p'(\lambda)=0$) *and* $\lambda$ is a defective eigenvalue of $T$ (has a nontrivial Jordan block).**

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
> Diagonalizable means the eigenspaces are *as big as they can be* — big enough to fill $V$. Each condition says "fill $V$" in a different dialect: enough eigenvectors for a basis, eigenspaces summing directly to $V$, dimensions adding to $\dim V$, and — the min-poly one — no eigenvalue needing a repeated factor to be annihilated. A repeated factor is exactly the symptom of an eigenspace that came up short: for
>
> $$T = \begin{pmatrix} 5 & 1 \\\\ 0 & 5 \end{pmatrix},$$
>
> $(T - 5I)$ kills $(1,0)$ but only maps $(0,1) \mapsto (1,0)$; it takes a second application to finish the job, so $(z-5)^2$ is the minimal polynomial and $\dim E(5,T) = 1 < 2 = \dim G(5,T)$.

{: .prompt-proof }
> Proof of (a) $\iff$ (f). Let $n = \dim V$.
>
> *($\Rightarrow$)* If $T$ is diagonalizable, a basis of eigenvectors of $T$ is also a basis of eigenvectors of $T - \lambda I$ (with eigenvalues $\lambda_j - \lambda$), so $T - \lambda I$ is diagonalizable, and for a diagonalizable operator $S$ we have $V = \operatorname{null} S \oplus \operatorname{range} S$.
>
> *($\Leftarrow$)* Assume $V = \operatorname{null}(T-\lambda I) \oplus \operatorname{range}(T-\lambda I)$ for every $\lambda \in \mathbb{C}$. Fix $\lambda$ and write $S = T - \lambda I$.
>
> **Step 1: $\operatorname{null} S = \operatorname{null} S^2$.** The inclusion $\subseteq$ always holds. Conversely, if $v \in \operatorname{null} S^2$ then $S(Sv) = 0$, so $Sv \in \operatorname{null} S$; also $Sv \in \operatorname{range} S$. Directness of the sum forces $\operatorname{null} S \cap \operatorname{range} S = \\{0\\}$, so $Sv = 0$.
>
> **Step 2: $G(\lambda, T) = E(\lambda, T)$.** By the result on stabilization of the null-space chain, Step 1 with $m = 1$ gives $\operatorname{null} S = \operatorname{null} S^j$ for every $j \geq 1$. Taking $j = n$ and using $G(\lambda,T) = \operatorname{null}(T-\lambda I)^{n}$:
>
> $$G(\lambda, T) = \operatorname{null} S^{n} = \operatorname{null} S = E(\lambda, T).$$
>
> **Step 3: conclude.** Let $\lambda_1, \dots, \lambda_m$ be the distinct eigenvalues of $T$ (there is at least one, since $V \neq \\{0\\}$ and $\mathbf{F} = \mathbb{C}$). The generalized eigenspace decomposition says
>
> $$V = G(\lambda_1, T) \oplus \dots \oplus G(\lambda_m, T).$$
>
> By Step 2 each summand equals $E(\lambda_j, T)$, so
>
> $$V = E(\lambda_1, T) \oplus \dots \oplus E(\lambda_m, T),$$
>
> which is condition (c). Hence $T$ is diagonalizable. $\blacksquare$

## The Hierarchy of Normal Forms

| $T \in \mathcal{L}(V)$ | Basis                                                                                                                                                  | Subspaces                                                                  | Dimensions                                                                                                                                                         | Minimal polynomial                                                                       | Nullspace and range                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Upper-triangularizable | $Tv_k \in \operatorname{span}(v_1,\dots,v_k)$ for each $k$                                                                                             | $\operatorname{span}(v_1,\dots,v_k)$ invariant for each $k$                | $\lambda_k$ occurs $\dim G(\lambda_k,T)$ times on the diagonal                                                                                                     | $(z-\lambda_1)\cdots(z-\lambda_m)$, $\lambda_i \in \mathbf{F}$, repetitions allowed      | —                                                                                                                          |
| Lower-triangularizable | $Tv_k \in \operatorname{span}(v_k,\dots,v_n)$ for each $k$                                                                                             | $\operatorname{span}(v_k,\dots,v_n)$ invariant for each $k$                | same (reverse the basis)                                                                                                                                           | same                                                                                     | —                                                                                                                          |
| Primary decomposition  | basis is a concatenation of bases of the $G(\lambda_k,T)$; inside group $k$, $(T-\lambda_k I)v \in \operatorname{span}$(earlier vectors of that group) | $V = G(\lambda_1,T)\oplus\dots\oplus G(\lambda_m,T)$, $\lambda_k$ distinct | $\sum_k \dim G(\lambda_k,T) = \dim V$; block $k$ has size $d_k$                                                                                                    | $\prod_k (z-\lambda_k)^{e_{\lambda_k}}$, $\lambda_k$ **distinct**, $e_{\lambda_k} \ge 1$ | $G(\lambda,T) = \operatorname{null}(T-\lambda I)^{\dim V}$                                                                 |
| Jordan form            | basis is a disjoint union of chains $v,\,(T{-}\lambda)v,\dots,(T{-}\lambda)^{s-1}v$; equivalently $Tv_k - \lambda v_k \in \\{0,\,v_{k-1}\\}$           | $V = $ direct sum of **indecomposable** invariant subspaces, one per chain | number of blocks for $\lambda$: $\dim E(\lambda,T)$; blocks of size $\ge j$: $\dim\operatorname{null}(T{-}\lambda)^j - \dim\operatorname{null}(T{-}\lambda)^{j-1}$ | $\prod_k (z-\lambda_k)^{e_{\lambda_k}}$, exponent $=$ largest block size                 | $\operatorname{rank}(T-\lambda)^j$ for all $j$ determines the entire diagram                                               |
| Diagonalizable         | a basis of eigenvectors of $T$                                                                                                                         | $V = E(\lambda_1,T)\oplus\dots\oplus E(\lambda_m,T)$                       | $\sum_k \dim E(\lambda_k,T) = \dim V$                                                                                                                              | $(z-\lambda_1)\cdots(z-\lambda_m)$, $\lambda_k$ **distinct**                             | $V = \operatorname{null}(T{-}\lambda I) \oplus \operatorname{range}(T{-}\lambda I)$ **for every** $\lambda \in \mathbf{F}$ |

![Hierarchy of upper-triangular, primary decomposition, and Jordan form](/assets/img/math/upper_triangular_primary_jordan_hierarchy.png)
_Each step refines the previous one._

{: .prompt-tip }
> Moving right buys you finer blocks and pays for it in uniqueness. The $G(\lambda_k,T)$ are determined by $T$ alone — no choices. The individual Jordan blocks inside a $G$ are not: with $\mu = (2,1)$ there's a whole family of valid choices for the size-2 and size-1 summands, and only the multiset $\\{2,1\\}$ is forced.

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
> Let $T \in \mathcal{L}(V)$, $n = \dim V$. Since $\mathbf{F} = \mathbb{C}$, there is a basis $v_1,\dots,v_n$ with respect to which $\mathcal{M}(T)$ is upper triangular, with diagonal entries $\lambda_1,\dots,\lambda_n$. Given $\varepsilon > 0$, choose $\varepsilon_1,\dots,\varepsilon_n \in \mathbb{C}$ with $\lvert \varepsilon_j \rvert < \varepsilon$ such that $\lambda_1 + \varepsilon_1, \dots, \lambda_n + \varepsilon_n$ are pairwise distinct — always possible, since each $\varepsilon_j$ needs only to avoid finitely many values, and any disc is infinite.
>
> Define the perturbation $P \in \mathcal{L}(V)$ by $Pv_j = \varepsilon_j v_j$, and set $D = T + P$. Then $\mathcal{M}(D)$ is upper triangular with the distinct entries $\lambda_j + \varepsilon_j$ on the diagonal. The diagonal of a triangular matrix lists the eigenvalues, so $D$ has $n$ distinct eigenvalues in a space of dimension $n$, hence is diagonalizable. And $D$ is close to $T$: in the operator norm induced by the inner product making $v_1,\dots,v_n$ orthonormal,
>
> $$\lVert T - D \rVert = \lVert P \rVert = \max_j \lvert \varepsilon_j \rvert < \varepsilon. \qquad \blacksquare$$

{: .prompt-warning }
> Density fails over $\mathbb{R}$ in the strong sense used here: a real operator with non-real eigenvalues (e.g. a rotation by $\pi/2$ on $\mathbb{R}^2$) has a whole neighbourhood of operators with no real eigenvalues at all, none of which is diagonalizable over $\mathbb{R}$.

