---
title:  "Commuting Operators"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
---

This post continues [Nilpotent Operators]({% post_url 2026-07-25-nilpotent-operators %}) and [Eigenvalues and Eigenvectors]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}), and asks which operators commute with a given $T \in \mathcal{L}(V)$. The first part collects what can be said about commuting operators in general, and ends with a reduction: over $\mathbb{C}$, describing the operators that commute with $T$ comes down to the same question for a **nilpotent** operator. The second part answers that question completely using the Young diagram, and the last section carries the answer back to an arbitrary $T$.

## Notation

{% include notation-table.md keys="n LV PF eigenspace gen_eigenspace min_poly commutant poly_alg partition conj_partition index_nilpotency" %}

Both $\mathcal{C}(T)$ and $\mathbf{F}[T]$ are subspaces of $\mathcal{L}(V)$, and in fact subalgebras: they are closed under composition.

{: .prompt-warning }
> $\mathbf{F}[T] \subseteq \mathcal{C}(T)$ always, since $T$ commutes with $I$ and with itself, hence with every polynomial in itself. That direction is free, and it is not the interesting one. The question this post answers is the **converse**: when is every operator commuting with $T$ a polynomial in $T$?

The smaller of the two spaces has a dimension that is easy to read off.

{: .prompt-info }
> $\dim\mathbf{F}[T] = \deg p_T$.

{: .prompt-proof }
> Let $d = \deg p_T$. *Spanning:* for $q \in \mathcal{P}(\mathbf{F})$, division gives $q = a\,p_T + r$ with $\deg r < d$, so $q(T) = r(T)$ and $I, T, \dots, T^{d-1}$ span $\mathbf{F}[T]$. *Independence:* a nontrivial relation $\sum_{t<d} c_tT^t = 0$ would be a nonzero polynomial of degree less than $d$ annihilating $T$, contradicting the minimality of $p_T$. $\blacksquare$

{: .prompt-tip }
> *Notation recall.* A nilpotent $N \in \mathcal{L}(V)$ splits $V$ into Jordan chains with tops $v_1,\dots,v_k$ and lengths $m_1 \ge \dots \ge m_k$, forming a partition $\mu$ of $\dim V$ drawn as a bottom-aligned array of boxes with the chains as columns. Its conjugate $\mu'$ has $\mu'_j = \\#\\{i : m_i \ge j\\}$, the size of level $j$, and $d_j = \dim\operatorname{null}N^j = \mu'_1 + \dots + \mu'_j$. Throughout, $p = m_1$ is the index of nilpotency and $n = \dim V$.

## Commuting Operators in General

Nothing in this section assumes nilpotency.

### Invariant Subspaces

{: .prompt-info }
> Suppose $ S,T \in \mathcal{L}(V) $ are such that $ ST = TS $. Suppose $ q \in \mathcal{P}(\mathbf{F}) $. Then
>
> $ \operatorname{null} q(S) $ and $ \operatorname{range} q(S) $ are invariant under $ T $.

{: .prompt-proof }
> $T$ commutes with every power of $S$, hence with $q(S)$. If $q(S)v = 0$, then $q(S)(Tv) = T\,q(S)v = 0$; and $T\,q(S)v = q(S)(Tv) \in \operatorname{range}q(S)$. $\blacksquare$

{: .prompt-tip }
> Special cases:
>
> * $ q(z) = z - \lambda $, then $ E(\lambda, S) $ is invariant under $ T $.
> * $ q(z) = (z - \lambda)^{\dim V} $, then $ G(\lambda, S) $ is invariant under $ T $.

### Simultaneous Diagonalization

{: .prompt-info }
> *simultaneous diagonalizability $\iff$ commutativity*
>
> Suppose $ \mathcal{E} $ is a subset of $ \mathcal{L}(V) $ and every element of $ \mathcal{E} $ is diagonalizable.
>
> There exists a basis of $ V $ with respect to which every element of $ \mathcal{E} $ has a diagonal matrix $\iff$ every pair of elements of $ \mathcal{E} $ commutes.

{: .prompt-proof }
> ($\Rightarrow$) Diagonal matrices commute, and $\mathcal{M}(ST) = \mathcal{M}(S)\mathcal{M}(T)$, so any two elements of $\mathcal{E}$ commute.
>
> ($\Leftarrow$) Suppose every pair in $\mathcal{E}$ commutes. Induct on $n = \dim V$.
>
> **Case 1: every $T \in \mathcal{E}$ is a scalar multiple of $I$.** Then every basis of $V$ works. (This covers $n = 1$, so the base case is free.)
>
> **Case 2: some $S \in \mathcal{E}$ is not a scalar multiple of $I$.** Since $S$ is diagonalizable with eigenvalues $\lambda_1,\dots,\lambda_m$,
>
> $$V = E(\lambda_1,S) \oplus \dots \oplus E(\lambda_m,S),$$
>
> and $m \geq 2$ (otherwise $S = \lambda_1 I$). So each $E(\lambda_j, S)$ is a subspace of dimension strictly less than $n$.
>
> Fix $j$ and write $W = E(\lambda_j, S)$. By [Invariant Subspaces](#invariant-subspaces), $W$ is invariant under every $T \in \mathcal{E}$. Each $\left. T \right\rvert_W$ is diagonalizable: its minimal polynomial divides $p_T$, which is a [product of distinct linear factors]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#generalized-eigenspaces) because $T$ is diagonalizable, so it is one too. The restricted family also commutes: for $T, R \in \mathcal{E}$ and $w \in W$, invariance gives $(\left. T \right\rvert_W)(\left. R \right\rvert_W)w = T(Rw) = (TR)w = (RT)w = (\left. R \right\rvert_W)(\left. T \right\rvert_W)w$.
>
> So $\mathcal{E}_j = \{\left. T\right\rvert_W : T \in \mathcal{E}\}$ is a commuting family of diagonalizable operators on a space of dimension $< n$. By the induction hypothesis there is a basis $\mathcal{B}_j$ of $W$ making *every* element of $\mathcal{E}_j$ diagonal. That is, every vector of $\mathcal{B}_j$ is an eigenvector of $T$ for every $T \in \mathcal{E}$ simultaneously.
>
> Now let $\mathcal{B} = \mathcal{B}_1 \cup \dots \cup \mathcal{B}_m$. Because $V$ is the direct sum of the $E(\lambda_j,S)$, this is a basis of $V$, and each of its vectors is an eigenvector of every $T \in \mathcal{E}$. So every element of $\mathcal{E}$ has a diagonal matrix with respect to $\mathcal{B}$. $\blacksquare$

![block diagonal to diagonal](/assets/img/math/block_diagonal_refinement_to_diagonal.png)

### Simultaneous Upper Triangularization

Without diagonalizability, commuting operators over $\mathbb{C}$ can still be triangularized together.

{: .prompt-info }
> Suppose $V$ is a finite-dimensional nonzero *complex* vector space. Suppose that $ \mathcal{E} \subseteq \mathcal{L}(V) $ is such that $S$ and $T$ commute for all $S,T \in \mathcal{E}$.
>
> (a) There is a vector in $V$ that is an eigenvector for every element of $\mathcal{E}$.
>
> (b) There is a basis of $V$ with respect to which every element of $\mathcal{E}$ has an upper-triangular matrix.

{: .prompt-proof }
> Both parts induct on $n = \dim V$, and both are trivial for $n = 1$.
>
> (a) If every element of $\mathcal{E}$ is a scalar multiple of $I$, any nonzero vector works. Otherwise pick $S \in \mathcal{E}$ that is not. Over $\mathbb{C}$, $S$ has an eigenvalue $\lambda$, and $W = E(\lambda, S)$ is nonzero and, since $S \ne \lambda I$, not all of $V$. As in the previous proof, $W$ is invariant under every element of $\mathcal{E}$ and the restrictions to $W$ commute. By the induction hypothesis some $w \in W$ is an eigenvector of every restriction, hence of every element of $\mathcal{E}$.
>
> (b) By (a) pick a common eigenvector $v_1$ and let $U = \operatorname{span}(v_1)$, which is invariant under every $T \in \mathcal{E}$. The [quotient operators]({% post_url 2026-07-04-quotient-spaces %}#quotient-operator) $T/U$ commute, since $(T/U)(R/U)(x + U) = TRx + U = RTx + U = (R/U)(T/U)(x+U)$. By the induction hypothesis, $V/U$ has a basis $v_2 + U, \dots, v_n + U$ making every $T/U$ upper triangular, i.e. $Tv_j \in \operatorname{span}(v_2,\dots,v_j) + U$ for $j \ge 2$. Then $v_1, \dots, v_n$ is a basis of $V$ with $Tv_j \in \operatorname{span}(v_1, \dots, v_j)$ for every $j$ and every $T \in \mathcal{E}$. $\blacksquare$

### Invertible Plus Nilpotent

{: .prompt-info }
> Suppose $S \in \mathcal{L}(V)$ is invertible, $N \in \mathcal{L}(V)$ is nilpotent with $N^p = 0$, and $SN = NS$. Then $S + N$ is invertible, and
>
> $$(S+N)^{-1} = \sum_{t=0}^{p-1}(-1)^t S^{-(t+1)}N^t.$$

{: .prompt-proof }
> Multiplying $SN = NS$ by $S^{-1}$ on both sides gives $NS^{-1} = S^{-1}N$, so $S^{-1}$ also commutes with $N$. Let $B = -S^{-1}N$. Commutativity lets the powers separate: $B^t = (-1)^tS^{-t}N^t$, so $B^p = 0$. The [geometric-series argument]({% post_url 2026-07-25-nilpotent-operators %}#inverse-of-a-jordan-block) then shows that $I - B$ is invertible with inverse $\sum_{t=0}^{p-1}B^t$. Since $S + N = S(I - B)$,
>
> $$(S+N)^{-1} = (I-B)^{-1}S^{-1} = \sum_{t=0}^{p-1}(-1)^tS^{-t}N^tS^{-1} = \sum_{t=0}^{p-1}(-1)^tS^{-(t+1)}N^t. \qquad \blacksquare$$

The case $S = \lambda I$ is the [inverse of $\lambda I + N$]({% post_url 2026-07-25-nilpotent-operators %}#inverse-of-a-jordan-block) from Nilpotent Operators.

{: .prompt-warning }
> Commutativity is essential. On $\mathbf{F}^2$, the invertible $$S = \begin{pmatrix}0&1\\1&0\end{pmatrix}$$ and the nilpotent $$N = \begin{pmatrix}0&0\\-1&0\end{pmatrix}$$ do not commute, and $$S + N = \begin{pmatrix}0&1\\0&0\end{pmatrix}$$ is not invertible.

### Reduction to the Nilpotent Case

The results above describe commuting *families*. To describe the commutant of a single
operator, the first move is to break $V$ into generalized eigenspaces, on each of which the
operator is a scalar plus a nilpotent.

{: .prompt-info }
> Suppose $\mathbf{F} = \mathbb{C}$ and $T$ has distinct eigenvalues $\lambda_1, \dots, \lambda_m$. Write $G_k = G(\lambda_k, T)$, so that $V = G_1 \oplus \dots \oplus G_m$, and let $N_k = \left.(T - \lambda_k I)\right\rvert_{G_k}$, which is nilpotent.
>
> (a) $ST = TS \iff$ each $G_k$ is invariant under $S$ **and** $\left. S \right\rvert_{G_k}$ commutes with $N_k$.
>
> (b) The map $S \mapsto \big(\left.S\right\rvert_{G_1}, \dots, \left.S\right\rvert_{G_m}\big)$ is an isomorphism
>
> $$\mathcal{C}(T) \;\cong\; \mathcal{C}(N_1) \times \dots \times \mathcal{C}(N_m), \qquad \dim\mathcal{C}(T) = \sum_{k=1}^m \dim\mathcal{C}(N_k).$$

{: .prompt-proof }
> (a) ($\Rightarrow$) $G_k = \operatorname{null}(T - \lambda_kI)^n$ is invariant under $S$ by [Invariant Subspaces](#invariant-subspaces), with the roles of $S$ and $T$ swapped. Restricting $ST = TS$ to $G_k$ and subtracting $\lambda_k \left.S\right\rvert_{G_k}$ from both sides gives $\left.S\right\rvert_{G_k}N_k = N_k\left.S\right\rvert_{G_k}$.
>
> ($\Leftarrow$) $\left.S\right\rvert_{G_k}$ commutes with $N_k + \lambda_kI = \left.T\right\rvert_{G_k}$, so $STv = TSv$ for $v$ in each $G_k$, hence for every $v \in V$.
>
> (b) By (a) the map lands in the product. It is linear, and injective because an operator is determined by its restrictions to the summands of a direct sum. For surjectivity, given $R_k \in \mathcal{C}(N_k)$, define $S$ to be $R_k$ on $G_k$; then each $G_k$ is $S$-invariant with $\left.S\right\rvert_{G_k} = R_k$, so $S \in \mathcal{C}(T)$ by (a). $\blacksquare$

{: .prompt-tip }
> Adding a scalar multiple of $I$ changes nothing about what commutes with an operator: $\mathcal{C}(\lambda I + N) = \mathcal{C}(N)$. That is why (a) can use $N_k$ in place of $\left.T\right\rvert_{G_k} = \lambda_kI + N_k$. **Everything below therefore takes $N$ nilpotent, with no loss**; [Back to Arbitrary Operators](#back-to-arbitrary-operators) assembles the blocks.

## The Commutant of a Nilpotent Operator

From here on $N \in \mathcal{L}(V)$ is nilpotent, with a Jordan basis of tops $v_1, \dots, v_k$ and lengths $m_1 \ge \dots \ge m_k$, so the vectors $N^jv_i$ with $0 \le j \le m_i - 1$ form a basis of $V$.

### Parametrization

A commuting operator is pinned down by where it sends the $k$ tops, and each top may go anywhere killed by the right power of $N$.

{: .prompt-info }
> The map $S \mapsto (Sv_1,\dots,Sv_k)$ is an isomorphism
>
> $$\mathcal{C}(N) \;\cong\; \operatorname{null}N^{m_1} \times \dots \times \operatorname{null}N^{m_k}.$$
>
> Its inverse sends $(w_1, \dots, w_k)$ to the operator defined on the Jordan basis by
>
> $$S(N^jv_i) := N^j w_i, \qquad 0 \le j \le m_i - 1.$$

{: .prompt-proof }
> *Into the product.* If $SN = NS$, then $SN^j = N^jS$ for all $j$. Since $N^{m_i}v_i = 0$,
>
> $$0 = S(N^{m_i}v_i) = N^{m_i}(Sv_i),$$
>
> so $Sv_i \in \operatorname{null}N^{m_i}$. The map is clearly linear.
>
> *Injective.* $S(N^jv_i) = N^j(Sv_i)$. The left side is $S$ on an arbitrary basis vector; the right side only involves the $k$ vectors $Sv_1, \dots, Sv_k$.
>
> *Surjective.* Given $w_i \in \operatorname{null}N^{m_i}$, define $S$ as in the statement and check $SN = NS$ on a basis vector $N^jv_i$:
>
> - If $j < m_i - 1$: $\;SN(N^jv_i) = S(N^{j+1}v_i) = N^{j+1}w_i$, while $NS(N^jv_i) = N(N^jw_i) = N^{j+1}w_i$. Equal.
> - If $j = m_i-1$ (bottom of the chain): $\;SN(N^{m_i-1}v_i) = S(0) = 0$, while $NS(N^{m_i-1}v_i) = N^{m_i}w_i = 0$ by the constraint.
>
> So the constraint $w_i \in \operatorname{null}N^{m_i}$ is precisely what is needed at the *bottom* of each chain, and nowhere else. $\blacksquare$

In practice: find a Jordan basis, choose each $w_i$ in $\operatorname{null}N^{m_i}$, and extend by $S(N^jv_i) := N^jw_i$. Every commuting operator arises this way, exactly once.

### Dimension

$\operatorname{null}N^m$ is spanned by the bottom $\min(m, m_j)$ vectors of every chain $j$, so $d_m = \sum_j \min(m, m_j)$.

{: .prompt-info }
> $$\dim\mathcal{C}(N) = \sum_{i,j}\min(m_i,m_j) = \sum_l \left(\mu'_l\right)^2.$$

{: .prompt-proof }
> By the parametrization, $\dim\mathcal{C}(N) = \sum_i d_{m_i} = \sum_{i,j}\min(m_i, m_j)$. Writing $\min(m_i,m_j) = \\#\\{l : l \le m_i \text{ and } l \le m_j\\}$ turns the double sum into a count of triples $(i,j,l)$:
>
> $$\sum_{i,j}\min(m_i,m_j) \;=\; \sum_{l \ge 1}\#\{i : m_i \ge l\}\cdot\#\{j : m_j \ge l\} \;=\; \sum_{l\ge1}\left(\mu'_l\right)^2. \qquad \blacksquare$$

Here are the windows $\operatorname{null}N^{m_i}$ for $\mu = (3,2,1)$:

![w windows](/assets/img/math/null_n_mi_window_clipping_ragged_diagram.png)

So $\dim\mathcal{C}(N)$ is the sum of squares of the level sizes: the commutant is large when the diagram is short and wide, with many chains of similar length, and small when it is tall and thin. The extremes are as far apart as they could be.

{: .prompt-info }
> For every nilpotent $N$ on $V$ with $\dim V = n$,
>
> $$n \;\le\; \dim\mathcal{C}(N) \;\le\; n^2,$$
>
> with equality on the left exactly when the diagram is a single column ($k = 1$), and on the right exactly when it is a single level ($N = 0$).

{: .prompt-proof }
> Since each nonzero $\mu'_l$ is at least $1$,
>
> $$\sum_l (\mu'_l)^2 \;\ge\; \sum_l \mu'_l \;=\; n,$$
>
> with equality iff every level has size $1$, i.e. $k = \mu'_1 = 1$. And since all terms are non-negative,
>
> $$\sum_l (\mu'_l)^2 \;\le\; \left(\sum_l \mu'_l\right)^2 \;=\; n^2,$$
>
> with equality iff only one term is nonzero, i.e. $p = 1$ and $N = 0$. $\blacksquare$

The [two extremes](#two-extremes) are worked out below.

### Commuting Operators Need Not Preserve Chains

{: .prompt-warning }
> Nothing in the parametrization says $Sv_i$ has to lie in chain $i$. The constraint space $\operatorname{null}N^{m_i}$ is not a subspace of chain $i$: it is spanned by the bottom $\min(m_i, m_j)$ vectors of **every** chain $j$, so it reaches across the whole array. When all lengths equal $p$, $\operatorname{null}N^{m_i} = \operatorname{null}N^p = V$ and there is no constraint whatsoever.

**Swapping chains.** Take $\mathbf{F}^4$ with two chains of length $2$:

$$e_2 \mapsto e_1 \mapsto 0, \qquad e_4 \mapsto e_3 \mapsto 0.$$

Here $N^2 = 0$, so both constraints read $w_i \in \mathbf{F}^4$ and any pair of tops is legal. Swapping the chains is the choice $w_1 = e_4$, $w_2 = e_2$, which forces

$$Se_2 = e_4, \quad Se_1 = Ne_4 = e_3, \qquad Se_4 = e_2, \quad Se_3 = Ne_2 = e_1.$$

The construction guarantees $SN = NS$; as a spot-check, $SNe_2 = Se_1 = e_3$ and $NSe_2 = Ne_4 = e_3$. So $S$ commutes with $N$ while carrying chain $1$ onto chain $2$.

**Shearing.** Take $w_1 = e_2 + e_4$ and $w_2 = e_4$ instead, giving

$$Se_2 = e_2 + e_4, \quad Se_1 = e_1 + e_3, \qquad Se_4 = e_4, \quad Se_3 = e_3.$$

This blends chain $1$ into chain $2$ without being a permutation of the basis.

**No constraint at all.** For $N = 0$ on $\mathbf{F}^2$, two chains of length $1$, every operator commutes, so the commutant is all of $\mathcal{L}(\mathbf{F}^2)$, of dimension $4$. The ones preserving the chain $\operatorname{span}(e_1)$ are the upper triangular matrices, a proper subspace of dimension $3$. Preserving a chosen chain is a genuine restriction that most commuting operators fail.

**Unequal lengths.** This is where the constraint bites. Take $\mathbf{F}^3$ with chains

$$e_2 \mapsto e_1 \mapsto 0, \qquad f \mapsto 0,$$

so $m_1 = 2$ and $m_2 = 1$. The top of the long chain is unconstrained, since $\operatorname{null}N^2 = \mathbf{F}^3$, so $Se_2$ may be $f$. But the top of the short chain satisfies $Sf \in \operatorname{null}N = \operatorname{span}(e_1, f)$, so $Sf$ cannot be $e_2$: a short chain can only be sent into the bottom portion of a longer one, never onto its top. The freedom is **asymmetric**, and $\min(m_i, m_j)$ is exactly the bookkeeping for it.

**Worked example**

Take $N = \partial/\partial x$ on $V = \operatorname{span}(1,x,y,x^2,xy)$, with the Jordan basis [found in Nilpotent Operators]({% post_url 2026-07-25-nilpotent-operators %}#find-a-jordan-basis): chain $x^2 \mapsto 2x \mapsto 2$ and chain $xy \mapsto y$. So $v_1 = x^2$ with $m_1 = 3$, and $v_2 = xy$ with $m_2 = 2$.

Constraints: $w_1 \in \operatorname{null}N^3 = V$, unrestricted; $w_2 \in \operatorname{null}N^2 = \operatorname{span}(1,y,x,xy)$. The asymmetry is visible: $w_1$ may be $xy$, but $w_2$ may **not** be $x^2$, since $N^2x^2 = 2 \neq 0$.

Choose $w_1 = xy$ and $w_2 = 2x$. Then

$$S(x^2) = xy, \quad S(2x) = N(xy) = y, \quad S(2) = N^2(xy) = 0,$$
$$S(xy) = 2x, \quad S(y) = N(2x) = 2.$$

Spot-check: $SN(x^2) = S(2x) = y$ and $NS(x^2) = N(xy) = y$; also $SN(xy) = S(y) = 2$ and $NS(xy) = N(2x) = 2$.

Dimension check: $\dim\operatorname{null}N^3 + \dim\operatorname{null}N^2 = 5 + 4 = 9$, agreeing with $\sum_{i,j}\min(m_i,m_j) = 3+2+2+2$ and with $\sum_l(\mu'_l)^2 = 4 + 4 + 1$, out of $25$ for all of $\mathcal{L}(V)$.

The isomorphism also gives a **basis** of the commutant for free: run over a basis of $\operatorname{null}N^{m_i}$ for one index with the other $w$'s set to $0$. Here that gives $5 + 4 = 9$ explicit commuting operators spanning the commutant.

**Exercise**

{: .prompt-info }
> Give an example of two commuting operators $S,T$ on $\mathbf{F}^4$ such that there is a subspace of $\mathbf{F}^4$ that is invariant under $S$ but not under $T$ and there is a subspace of $\mathbf{F}^4$ that is invariant under $T$ but not under $S$.

Take the two chains of length $2$ and the chain swap $S$ from [Swapping chains](#commuting-operators-need-not-preserve-chains) above, so $SN = NS$. Then

- $\operatorname{span}(e_1)$ is $N$-invariant, since $Ne_1 = 0$, but not $S$-invariant, since $Se_1 = e_3$;
- $\operatorname{span}(e_2+e_4)$ is $S$-invariant, since $S(e_2+e_4) = e_2+e_4$, but not $N$-invariant, since $N(e_2+e_4) = e_1+e_3$.

Contrast [Invariant Subspaces](#invariant-subspaces): commuting operators share every subspace of the form $\operatorname{null}q(S)$ or $\operatorname{range}q(S)$, but not invariant subspaces in general.

## Two Extremes

[Nilpotent Operators]({% post_url 2026-07-25-nilpotent-operators %}#two-extremes) singles out two shapes. **One column**, $\mu = (n)$ and $\mu' = (1^n)$, is a single
chain. **One level**, $\mu = (1^n)$ and $\mu' = (n)$, is the zero operator. They are conjugate
partitions, and they are the two equality cases of $n \le \dim\mathcal{C}(N) \le n^2$: one commutant
is as small as it can be, the other as large.

### One Column: a Single Chain

If $k = 1$ with length $m$, then $\operatorname{null}N^m = V$, so there is **no constraint at all** and $S$ is determined by an arbitrary $w = Sv$. Since $k = 1$, the chain

$$v,\; Nv,\; \dots,\; N^{m-1}v$$

is a basis of $V$, so expand $w$ in it:

$$w = c_0v + c_1Nv + \dots + c_{m-1}N^{m-1}v = q(N)v, \qquad q(z) = c_0 + c_1z + \dots + c_{m-1}z^{m-1}.$$

Now compare $S$ with $q(N)$ on each basis vector $N^tv$, for $0 \le t \le m-1$:

$$S(N^tv) \;=\; N^t(Sv) \;=\; N^t\big(q(N)v\big) \;=\; q(N)(N^tv).$$

The first equality is $SN^t = N^tS$, from $SN = NS$. The second substitutes $Sv = w = q(N)v$. The third is $N^tq(N) = q(N)N^t$, since polynomials in $N$ commute with powers of $N$.

So $S$ and $q(N)$ agree on a basis of $V$, hence are equal as operators:

$$S = c_0I + c_1N + \dots + c_{m-1}N^{m-1}.$$

In the basis $(N^{m-1}v,\dots,Nv,v)$ this is the upper-triangular [Toeplitz matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix) with $c_0$ down the diagonal, $c_1$ on the next diagonal, and so on. For $n = 3$, with $e_3 \mapsto e_2 \mapsto e_1 \mapsto 0$ and $Se_3 = w = ae_1 + be_2 + ce_3$,

$$N = \begin{pmatrix} 0&1&0\\ 0&0&1\\ 0&0&0\end{pmatrix}, \qquad S = \begin{pmatrix} c&b&a\\ 0&c&b\\ 0&0&c\end{pmatrix} = cI + bN + aN^2.$$

So for one column the commutant is exactly $\mathbf{F}[N]$, and $\dim\mathcal{C}(N) = n$.
That turns out to characterize this shape: as soon as $k \ge 2$, "commutes with $N$" stops being the same condition as "is a polynomial in $N$".

{: .prompt-info }
> For $N$ nilpotent with $k$ chains,
>
> $$\mathcal{C}(N) = \mathbf{F}[N] \iff k = 1,$$
>
> that is, iff $N$ has a single Jordan block, equivalently $\dim\operatorname{null}N = 1$.

{: .prompt-proof }
> ($\Leftarrow$) is the computation above.
>
> ($\Rightarrow$) The minimal polynomial of $N$ is $z^p$, so $\dim\mathbf{F}[N] = p$. If $k \ge 2$, then $n = \sum_i m_i > m_1 = p$, since the omitted lengths are positive, so by the [lower bound](#dimension)
>
> $$\dim\mathcal{C}(N) \ \ge\ n \ >\ p \ =\ \dim\mathbf{F}[N]$$
>
> and the inclusion $\mathbf{F}[N] \subseteq \mathcal{C}(N)$ is strict. $\blacksquare$

### One Level: the Zero Operator

The opposite shape is $N = 0$, where every chain has length $1$. Now the constraint
$w_i \in \operatorname{null}N^{m_i}$ reads $w_i \in \operatorname{null}N = V$ for every $i$, so
there is no constraint at all and *every* operator commutes:

$$\mathcal{C}(0) = \mathcal{L}(V), \qquad \dim\mathcal{C}(0) = n^2.$$

Meanwhile $N^1 = 0$ already, so $p = 1$ and $\mathbf{F}[N] = \\{c I : c \in \mathbf{F}\\}$ is just
the scalars, of dimension $1$. This is the failure of $\mathcal{C}(N) = \mathbf{F}[N]$ at its
most extreme: $n^2$ against $1$.

### Comparing the Ends

|                                    | one column | one level          |
| ---------------------------------- | ---------- | ------------------ |
| $\mu$                              | $(n)$      | $(1^n)$            |
| $\mu'$                             | $(1^n)$    | $(n)$              |
| $\dim\mathcal{C}(N)$               | $n$        | $n^2$              |
| $\dim\mathbf{F}[N] = p$           | $n$        | $1$                |
| $\mathcal{C}(N) = \mathbf{F}[N]$? | yes        | no, unless $n = 1$ |

{: .prompt-tip }
> *Why the commutant measures how non-unique a Jordan basis is.*
>
> The invertible elements of $\mathcal{C}(N)$ act simply transitively on the Jordan bases of a
> fixed shape. Given two of them, with tops $v_i$ and $v'_i$, each $v'_i$ lies in $\operatorname{null}N^{m_i}$, so the
> [parametrization](#parametrization) gives $S \in \mathcal{C}(N)$ with $Sv_i = v'_i$, hence $S(N^jv_i) = N^jv'_i$:
> $S$ carries the first basis onto the second, so it is invertible, and it is the only element of $\mathcal{C}(N)$ that does so.
> Conversely, an invertible $S \in \mathcal{C}(N)$ carries a Jordan basis to a Jordan basis.
> So $\dim\mathcal{C}(N)$ is the dimension of the choice available,
> which is where the counts in [Nilpotent Operators]({% post_url 2026-07-25-nilpotent-operators %}#two-extremes) came from: every basis of $V$ when $N = 0$
> ($n^2$ worth of freedom), and only the choice of a top when $k = 1$ ($n$ worth).
>
> The complementary count is just as clean. The operators similar to $N$ form a set of dimension
>
> $$\dim\mathcal{L}(V) - \dim\mathcal{C}(N) \;=\; n^2 - \sum_j (\mu'_j)^2 .$$
>
> Freedom in the basis and size of the similarity class always sum to $n^2$. For $N = 0$ the
> class is a single point, since nothing is similar to the zero operator but itself; for a
> single chain it is $n^2 - n$, as large as a nilpotent operator's class can be.

## Back to Arbitrary Operators

Over $\mathbb{C}$, the [reduction](#reduction-to-the-nilpotent-case) assembles the nilpotent answers. With $G_k$ and $N_k$ as there and $\mu^{(k)}$ the diagram of $N_k$,

$$\dim\mathcal{C}(T) = \sum_{k=1}^m \dim\mathcal{C}(N_k) = \sum_{k=1}^m \sum_l \left(\mu^{(k)\prime}_l\right)^2.$$

The question from the start of the post now has a complete answer.

{: .prompt-info }
> Suppose $\mathbf{F} = \mathbb{C}$. Then $\mathcal{C}(T) = \mathbf{F}[T]$ iff the minimal and characteristic polynomials of $T$ coincide.

{: .prompt-proof }
> Write $d_k = \dim G_k$. By the lower bound applied to each $N_k$, $\dim\mathcal{C}(N_k) \ge d_k$, with equality iff $N_k$ is a single chain. Summing, $\dim\mathcal{C}(T) \ge n$, with equality iff every $N_k$ is a single chain.
>
> The exponent of $z - \lambda_k$ in the characteristic polynomial is $d_k$, and in $p_T$ it is the [index of nilpotency of $N_k$]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#the-nilpotent-part). A nilpotent operator on a $d_k$-dimensional space has index $d_k$ iff it is a single chain, since $d_k = \sum_i m_i \ge m_1$ with equality iff there is one chain. So the two polynomials coincide iff every $N_k$ is a single chain, iff $\dim\mathcal{C}(T) = n$.
>
> If they coincide, $\dim\mathbf{F}[T] = \deg p_T = n = \dim\mathcal{C}(T)$, and an inclusion of subspaces of equal finite dimension is an equality. If not, $\dim\mathbf{F}[T] = \deg p_T < n \le \dim\mathcal{C}(T)$, and the inclusion is strict. $\blacksquare$

{: .prompt-tip }
> The minimal and characteristic polynomials coincide iff $T$ is **cyclic**: some $v$ has $v, Tv, \dots, T^{n-1}v$ a basis of $V$. For cyclic $T$ the [One Column](#one-column-a-single-chain) computation runs verbatim with $T$ in place of $N$: write $Sv = q(T)v$, then $S(T^jv) = T^jq(T)v = q(T)(T^jv)$, so $S = q(T)$. For nilpotent $N$ the minimal polynomial is $z^p$ and the characteristic polynomial is $z^n$, so the condition is $p = n$: a single chain filling all of $V$.
