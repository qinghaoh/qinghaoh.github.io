---
title:  "Nilpotent Operators"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
---

## Notation

{% include notation-table.md keys="n LV quotient_space quotient_op jordan_block partition conj_partition index_nilpotency commutant MT" %}

## Jordan Basis

{: .prompt-info }
> Let $N \in \mathcal{L}(V)$ be nilpotent. A **Jordan basis** for $N$ consists of vectors $v_1,\dots,v_k$ (the **tops**) and lengths $m_1,\dots,m_k$ such that
>
> $$\big\{\, N^j v_i \;:\; i = 1,\dots,k,\;\; j = 0,\dots,m_i-1 \,\big\}$$
>
> is a basis of $V$ and $N^{m_i}v_i = 0$ for each $i$. The list $v_i, Nv_i, \dots, N^{m_i-1}v_i$ is a **Jordan chain**, and its last vector $N^{m_i-1}v_i$ is the chain's **bottom**.
>
> Every nilpotent operator has a Jordan basis.

The algorithm in [Find a Jordan Basis](#find-a-jordan-basis) below constructs one, which proves existence. Until then, fix a Jordan basis and study its shape.

Order the chains so that $m_1 \ge m_2 \ge \dots \ge m_k$. The lengths then form a **partition** of $\dim V$,

$$\mu = (m_1 \ge m_2 \ge \dots \ge m_k), \qquad m_1 + \dots + m_k = \dim V,$$

whose **parts** are the $m_i$. Picture each chain as a column, top vector at the top, with the columns **aligned at the bottom**:

$$\begin{array}{cccc|l} v_1 & & & & \mu'_{m_1}\\ Nv_1 & v_2 & & & \\ \vdots & \vdots & \ddots & & \vdots\\ N^{m_1-2}v_1 & N^{m_2-2}v_2 & & v_k & \mu'_2\\ N^{m_1-1}v_1 & N^{m_2-1}v_2 & \cdots & N^{m_k-1}v_k & \mu'_1\\ \hline m_1 & m_2 & \cdots & m_k & \end{array}$$

$N$ moves each entry one step down its column, and off the bottom to $0$. Each part $m_i$ is the height of column $i$, recorded along the bottom margin.

Now read the same boxes across instead of down, numbering rows from the bottom: the bottom row is **level $1$**, the row above it level $2$, and so on. Level $j$ contains one box from every chain tall enough to reach it, so its size, recorded along the right margin, is

$$\mu'_j \;:=\; \#\{\,i : m_i \ge j\,\}.$$

The sequence $\mu' = (\mu'_1, \mu'_2, \dots)$ is again a partition of $\dim V$, called the **conjugate** of $\mu$, and conjugating twice returns $\mu$. Every chain reaches level $1$ and none reaches above level $m_1$, so

$$\mu'_1 = k, \qquad \mu'_j = 0 \ \text{ for } j > m_1.$$

Thus $\mu'$ has exactly $m_1$ parts, and its first part is the number of chains.

The tallest column also gives the index of nilpotency: $N^{m_1}$ kills every chain, while $N^{m_1-1}v_1 \neq 0$ is the bottom of the longest one. So $p = m_1$. In particular $p \le \dim V$, so $N^{\dim V} = 0$.

{: .prompt-info }
> Write $n = \dim V$. The number of chains $k$, the index of nilpotency $p$, and $n$ satisfy
>
> $$1 \le k \le n, \qquad 1 \le p \le n, \qquad k + p - 1 \le n,$$
>
> with $k + p - 1 = n$ exactly when the diagram is a **hook**: one column of any height, all others of height $1$.

{: .prompt-proof }
> Count boxes. There are $n$ in total, $k$ columns each with at least one, and a tallest column with $p$. That column and one box from each of the other $k - 1$ columns are $p + k - 1$ distinct boxes, with equality exactly when the other columns have no further boxes. $\blacksquare$

{: .prompt-tip }
> There is no inequality between $k$ and $p$ alone: the two hooks $k = 1, p = n$ (a single column) and $k = n, p = 1$ (a single level) are the [two extremes](#two-extremes) below.

{: .prompt-warning }
> Most references draw a partition with its parts as *rows*: the conventional [Young diagram](https://en.wikipedia.org/wiki/Young_tableau#Diagrams) of $\mu$ puts $m_i$ boxes in row $i$, where this array puts them in column $i$. Parts are columns here because $N$ acts down them.

{: .prompt-info }
> For every $j \ge 0$, $\operatorname{null}N^j \subseteq \operatorname{null}N^{j+1}$, and each $\operatorname{null}N^j$ is invariant under $N$.

If $N^ju = 0$, then $N^{j+1}u = N(N^ju) = 0$, which is the inclusion, and $N^j(Nu) = N(N^ju) = 0$, which is the invariance. So the null spaces form a chain of invariant subspaces

$$\{0\} = \operatorname{null}N^0 \subseteq \operatorname{null}N \subseteq \operatorname{null}N^2 \subseteq \dots \subseteq V,$$

with dimensions $d_j := \dim\operatorname{null}N^j$, so $d_0 = 0$. The diagram computes them.

{: .prompt-info }
> For every $j \ge 0$, the boxes in the bottom $j$ levels form a basis of $\operatorname{null}N^j$, and
>
> $$d_j = \mu'_1 + \dots + \mu'_j = \sum_i \min(j, m_i), \qquad \dim\operatorname{range}N^j = \sum_i \max(0, m_i - j).$$
>
> Consequently, for every $j \ge 1$,
>
> $$\mu'_j = d_j - d_{j-1}, \qquad \#\{\,i : m_i = j\,\} = \mu'_j - \mu'_{j+1} = 2d_j - d_{j-1} - d_{j+1}.$$

{: .prompt-proof }
> $N^j$ kills every box in the bottom $j$ levels and moves every other box $j$ steps down its column, to a different box. The images of the boxes above level $j$ are therefore distinct basis vectors, hence independent, so a vector is killed by $N^j$ exactly when its coordinates on those boxes vanish. This proves the basis claim, and counting the boxes level by level gives $d_j = \mu'_1 + \dots + \mu'_j$.
>
> Counting the same boxes column by column instead: column $i$ occupies levels $1$ through $m_i$, so it contributes $\min(j, m_i)$ boxes to the bottom $j$ levels, all $m_i$ of them if $m_i \le j$ and only $j$ if $m_i > j$. Summing gives $d_j = \sum_i \min(j, m_i)$. By the fundamental theorem of linear maps, $\dim\operatorname{range}N^j = \dim V - d_j$ counts the remaining boxes, $\max(0, m_i - j)$ in column $i$.
>
> Differencing the partial sums gives $$\mu'_j = d_j - d_{j-1}$$. A column has height exactly $j$ when it reaches level $j$ but not level $j+1$, so differencing once more counts the columns of height $j$. $\blacksquare$

The formula $$\mu'_j = d_j - d_{j-1}$$ is how the Weyr characteristic is usually defined, and the last formula recovers $\mu$ from the dimensions $d_j$ by pure arithmetic.

{: .prompt-warning }
> From here on, $j$ indexes both a level and a power of $N$. Levels stop at $p$; powers do not. For $j > p$ there is no level $j$, so $\mu'_j = 0$, and $\operatorname{null}N^j = V$, so $d_j = \dim V$.

{: .prompt-tip }
> Since the number of columns of height $j$ is never negative, $(d_j)$ is **concave**:
>
> $$2d_j \ge d_{j-1} + d_{j+1}.$$
>
> Equivalently, the increments $\mu'_j$ of $d$ never increase: in the diagram, no level is wider than the one below it.

**Example**

Let $V = \operatorname{span}(1,\, x,\, y,\, x^2,\, xy)$ and $N = \partial/\partial x$, which sends

$$1 \mapsto 0,\quad x \mapsto 1,\quad y \mapsto 0,\quad x^2 \mapsto 2x,\quad xy \mapsto y.$$

The chains $x^2 \mapsto 2x \mapsto 2$ and $xy \mapsto y$ form a Jordan basis (this example recurs through the rest of the post):

$$\begin{array}{cc} x^2 & \\ 2x & xy \\ 2 & y \end{array}$$

The columns have heights $3$ and $2$, so $\mu = (3,2)$. The levels, read from the bottom up, have sizes $2$, $2$, $1$, so $\mu' = (2,2,1)$. Both sum to $5 = \dim V$, since they count the same five boxes.

Partial sums give $(d_1,d_2,d_3) = (2,4,5)$, matching $\operatorname{null}N = \operatorname{span}(1,y)$, $\operatorname{null}N^2 = \operatorname{span}(1,y,x,xy)$, and $\operatorname{null}N^3 = V$.

{: .prompt-info }
> Let $U_i$ be the span of the $i$-th Jordan chain. Then $V = U_1 \oplus \dots \oplus U_k$, each $U_i$ is invariant under $N$, and with respect to a suitable ordering of each chain,
>
> $$\mathcal{M}(N) = J_{m_1}(0) \oplus \dots \oplus J_{m_k}(0).$$

{: .prompt-proof }
> The sum is direct because the chains together form a basis of $V$. Each $U_i$ is invariant because $N$ sends each chain vector to the next one, or to $0$. Order the chain **bottom to top**:
>
> $$u_1 = N^{m_i-1}v_i,\quad u_2 = N^{m_i-2}v_i,\quad \dots,\quad u_{m_i} = v_i.$$
>
> Then $Nu_1 = N^{m_i}v_i = 0$ and $Nu_j = u_{j-1}$ for $j \ge 2$. Column $j$ of a matrix records the image of the $j$-th basis vector, so the $1$ from $Nu_j = u_{j-1}$ sits in row $j-1$, column $j$, on the superdiagonal:
>
> $$\mathcal{M}\big(N|_{U_i}, (u_1,\dots,u_{m_i})\big) = \begin{pmatrix} 0&1& & \\ &0&\ddots& \\ & &\ddots&1\\ & & &0\end{pmatrix} = J_{m_i}(0).$$
>
> With respect to the basis formed by concatenating these ordered chains, in the order $U_1,\dots,U_k$, the matrix of $N$ is therefore block diagonal with blocks $J_{m_1}(0), \dots, J_{m_k}(0)$. $\blacksquare$

{: .prompt-info }
> Let $N$ be nilpotent on $V$. The partition $\mu$ does not depend on which Jordan basis produced it, and two nilpotent operators are [similar]({% post_url 2026-06-11-linear-algebra %}#matrices) if and only if their partitions agree.

{: .prompt-proof }
> **Independence of the basis.** The level sizes of the diagram are $$\mu'_j = d_j - d_{j-1}$$, and $$d_j = \dim\operatorname{null}N^j$$ makes no reference to a basis. So $\mu'$ is determined by $N$ alone, and therefore so is its conjugate $\mu$. Different Jordan bases genuinely differ in their vectors, but the shape those vectors fill is fixed before any choice is made. In matrix terms, any two Jordan bases give the same block diagonal matrix up to the order of the blocks, and with the chains sorted as above, the very same matrix.
>
> **Similar operators agree.** If $N' = C^{-1}NC$, the inner factors telescope: $(N')^j = C^{-1}N^jC$. So for $v \in \operatorname{null}N^j$,
>
> $$(N')^j\big(C^{-1}v\big) = C^{-1}N^jCC^{-1}v = C^{-1}N^jv = 0,$$
>
> putting $C^{-1}v \in \operatorname{null}(N')^j$. So $C^{-1}$ maps $\operatorname{null}N^j$ into $\operatorname{null}(N')^j$, injectively because $C$ is invertible, and onto because $N = CN'C^{-1}$ runs the same argument backwards. Hence the two operators have the same $d_j$ for every $j$, so the same $\mu'$, so the same $\mu$.
>
> **Agreeing operators are similar.** Each of $N$ and $N'$ has a Jordan basis in which its matrix is $J_{m_1}(0)\oplus\dots\oplus J_{m_k}(0)$. Equal $\mu$ makes these the same matrix, so $N$ and $N'$ have equal matrices with respect to their two Jordan bases, which is similarity. $\blacksquare$

### Jordan Block

{: .prompt-info }
> $\dim\operatorname{null}\big(J_m(\lambda) - \lambda I\big) = 1$.

Indeed $\operatorname{null}\big(J_m(\lambda) - \lambda I\big) = \operatorname{null}J_m(0)$, and $J_m(0)$ is a single chain, so its null space has dimension $d_1 = \mu'_1 = 1$.

#### Nilpotent Block

Let $N = J_m(0)$. With each power, the band of $1$s moves one step toward the upper right and loses one entry. With $m = 5$:

$$N = \begin{pmatrix}0&1&0&0&0\\ &0&1&0&0\\ & &0&1&0\\ & & &0&1\\ & & & &0\end{pmatrix},\quad N^2 = \begin{pmatrix}0&0&1&0&0\\ &0&0&1&0\\ & &0&0&1\\ & & &0&0\\ & & & &0\end{pmatrix},\quad N^3 = \begin{pmatrix}0&0&0&1&0\\ &0&0&0&1\\ & &0&0&0\\ & & &0&0\\ & & & &0\end{pmatrix}$$

Then $N^4$ has a single $1$ in the top-right corner, and $N^5 = 0$.

In general, write $u_1, \dots, u_m$ for the standard basis, so that $Nu_i = u_{i-1}$ with $u_i := 0$ for $i \le 0$. Then $N^ju_i = u_{i-j}$: column $i$ of $N^j$ has its $1$ in row $i - j$, on the $j$-th superdiagonal, and the columns $i \le j$ are zero. The remaining $\max(0, m-j)$ columns are distinct standard basis vectors, so

$$\dim\operatorname{null}N^j = \min(j, m), \qquad \dim\operatorname{range}N^j = \max(0, m-j).$$

This is the single-chain case of $d_j = \sum_i\min(j,m_i)$ and $\dim\operatorname{range}N^j = \sum_i\max(0,m_i-j)$: the band leaving through the corner is $N^j$ killing the bottom $\min(j,m)$ boxes of a column of height $m$.

#### Non-nilpotent Block

If $\lambda \neq 0$, write $A = J_m(\lambda) = \lambda I + N$. The two terms commute, so the binomial theorem applies, and $N^t = 0$ for $t \ge m$:

$$A^j = \sum_{t=0}^{\min(j,\,m-1)} \binom{j}{t}\lambda^{j-t}N^t.$$

Rather than shifting a single band, this **fills the entire upper triangle**, with $\lambda^j$ on the diagonal, $j\lambda^{j-1}$ on the first superdiagonal, $\binom{j}{2}\lambda^{j-2}$ on the second, and so on. Each diagonal is constant, so $A^j$ is upper triangular [Toeplitz](https://en.wikipedia.org/wiki/Toeplitz_matrix). For $m = 4$:

$$A^j = \begin{pmatrix} \lambda^j & j\lambda^{j-1} & \binom{j}{2}\lambda^{j-2} & \binom{j}{3}\lambda^{j-3}\\ & \lambda^j & j\lambda^{j-1} & \binom{j}{2}\lambda^{j-2}\\ & & \lambda^j & j\lambda^{j-1}\\ & & & \lambda^j \end{pmatrix}$$

Two structural differences from the nilpotent case: $A^j$ is never $0$, since $\det A^j = \lambda^{jm} \neq 0$, so $J_m(\lambda)$ is invertible rather than nilpotent; and the shifting behaviour is recovered only through $A - \lambda I = N$. This is why the general Jordan form is proved by subtracting $\lambda$ on each generalized eigenspace and working with the nilpotent part.

#### Inverse of a Jordan Block

{: .prompt-info }
> Let $W$ be a vector space over $\mathbf{F}$, let $N \in \mathcal{L}(W)$ be nilpotent with $N^p = 0$, and let $\lambda \in \mathbf{F}$ with $\lambda \neq 0$. Then $\lambda I + N$ is invertible, and
>
> $$(\lambda I+N)^{-1} \;=\; \sum_{t=0}^{p-1} \frac{(-1)^t}{\lambda^{\,t+1}}\,N^t .$$

{: .prompt-proof }
> Set $B = -\lambda^{-1}N$. Since scalars commute with everything, $B^p = (-\lambda^{-1})^pN^p = 0$, and
>
> $$\lambda I + N = \lambda\left(I - B\right).$$
>
> Let $Q = \sum_{t=0}^{p-1}B^t$, a finite sum. The product telescopes:
>
> $$(I-B)Q \;=\; \sum_{t=0}^{p-1}B^t \;-\; \sum_{t=0}^{p-1}B^{t+1} \;=\; \sum_{t=0}^{p-1}B^t \;-\; \sum_{t=1}^{p}B^{t} \;=\; B^0 - B^p \;=\; I,$$
>
> and the same computation with the factors reversed gives $Q(I-B) = I$, since $B$ commutes with its own powers. So $I - B$ is invertible with inverse $Q$, hence $\lambda I + N = \lambda(I-B)$ is invertible with inverse $\lambda^{-1}Q$. Expanding $B$:
>
> $$\lambda^{-1}Q \;=\; \lambda^{-1}\sum_{t=0}^{p-1}\left(-\lambda^{-1}\right)^tN^t \;=\; \sum_{t=0}^{p-1}\frac{(-1)^t}{\lambda^{\,t+1}}N^t. \qquad \blacksquare$$

{: .prompt-tip }
> The same computation, run as an infinite series, is the [Neumann series](https://en.wikipedia.org/wiki/Neumann_series): in a normed setting, such as operators on a finite-dimensional space over $\mathbb{R}$ or $\mathbb{C}$, $I - B$ is invertible with inverse $\sum_{t \ge 0} B^t$ whenever this series converges, for example when $\lVert B \rVert < 1$. Here no convergence is needed, because $B^t = 0$ for $t \ge p$: the series is a finite sum, which makes sense over any field and for any size of $B$.

{: .prompt-tip }
> *Consistency with the power formula.* For $A = J_m(\lambda)$, the formula $A^j = \sum_{t}\binom{j}{t}\lambda^{j-t}N^t$ above comes from the binomial theorem, so it is proved only for $j \ge 0$. Substituting $j = -1$ anyway, with [$\binom{-1}{t} = (-1)^t$](https://en.wikipedia.org/wiki/Binomial_coefficient#Generalization_to_negative_integers_n) and the sum running over $t \le m-1$ because $N^t = 0$ for $t \ge m$, predicts
>
> $$A^{-1} = \sum_{t=0}^{m-1}(-1)^t\lambda^{-1-t}N^t.$$
>
> This prediction is exactly the theorem above with $p = m$, so it is correct. In matrix terms, $A^{-1}$ is upper-triangular Toeplitz like the nonnegative powers, with $(-1)^t\lambda^{-1-t}$ on the $t$-th superdiagonal.

**Worked example**

Take $N = J_3(0)$ on $\mathbf{F}^3$, so $p = 3$ and $\lambda I + N = J_3(\lambda)$:

$$J_3(\lambda)^{-1} = \frac{1}{\lambda}I - \frac{1}{\lambda^2}N + \frac{1}{\lambda^3}N^2 = \begin{pmatrix} 1/\lambda & -1/\lambda^2 & 1/\lambda^3 \\ 0 & 1/\lambda & -1/\lambda^2 \\ 0 & 0 & 1/\lambda\end{pmatrix}.$$

Multiplying out confirms it, with $N^3 = 0$:

$$(\lambda I+N)\left(\tfrac{1}{\lambda}I - \tfrac{1}{\lambda^2}N + \tfrac{1}{\lambda^3}N^2\right) = I - \tfrac{1}{\lambda}N + \tfrac{1}{\lambda^2}N^2 + \tfrac{1}{\lambda}N - \tfrac{1}{\lambda^2}N^2 + \tfrac{1}{\lambda^3}N^3 = I.$$

The inverse is again upper-triangular Toeplitz, with alternating signs and increasing powers of $1/\lambda$ along successive diagonals.

### Find a Jordan Basis

{: .prompt-info }
> **Input.** A nilpotent $N \in \mathcal{L}(V)$.
>
> **Precompute.** The increasing chain of null spaces and their dimensions,
>
> $$\{0\} = K_0 \subsetneq K_1 \subsetneq \dots \subsetneq K_p = V, \qquad K_j := \operatorname{null}N^j, \quad d_j := \dim K_j,$$
>
> up to the index of nilpotency $p$. Keep an explicit basis of each $K_j$.
>
> Recall that $$\mu'_j = d_j - d_{j-1}$$ is the size of level $j$. The algorithm builds the diagram level by level from the top down, so this is the number of vectors it must produce at level $j$.
>
> **State carried between levels.** As the loop runs it carries a list $H_j$ of vectors in $K_j$, pushed down from the level above. At each level it also outputs a list $T_j$ of *new chain tops*.
>
> ---
>
> **Step 0 (initialize).** Set $j := p$ and $H_p := \varnothing$.
>
> **Step 1 (choose new tops at level $j$).** Form one list, in this order:
>
> 1. a basis of $K_{j-1}$,
> 2. the vectors of $H_j$,
> 3. any spanning set of $K_j$.
>
> Extend the first two groups to a basis of $K_j$ using vectors from the third. The vectors taken from group 3 are $T_j$. (In practice: put the list in as columns of a matrix and row reduce; the pivot columns from group 3 are $T_j$.)
>
> Each $v \in T_j$ is the top of a new chain of length exactly $j$:
>
> $$v,\; Nv,\; \dots,\; N^{j-1}v.$$
>
> Count check: $\lvert H_j \rvert + \lvert T_j \rvert = \mu'_j$. Every vector of $H_j$ should survive as a pivot; if one doesn't, there is an arithmetic error upstream.
>
> **Step 2 (push down).** Set
>
> $$H_{j-1} := \{\,Nw \;:\; w \in H_j \cup T_j\,\}.$$
>
> **Step 3 (loop).** Set $j := j-1$. If $j \ge 1$, go to Step 1. If $j = 0$, stop.
>
> At $j = 1$, group 1 is empty ($K_0 = \{0\}$), so Step 1 is ordinary extension of $H_1$ to a basis of $\operatorname{null}N$.
>
> **Output.** For each $v \in T_j$ (over all $j$), the chain $v, Nv, \dots, N^{j-1}v$. Together these chains form a Jordan basis.
>
> **Verification.** There should be $\lvert T_j \rvert = 2d_j - d_{j-1} - d_{j+1}$ chains of length $j$, and $\sum_j j\,\lvert T_j \rvert = \dim V$ vectors in total.

![Find-a-jordan-basis](/assets/img/math/jordan_chain_algorithm_levels_and_invariant.png)

{: .prompt-proof }
> **Step 1 in quotient terms.** Let $\pi_j : K_j \to K_j/K_{j-1}$ be the quotient map. Step 1 extends $\pi_j(H_j)$ to a basis of $K_j/K_{j-1}$ and lifts the added cosets to vectors; the append-and-extend recipe performs this computation without forming the quotient. Different lifts give different, equally valid Jordan bases.
>
> **The engine.** For $j \ge 2$ the map
>
> $$\bar N_j : K_j/K_{j-1} \longrightarrow K_{j-1}/K_{j-2}, \qquad v + K_{j-1} \longmapsto Nv + K_{j-2},$$
>
> is well defined ($v \in K_j \Rightarrow Nv \in K_{j-1}$, and $v \in K_{j-1} \Rightarrow Nv \in K_{j-2}$) and injective ($Nv \in K_{j-2}$ means $N^{j-1}v = 0$, i.e. $v \in K_{j-1}$). The restriction $j \ge 2$ ensures $K_{j-2}$ exists; the smallest case is $\bar N_2 : K_2/K_1 \to K_1$. By construction, the square
>
> $$\begin{array}{ccc} K_j & \xrightarrow{\ \ N\ \ } & K_{j-1} \\[2pt] \big\downarrow{\scriptstyle \pi_j} & & \big\downarrow{\scriptstyle \pi_{j-1}} \\[2pt] K_j/K_{j-1} & \xrightarrow{\ \bar N_j\ } & K_{j-1}/K_{j-2} \end{array}$$
>
> commutes: for $w \in K_j$,
>
> $$\pi_{j-1}(Nw) = Nw + K_{j-2} = \bar N_j\big(w + K_{j-1}\big) = \bar N_j\big(\pi_j(w)\big).$$
>
> Injectivity of $\bar N_{j+1}$ alone gives $\mu'_{j+1} \le \mu'_j$, a second proof that no level is wider than the one below it.
>
> **Loop invariant.** Step 1 at level $j$ can only run if $H_j \subseteq K_j$ and $\pi_j(H_j)$ is independent in $K_j/K_{j-1}$, since a dependent list cannot be extended to a basis. Step 2 re-establishes this one level down. Applying the commuting square to each vector of $H_j \cup T_j \subseteq K_j$,
>
> $$\pi_{j-1}(H_{j-1}) = \pi_{j-1}\big(N(H_j \cup T_j)\big) = \bar N_j\big(\pi_j(H_j \cup T_j)\big),$$
>
> where $\pi_j(H_j \cup T_j)$ is a *basis* of $K_j/K_{j-1}$ by Step 1 and $\bar N_j$ is injective. So:
>
> - $H_{j-1} \subseteq K_{j-1}$, since $w \in K_j \Rightarrow N^{j-1}(Nw) = N^jw = 0$.
> - $\pi_{j-1}(H_{j-1})$ is independent in $K_{j-1}/K_{j-2}$, since injective maps preserve independence. In particular pushing down neither merges two chains nor loses one: $\lvert H_{j-1}\rvert = \lvert H_j \cup T_j\rvert = \mu'_j$.
>
> This is exactly the precondition of Step 1 at level $j-1$, so nothing carried is ever discarded. At $j = p$ the precondition holds vacuously, since $H_p = \varnothing$. At $j = 1$ Step 2 pushes everything to $0$, but Step 3 stops before $H_0$ is used.
>
> **Chains have the advertised length.** Each $v \in T_j$ has $\pi_j(v) \neq 0$, i.e. $v \notin K_{j-1}$, so $N^{j-1}v \neq 0$ while $N^jv = 0$. Passing to the quotient is what rules out a chain dying early; plain linear independence would not.
>
> **The chains form a basis.** Every chain vector lies in exactly one list $H_j \cup T_j$: either it is a new top, or it was pushed down to that level. The vectors of $H_j \cup T_j$ lie in $K_j$ and lift a basis of $K_j/K_{j-1}$, so together with a basis of $K_{j-1}$ they form a basis of $K_j$. By induction on $j$, the chain vectors at levels $1, \dots, j$ form a basis of $K_j$. At $j = p$ this is a basis of $V$.
>
> **The counts.** Step 1 makes $\pi_j(H_j \cup T_j)$ a basis of $K_j/K_{j-1}$, so $\lvert H_j \cup T_j\rvert = \mu'_j$, and the push-down count gives $\lvert H_j\rvert = \mu'_{j+1}$. Hence
>
> $$\lvert T_j\rvert = \mu'_j - \mu'_{j+1} = 2d_j - d_{j-1} - d_{j+1},$$
>
> with the convention $\mu'_{p+1} = 0$. Each top in $T_j$ starts a chain of length $j$, so this is the number of columns of height $j$, as the diagram predicts. $\blacksquare$

{: .prompt-tip }
> *The new tops span a cokernel.* Step 2 one level up sets $H_j = N(H_{j+1} \cup T_{j+1})$, so by the commuting square $\pi_j(H_j) = \bar N_{j+1}\big(\pi_{j+1}(H_{j+1} \cup T_{j+1})\big)$. The list inside is a basis of $K_{j+1}/K_j$ and $\bar N_{j+1}$ is injective, so $\pi_j(H_j)$ is a basis of $\operatorname{range}\bar N_{j+1}$. Step 1 extends it to a basis of $K_j/K_{j-1}$, so $\pi_j(T_j)$ represents a basis of
>
> $$\big(K_j/K_{j-1}\big)\big/\operatorname{range}\bar N_{j+1} = \operatorname{coker}\bar N_{j+1},$$
>
> of dimension $$\mu'_j - \mu'_{j+1} = \lvert T_j\rvert$$. Level $j+1$ sends $$\mu'_{j+1}$$ independent directions down into level $j$, level $j$ has $$\mu'_j$$ directions to fill, and the new tops fill the shortfall: the cokernel is the part of level $j$ that nothing above it reaches. At $j = p$ there is no level above, so read $\operatorname{range}\bar N_{p+1} = \{0\}$.

{: .prompt-warning }
> Do not start from an arbitrary basis of $\operatorname{null}N$ and solve upward. Most vectors of $\operatorname{null}N$ are not in $\operatorname{range}N$, so their preimages may not exist. A bottom-up method works only if the bottom of each chain of length $j$ is chosen inside $\operatorname{null}N \cap \operatorname{range}N^{j-1}$, which requires computing ranges as well. Top-down avoids this: low-level vectors are *manufactured* by pushing down, never guessed.

**Worked example**

Return to $N = \partial/\partial x$ on $V = \operatorname{span}(1,\, x,\, y,\, x^2,\, xy)$. Precompute: $\operatorname{null}N = \operatorname{span}(1, y)$, $\operatorname{null}N^2 = \operatorname{span}(1,y,x,xy)$, $\operatorname{null}N^3 = V$. So $p = 3$ and $(d_0,d_1,d_2,d_3) = (0,2,4,5)$, which already forces $\mu' = (2,2,1)$ and $\mu = (3,2)$. The algorithm finds vectors filling this shape.

| $j$ | $H_j$ (pushed down) | need $d_j - d_{j-1}$ | new tops $T_j$ |
| --- | ------------------- | -------------------- | -------------- |
| $3$ | $\varnothing$       | $5-4 = 1$            | $\{x^2\}$      |
| $2$ | $\{2x\}$            | $4-2 = 2$            | $\{xy\}$       |
| $1$ | $\{2,\; y\}$        | $2-0 = 2$            | $\varnothing$  |

Reading the levels:

- **$j=3$:** nothing carried in. Need $1$ vector of $\operatorname{null}N^3 = V$ outside $\operatorname{null}N^2$; take $x^2$. Push down: $H_2 = \{2x\}$.
- **$j=2$:** carrying $2x$. Need $2$ vectors total, independent modulo $\operatorname{null}N = \operatorname{span}(1,y)$; $2x$ supplies one, so add one more from $\operatorname{null}N^2 = \operatorname{span}(1,y,x,xy)$; take $xy$. Push down: $H_1 = \{N(2x), N(xy)\} = \{2, y\}$.
- **$j=1$:** carrying $\{2,y\}$, which is already a basis of the $2$-dimensional $\operatorname{null}N$. Nothing to add.

Chains: $x^2 \mapsto 2x \mapsto 2 \mapsto 0$ and $xy \mapsto y \mapsto 0$. Ordering each chain bottom to top gives the Jordan basis

$$(2,\; 2x,\; x^2,\; y,\; xy),$$

with $\mathcal{M}(N) = J_3(0)\oplus J_2(0)$. Checks: $\lvert T_3 \rvert = 1 = 2(5)-4-5$, $\lvert T_2 \rvert = 1 = 2(4)-2-5$, $\lvert T_1 \rvert = 0 = 2(2)-0-4$, and $3 + 2 = 5 = \dim V$.

{: .prompt-tip }
> *The free choices.* Up to scaling, the top at $j = 3$ can be any $x^2 + q$ with $q \in \operatorname{null}N^2$, and the new top at $j = 2$ any $xy + cx + r$ with $c \in \mathbf{F}$ and $r \in \operatorname{null}N$. The chains become
>
> $$x^2 + q \mapsto 2x + Nq \mapsto 2 \qquad\text{and}\qquad xy + cx + r \mapsto y + c.$$
>
> So not only the basis moves but the invariant subspaces spanned by the chains: for $c \ne 0$, the second chain spans $\operatorname{span}(xy + cx + r,\, y + c) \ne \operatorname{span}(xy, y)$. Only the shape $(3,2)$ is forced.

**Row reduction at level $j = 2$**

Coordinates are with respect to the ordered basis $(1, x, y, x^2, xy)$.

The list is: group 1 is $1, y$; group 2 is $2x$; group 3 is $1, y, x, xy$. As columns,

$$\left(\begin{array}{ccc|cccc} 1 & 0 & 0 & 1 & 0 & 0 & 0\\ 0 & 0 & 2 & 0 & 0 & 1 & 0\\ 0 & 1 & 0 & 0 & 1 & 0 & 0\\ 0 & 0 & 0 & 0 & 0 & 0 & 0\\ 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{array}\right)$$

Reducing (here only row swaps are needed) puts pivots in columns $1, 2, 3, 7$. Columns $4$ and $5$ repeat group 1, and column $6$ is a scalar multiple of column $3$, so all three are non-pivots. The single pivot inside group 3 is column $7$, giving

$$T_2 = \{xy\},$$

which matches what was found by hand. The count check holds: $\lvert H_2\rvert + \lvert T_2\rvert = 1 + 1 = 2 = \mu'_2$.

{: .prompt-tip }
> - Full reduced echelon form is unnecessary. Forward elimination to echelon form already reveals the pivot positions, which is all you need.
> - Group 3 can be any spanning set of $\operatorname{null}N^j$; a basis is the convenient choice, and duplicates cost nothing since they simply fail to be pivots.
> - The same elimination produces the null spaces in the first place, by solving $N^j x = 0$, so you can compute all of $\operatorname{null}N, \dots, \operatorname{null}N^p$ with the same tool before the loop starts.

### Two Extremes

Two degenerate shapes sit at opposite ends: a diagram that is a single column, and a diagram that is a single level. Before working them out, one more quantity is needed.

{: .prompt-info }
> *Freedom in a Jordan basis.* The shape of the diagram is forced by $N$; the basis filling it is not. A Jordan basis, listed in a fixed order, is a point of $V^n$, which has dimension $n^2$. In both cases below, the Jordan bases form a nonempty open subset of a linear subspace of $V^n$, namely the points of that subspace where some polynomial does not vanish. Write $r$ for the dimension of that subspace.

#### One Column: a Single Chain

$\mu = (n)$ and $\mu' = (1,\dots,1)$. There is one chain of length $n$: a single top, a single bottom, and every level of size $1$.

{: .prompt-info }
> $$k = \mu'_1 = 1, \qquad p = m_1 = n, \qquad d_j = \min(j,n), \qquad \dim\operatorname{range}N^j = \max(0, n-j),$$
>
> and $\mathcal{M}(N) = J_n(0)$ is a single Jordan block.

This is as far from the zero operator as a nilpotent operator gets: $N^{n-1} \neq 0$, and the null spaces

$$\{0\} \subsetneq \operatorname{null}N \subsetneq \operatorname{null}N^2 \subsetneq \dots \subsetneq \operatorname{null}N^n = V$$

climb one dimension at a time. For any nilpotent $N$, each increment $\mu'_j$ with $j \le p$ is at least $1$, so $d_j \ge \min(j, n)$; one column is the case of equality, the slowest possible climb.

{: .prompt-tip }
> *Freedom.* A Jordan basis here is determined by its top $v$ alone, since the rest of the chain, $Nv, N^2v, \dots$, is forced. The linear map $v \mapsto (N^{n-1}v, \dots, Nv, v)$ identifies the Jordan bases with the $v$ satisfying $N^{n-1}v \neq 0$, so $r = n$, against $n^2$ for an unconstrained list.

{: .prompt-info }
> *Invariant subspaces.* The only subspaces invariant under $N$ are the $n+1$ subspaces $\operatorname{null}N^j$ for $0 \le j \le n$: a single chain of them, totally ordered by inclusion. This is the most rigid an invariant-subspace lattice can be.

{: .prompt-proof }
> We prove by induction on $n$: for every nilpotent operator $N$ on an $n$-dimensional space $V$ whose diagram is a single column, the invariant subspaces are exactly $\operatorname{null}N^0, \operatorname{null}N^1, \dots, \operatorname{null}N^n$. These $n+1$ subspaces are invariant, and they are distinct because $d_j = j$; what needs proof is that there are no others.
>
> For $n = 1$, $N$ is $0$ on a line, whose only subspaces are $\\{0\\} = \operatorname{null}N^0$ and $V = \operatorname{null}N^1$.
>
> Let $n \ge 2$, and let $W$ be invariant under $N$. If $W = \\{0\\}$, then $W = \operatorname{null}N^0$, so assume $W \ne \\{0\\}$. Write $v$ for the top, so $v, Nv, \dots, N^{n-1}v$ is a basis of $V$, and let
>
> $$U := \operatorname{null}N = \operatorname{span}(N^{n-1}v),$$
>
> a line since $d_1 = 1$. Let $\pi : V \to V/U$ be the quotient map. As in [Subspaces containing $U$]({% post_url 2026-07-04-quotient-spaces %}#quotient-space), $\pi^{-1}(\cdot)$ denotes a preimage, not an inverse map: $\pi$ is not invertible, since $\operatorname{null}\pi = U \ne \\{0\\}$. The plan: show $U \subseteq W$, apply the induction hypothesis to $W/U$, and recover $W$ as the preimage $\pi^{-1}(W/U)$.
>
> **Step 1: $U \subseteq W$.** Pick $0 \ne w \in W$ and take the largest $t$ with $N^tw \ne 0$. Then $N^tw \in W$ by invariance and $N(N^tw) = 0$, so $W \cap U \ne \\{0\\}$. Since $U$ is a line, $U \subseteq W$.
>
> **Step 2: $N/U$ is a single column of height $n-1$.** $U$ is invariant under $N$, so the [quotient operator]({% post_url 2026-07-04-quotient-spaces %}#quotient-operator) $N/U \in \mathcal{L}(V/U)$ is defined, and applying its definition $j$ times gives $(N/U)^j(x + U) = N^jx + U$. The quotient map is linear and surjective, so it sends the basis $v, Nv, \dots, N^{n-1}v$ of $V$ to a spanning list $v + U, Nv + U, \dots, N^{n-1}v + U$ of $V/U$. Its last vector is $N^{n-1}v + U = (N/U)^{n-1}(v + U)$, which is $0$ because $N^{n-1}v \in U$. Dropping it leaves the spanning list $v + U, \dots, N^{n-2}v + U$ of $n - 1 = \dim V/U$ vectors, which is therefore a basis. So $v + U$ is the top of a single chain of length $n-1$ for $N/U$.
>
> **Step 3: $W/U$ is a null space of a power of $N/U$.** $W/U$ is invariant under $N/U$: each of its elements is $w + U$ with $w \in W$, and $(N/U)(w + U) = Nw + U$ with $Nw \in W$. By Step 2, the induction hypothesis applies to $N/U$ on the $(n-1)$-dimensional space $V/U$, so
>
> $$W/U = \operatorname{null}(N/U)^i \qquad \text{for some } 0 \le i \le n-1.$$
>
> **Step 4: take preimages.** For every $x \in V$,
>
> $$x \in \pi^{-1}\big(\operatorname{null}(N/U)^i\big) \iff N^ix + U = U \iff N^ix \in U = \operatorname{null}N \iff N^{i+1}x = 0,$$
>
> so $\pi^{-1}\big(\operatorname{null}(N/U)^i\big) = \operatorname{null}N^{i+1}$. Since $U \subseteq W$ by Step 1, [Subspaces containing $U$]({% post_url 2026-07-04-quotient-spaces %}#quotient-space) gives $W = \pi^{-1}(W/U)$. Combining with Step 3,
>
> $$W = \pi^{-1}(W/U) = \pi^{-1}\big(\operatorname{null}(N/U)^i\big) = \operatorname{null}N^{i+1}. \qquad \blacksquare$$

{: .prompt-tip }
> *Cyclic operators.* An operator like this one, for which some vector $v$ makes $v, Nv, \dots, N^{n-1}v$ a basis of $V$, is called **cyclic** (or **nonderogatory**). Equivalently, its minimal polynomial ($z^n$ here) equals its characteristic polynomial.

#### One Level: the Zero Operator

$\mu = (1,\dots,1)$ and $\mu' = (n)$. Now there are $n$ chains, each of length $1$, so every box is both a top and a bottom.

{: .prompt-info }
> $$k = \mu'_1 = n, \qquad p = m_1 = 1, \qquad d_j = n \ \text{for all } j \ge 1, \qquad \dim\operatorname{range}N^j = 0,$$
>
> and $\mathcal{M}(N)$ is the zero matrix.

Here $p = 1$ says $N = 0$: this shape *is* the zero operator, and every null space is already $V$.

{: .prompt-tip }
> *Freedom.* All of it. Since $N = 0$ the chain condition is vacuous, so *every* basis of $V$ is a Jordan basis, and the only condition left is independence. The subspace is all of $V^n$, so $r = n^2$, the largest possible.

Every subspace of $V$ is invariant, again since $N = 0$. This is the least rigid an invariant-subspace lattice can be, the opposite of the totally ordered chain above.

This is the shape of an eigenvalue $\lambda$ of $T$ exactly when $\lambda$ is not defective; see [Eigenvalues and Eigenvectors]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#the-nilpotent-part).

#### Comparing the Ends

|                               | one column     | one level             |
| ----------------------------- | -------------- | --------------------- |
| $\mu$                         | $(n)$          | $(1^n)$               |
| $\mu'$                        | $(1^n)$        | $(n)$                 |
| chains $k = \mu'_1$           | $1$            | $n$                   |
| index of nilpotency $p = m_1$ | $n$            | $1$                   |
| $d_j$                         | $\min(j, n)$   | $n$ for all $j \ge 1$ |
| $\dim\operatorname{range}N^j$ | $\max(0, n-j)$ | $0$ for all $j \ge 1$ |
| $\mathcal{M}(N)$              | $J_n(0)$       | the zero matrix       |
| invariant subspaces           | $n+1$, a chain | all of them           |
| Jordan-basis freedom $r$      | $n$            | $n^2$                 |

The two values of $r$ came from unrelated arguments, one parametrizing a single top vector, the other observing that no condition applies at all. [Commuting Operators]({% post_url 2026-08-01-commuting-operators %}) explains why they belong to one story. The operators commuting with $N$ form a space $\mathcal{C}(N)$ of dimension $\sum_j (\mu'_j)^2$, which is $n$ for one column and $n^2$ for one level, the smallest and largest values it can take. The invertible elements of $\mathcal{C}(N)$ act simply transitively on the Jordan bases, so $r = \dim\mathcal{C}(N)$ always.

## What Next

[Eigenvalues and Eigenvectors]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#the-nilpotent-part) applies the diagram to an arbitrary operator $T$, one generalized eigenspace at a time: for each eigenvalue $\lambda$, the nilpotent operator is $T - \lambda I$ restricted to $G(\lambda,T)$.

This post extracted from $N$ its diagram, which determines $N$ up to similarity. [Commuting Operators]({% post_url 2026-08-01-commuting-operators %}) shows that the same diagram controls which *other* operators commute with $N$, and the two extremes above are the two extremes there as well.
