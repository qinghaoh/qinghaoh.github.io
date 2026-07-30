---
title:  "Nilpotent Operators"
category: math
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Jordan Basis

{: .prompt-info }
> Every nilpotent operator has a Jordan basis.

{: .prompt-tip }
> Let $N \in \mathcal{L}(V)$ be nilpotent. Fix a Jordan basis: there are vectors $v_1,\dots,v_k$ (the **tops**) and lengths $m_1,\dots,m_k$ with $m_1+\dots+m_k = \dim V$ such that
>
> $$\big\{\, N^j v_i \;:\; i = 1,\dots,k,\;\; j = 0,\dots,m_i-1 \,\big\}$$
>
> is a basis of $V$, and $N^{m_i}v_i = 0$ for each $i$.
>
> Picture it as a ragged array — one column per chain:
>
> $$\begin{array}{cccc} v_1 & v_2 & \cdots & v_k \\ Nv_1 & Nv_2 & & Nv_k \\ \vdots & \vdots & & \vdots \\ N^{m_1-1}v_1 & N^{m_2-1}v_2 & & N^{m_k-1}v_k \end{array}$$
>
> $N$ moves each entry one step down the column, and off the bottom to $0$.

{: .prompt-tip }
> The ragged array looks like a [Young diagram](https://en.wikipedia.org/wiki/Young_tableau#Diagrams) if columns descend in height.

{: .prompt-info }
> The span of one Jordan chain is one Jordan block. The whole space is the direct sum of the chain-spans.

{: .prompt-tip }
> Let $U_i = \operatorname{span}(v_i, Nv_i, \dots, N^{m_i-1}v_i)$ be the span of one chain. It's invariant under $N$, so $\left. N \right\rvert_{U_i}$ makes sense. Order the chain **bottom to top**:
>
> $$u_1 = N^{m_i-1}v_i,\quad u_2 = N^{m_i-2}v_i,\quad \dots,\quad u_{m_i} = v_i.$$
>
> Then $Nu_1 = N^{m_i}v_i = 0$ and $Nu_j = u_{j-1}$ for $j \ge 2$. Since column $j$ of a matrix records the image of the $j$-th basis vector, the $1$ from $Nu_j = u_{j-1}$ sits in row $j-1$, column $j$ — the superdiagonal:
>
> $$\mathcal{M}(N|_{U_i}) = \begin{pmatrix} 0&1& & \\ &0&\ddots& \\ & &\ddots&1\\ & & &0\end{pmatrix} = J_{m_i}(0).$$
>
> And since $V = U_1\oplus\dots\oplus U_k$ with each $U_i$ invariant, the matrix of $N$ with respect to the concatenated basis is block diagonal:
>
> $$\mathcal{M}(N) = J_{m_1}(0) \oplus \cdots \oplus J_{m_k}(0).$$

{: .prompt-info }
> $\dim\operatorname{null}N^m = \sum_j \min(m, m_j)$.

{: .prompt-proof }
> Since $V = \bigoplus_j U_j$ with each $U_j$ invariant, $\operatorname{null}N^m = \bigoplus_j(\operatorname{null}N^m \cap U_j)$, so count one column at a time. In column $j$, take $x = \sum_{t=0}^{m_j-1} c_t N^tv_j$. Then
>
> $$N^m x = \sum_{t=0}^{m_j-1} c_t N^{t+m}v_j.$$
>
Terms with $t + m \ge m_j$ die on their own (they've fallen off the bottom); the rest are distinct basis vectors, hence independent. So $N^mx = 0$ forces $c_t = 0$ exactly for $t < m_j - m$, leaving
>
> $$t \in \{\max(0,\, m_j - m),\ \dots,\ m_j-1\}, \qquad \text{count } = m_j - \max(0, m_j-m) = \min(m, m_j).$$
>
> $N^m$ annihilates the bottom $m$ entries of a column — but a column of height $m_j$ has only $m_j$ entries to give. Whichever runs out first wins:
>
> - **Short column** ($m_j \le m$): the reach of $N^m$ exceeds the column, so the *whole* column dies — $m_j$ dimensions.
> - **Tall column** ($m_j > m$): only the bottom $m$ entries die — $m$ dimensions.
>
>That's $\min(m, m_j)$, and summing over columns gives $\dim\operatorname{null}N^m$.$\blacksquare$

{: .prompt-tip }
> Let $N$ be nilpotent on $V$ and $d_j = \dim\operatorname{null}N^j$. Every Jordan basis of $N$ decomposes $V$ into chains whose lengths form a multiset $\{m_1,\dots,m_k\}$ with
>
> $$k = d_1, \qquad \#\{i : m_i \ge j\} = d_j - d_{j-1}, \qquad \#\{i : m_i = j\} = 2d_j - d_{j-1} - d_{j+1}.$$
>
> Since the $d_j$ are defined without reference to a basis, the multiset is an invariant of $N$ — a partition of $\dim V$, and a complete invariant up to similarity.

{: .prompt-tip }
> Any two Jordan bases of $N$ give matrices that differ only by a permutation of the diagonal blocks.

### Find a Jordan basis

{: .prompt-info }
> **Input.** A nilpotent $N \in \mathcal{L}(V)$.
>
> **Precompute.** The increasing chain of null spaces and their dimensions,
>
> $$\{0\} = K_0 \subsetneq K_1 \subsetneq \cdots \subsetneq K_p = V, \qquad K_j := \operatorname{null}N^j, \quad d_j := \dim K_j,$$
>
> where $p$ is the smallest exponent with $N^p = 0$. Keep an explicit basis of each $K_j$. Write
>
> $$\delta_j := d_j - d_{j-1}$$
>
> for the jump in dimension at level $j$; it will turn out to be the number of chains of length $\ge j$.
>
> **State carried between levels.** As the loop runs it carries a list $H_j$ of vectors sitting in $K_j$, pushed down from the level above. It also outputs a list $T_j$ of *new chain tops* at each level.
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
> Count check: $\lvert H_j \rvert + \lvert T_j \rvert = \delta_j$. Every vector of $H_j$ should survive as a pivot; if one doesn't, there is an arithmetic error upstream.
>
> **Step 2 (push down).** Set
>
> $$H_{j-1} := \{\,Nw \;:\; w \in H_j \cup T_j\,\}.$$
>
> **Step 3 (loop).** Set $j := j-1$. If $j \ge 1$, go to Step 1. If $j = 0$, stop.
>
> At $j = 1$, group 1 is empty ($K_0 = \{0\}$), so Step 1 is ordinary extension of $H_1$ to a basis of $\operatorname{null}N$.
>
> **Output.** For each $v \in T_j$ (over all $j$), the chain $v, Nv, \dots, N^{j-1}v$. Concatenating all chains gives a Jordan basis.
>
> **Verification.** You should have $\lvert T_j \rvert = 2d_j - d_{j-1} - d_{j+1}$ chains of length $j$, and $\sum_j j\,\lvert T_j \rvert = \dim V$ vectors in total.

![Find-a-jordan-basis](/assets/img/math/jordan_chain_algorithm_levels_and_invariant.png)

{: .prompt-tip }
> **Why it works.** Let $\pi_j : K_j \to K_j/K_{j-1}$ be the quotient map. Step 1 is really "extend $\pi_j(H_j)$ to a basis of $K_j/K_{j-1}$, then lift the added cosets to vectors"; the append-and-extend recipe is exactly that computation done without forming the quotient. Different lifts give different, equally valid Jordan bases.
>
> **The engine.** For $j \ge 2$ the map
>
> $$\bar N_j : K_j/K_{j-1} \longrightarrow K_{j-1}/K_{j-2}, \qquad v + K_{j-1} \longmapsto Nv + K_{j-2},$$
>
> is well defined ($v \in K_j \Rightarrow Nv \in K_{j-1}$, and $v \in K_{j-1} \Rightarrow Nv \in K_{j-2}$) and injective ($Nv \in K_{j-2}$ means $N^{j-1}v = 0$, i.e. $v \in K_{j-1}$). The restriction $j \ge 2$ is just so that $K_{j-2}$ exists; the smallest case is $\bar N_2 : K_2/K_1 \to K_1$.
>
> **The square commutes**.
>
> $$\begin{array}{ccc} K_j & \xrightarrow{\ \ N\ \ } & K_{j-1} \\[2pt] \big\downarrow{\scriptstyle \pi_j} & & \big\downarrow{\scriptstyle \pi_{j-1}} \\[2pt] K_j/K_{j-1} & \xrightarrow{\ \bar N_j\ } & K_{j-1}/K_{j-2} \end{array}$$
>
> For $w \in K_j$:
>
> $$\pi_{j-1}(Nw) = Nw + K_{j-2} = \bar N_j\big(w + K_{j-1}\big) = \bar N_j\big(\pi_j(w)\big).$$
>
> **Loop invariant.** Step 1 at level $j$ can only run if $H_j \subseteq K_j$ and $\pi_j(H_j)$ is independent in $K_j/K_{j-1}$, since a dependent list cannot be extended to a basis. Step 2 is what re-establishes this one level down. Applying the commuting square elementwise to the list $H_j \cup T_j \subseteq K_j$,
>
> $$\pi_{j-1}(H_{j-1}) = \pi_{j-1}\big(N(H_j \cup T_j)\big) = \bar N_j\big(\pi_j(H_j \cup T_j)\big),$$
>
> where $\pi_j(H_j \cup T_j)$ is a *basis* of $K_j/K_{j-1}$ by Step 1 and $\bar N_j$ is injective. So:
>
> - $H_{j-1} \subseteq K_{j-1}$, since $w \in K_j \Rightarrow N^{j-1}(Nw) = N^jw = 0$. This clause is what makes $\pi_{j-1}(H_{j-1})$ a well-formed expression at all.
> - $\pi_{j-1}(H_{j-1})$ is independent in $K_{j-1}/K_{j-2}$, since injective maps preserve independence.
>
> That is precisely the precondition Step 1 needs at level $j-1$, so the loop hands itself a valid input each time and nothing carried is ever discarded. The argument runs for $j \ge 2$, which is all that is needed. At $j = p$ both clauses hold vacuously, since $H_p = \varnothing$. At $j = 1$ every $w \in H_1 \cup T_1$ lies in $K_1 = \operatorname{null}N$, so Step 2 produces only $H_0 = \{0,\dots,0\}$, but Step 3 stops there and $H_0$ is never used.
>
> Free consequence, useful for hand-checking: $\pi_j(H_j \cup T_j)$ is a basis and $\bar N_j$ is injective, so pushing down neither merges two chains nor loses one:
>
> $$\lvert H_{j-1}\rvert = \lvert H_j\rvert + \lvert T_j\rvert = \delta_j.$$
>
> **Chains have the advertised length.** Each $v \in T_j$ has $\pi_j(v) \neq 0$, i.e. $v \notin K_{j-1}$, so $N^{j-1}v \neq 0$ while $N^jv = 0$. Passing to the quotient is what rules out a chain dying early; plain linear independence would not.
>
> **The chains form a basis.** By induction on $j$: the chain vectors at levels $\le j-1$ form a basis of $K_{j-1}$, and $H_j \cup T_j$ lifts a basis of $K_j/K_{j-1}$, so together they give a basis of $K_j$. At $j = p$ this is a basis of $V$, and each chain-span has matrix $J_j(0)$.
>
> **The counts.** Two facts combine. First, Step 1 makes $\pi_j(H_j \cup T_j)$ a basis of $K_j/K_{j-1}$, so the level is exactly as wide as the dimension jump:
>
> $$\lvert H_j \cup T_j\rvert = \dim\left(K_j/K_{j-1}\right) = \delta_j.$$
>
> Second, each chain visits each level at most once. Let a chain have top $v \in T_i$, so $N^{i-1}v \neq 0$ and $N^iv = 0$. Its member $N^tv$ then satisfies
>
> $$N^{i-t-1}\left(N^tv\right) = N^{i-1}v \neq 0, \qquad N^{i-t}\left(N^tv\right) = N^iv = 0,$$
>
> so $N^tv$ lies in $K_{i-t}$ but not in $K_{i-t-1}$, putting it at level $i-t$. As $t$ runs from $0$ to $i-1$ the level runs from $i$ down to $1$: one member at each level from $1$ to $i$, and nothing above level $i$. Step 2 files that member into $T_j$ if $j = i$ and into $H_j$ if $j < i$. Hence $H_j \cup T_j$ is in bijection with the chains of length $\ge j$, and
>
> $$\#\{\text{chains of length} \ge j\} = \delta_j.$$
>
> Subtracting consecutive levels isolates the chains that stop at level $j$, which are exactly the ones born there, namely $T_j$:
>
> $$\lvert T_j\rvert = \#\{\text{chains of length } = j\} = \delta_j - \delta_{j+1} = 2d_j - d_{j-1} - d_{j+1},$$
>
> with the convention $\delta_{p+1} = 0$.
>
> The same subtraction reads as a statement about $\bar N_{j+1}$. Start from Step 2 one level up, which sets $H_j = N(H_{j+1} \cup T_{j+1})$. Pushing that definition through the commuting square gives
>
> $$\pi_j(H_j) = \bar N_{j+1}\big(\pi_{j+1}(H_{j+1} \cup T_{j+1})\big),$$
>
> and by Step 1 at level $j+1$ the list inside the parentheses is a basis of $K_{j+1}/K_j$. An injective map carries a basis of its domain to a basis of its range, so
>
> $$\pi_j(H_j) \ \text{ is a basis of } \ \operatorname{range}\bar N_{j+1}, \qquad \dim\operatorname{range}\bar N_{j+1} = \delta_{j+1}.$$
>
> Now recall the general principle: if a basis of a subspace $W' \subseteq W$ is extended to a basis of $W$, the added vectors represent a basis of $W/W'$. Step 1 at level $j$ performs exactly such an extension, with
>
> $$W' = \operatorname{range}\bar N_{j+1}, \qquad W = K_j/K_{j-1},$$
>
> and the added vectors are $\pi_j(T_j)$. So $\pi_j(T_j)$ represents a basis of
>
> $$W/W' = \big(K_j/K_{j-1}\big)\big/\operatorname{range}\bar N_{j+1} = \operatorname{coker}\bar N_{j+1},$$
>
> of dimension $\delta_j - \delta_{j+1} = \lvert T_j\rvert$. Level $j+1$ sends $\delta_{j+1}$ independent directions down into level $j$, level $j$ has $\delta_j$ directions to fill, and the new tops fill the shortfall. The cokernel is the part of level $j$ that nothing above it reaches. At $j = p$ there is no level above, so read $\operatorname{range}\bar N_{p+1} = \{0\}$, consistent with $\delta_{p+1} = 0$.

{: .prompt-warning }
> Do not run this bottom-up. Most vectors of $\operatorname{null}N$ are not in $\operatorname{range}N$, so choosing a basis of $\operatorname{null}N$ first and hunting for preimages typically fails. Top-down, low-level vectors are *manufactured* by pushing down, never guessed.

{: .prompt-tip }
> *Basis-free confirmation on the chain lengths*
>
> For *any* decomposition into chains of lengths $m_1,\dots,m_k$ we had $d_m = \sum_i \min(m,m_i)$, so
>
> $$ \delta_j = d_j - d_{j-1} = \sum_i \big[\min(j,m_i)-\min(j-1,m_i)\big] = \#\{i : m_i \ge j\}.$$
>
> Injectivity gives
>
> $$\delta_p \le \delta_{p-1} \le \cdots \le \delta_1.$$

**Worked example**

Let $V = \operatorname{span}(1,\, x,\, y,\, x^2,\, xy)$ and $N = \partial/\partial x$. Then $N$ sends

$$1 \mapsto 0,\quad x \mapsto 1,\quad y \mapsto 0,\quad x^2 \mapsto 2x,\quad xy \mapsto y.$$

Precompute: $\operatorname{null}N = \operatorname{span}(1, y)$, $\operatorname{null}N^2 = \operatorname{span}(1,y,x,xy)$, $\operatorname{null}N^3 = V$. So $p = 3$ and $(d_0,d_1,d_2,d_3) = (0,2,4,5)$.

| $j$ | $H_j$ (pushed down) | need $d_j - d_{j-1}$ | new tops $T_j$ |
| --- | ------------------- | -------------------- | -------------- |
| $3$ | $\varnothing$       | $5-4 = 1$            | $\{x^2\}$      |
| $2$ | $\{2x\}$            | $4-2 = 2$            | $\{xy\}$       |
| $1$ | $\{2,\; y\}$        | $2-0 = 2$            | $\varnothing$  |

Reading the rows:

- **$j=3$:** nothing carried in. Need $1$ vector of $\operatorname{null}N^3 = V$ outside $\operatorname{null}N^2$; take $x^2$. Push down: $H_2 = \{2x\}$.
- **$j=2$:** carrying $2x$. Need $2$ vectors total independent modulo $\operatorname{null}N = \operatorname{span}(1,y)$; $2x$ supplies one, so add one more from $\operatorname{null}N^2 = \operatorname{span}(1,y,x,xy)$ — take $xy$. Push down: $H_1 = \{N(2x), N(xy)\} = \{2, y\}$.
- **$j=1$:** carrying $\{2,y\}$, which is already a basis of the $2$-dimensional $\operatorname{null}N$. Nothing to add.

Chains: $x^2 \to 2x \to 2 \to 0$ and $xy \to y \to 0$. Jordan basis (each chain bottom-to-top)

$$(2,\; 2x,\; x^2,\; y,\; xy),$$

giving $\mathcal{M}(N) = J_3(0)\oplus J_2(0)$. Checks: $\lvert T_3 \rvert = 1 = 2d_3-d_2-d_4 = 10-4-5$, $\lvert T_2 \rvert = 1 = 2(4)-2-5$, $\lvert T_1 \rvert = 0 = 2(2)-0-4$, and $3 + 2 = 5 = \dim V$.

{: .prompt-tip }
> Row reduction (Gaussian elimination) example at level $j = 2$.
>
> Coordinates are with respect to the ordered basis $(1, x, y, x^2, xy)$.
>
> The list is: group 1 is $1, y$; group 2 is $2x$; group 3 is $1, y, x, xy$. As columns,
>
> $$\left(\begin{array}{ccc|cccc} 1 & 0 & 0 & 1 & 0 & 0 & 0\\ 0 & 0 & 2 & 0 & 0 & 1 & 0\\ 0 & 1 & 0 & 0 & 1 & 0 & 0\\ 0 & 0 & 0 & 0 & 0 & 0 & 0\\ 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{array}\right)$$
>
> Reducing (here only row swaps are needed) puts pivots in columns $1, 2, 3, 7$. Columns $4$ and $5$ repeat group 1, and column $6$ is a scalar multiple of column $3$, so all three are non-pivots. The single pivot inside group 3 is column $7$, giving
>
> $$T_2 = \{xy\},$$
>
> which matches what was found by hand. The count check holds: $\lvert H_2\rvert + \lvert T_2\rvert = 1 + 1 = 2 = \delta_2$.

{: .prompt-tip }
> - Full reduced echelon form is unnecessary. Forward elimination to echelon form already reveals the pivot positions, which is all you need.
> - Group 3 can be any spanning set of $\operatorname{null}N^j$; a basis is the convenient choice, and duplicates cost nothing since they simply fail to be pivots.
> - The same elimination produces the null spaces in the first place, by solving $N^j x = 0$, so you can compute the entire tower $\operatorname{null}N, \dots, \operatorname{null}N^p$ with the same tool before the loop starts.

## Invertibility

{: .prompt-info }
> Let $W$ be a finite-dimensional vector space, let $N \in \mathcal{L}(W)$ be nilpotent, and let $c \in \mathbb{F}$ with $c \neq 0$. Then $cI + N$ is invertible.

{: .prompt-proof }
> **Lemma** *Let $A \in \mathcal{L}(W)$, $c \in \mathbb{F}$, and set $B = cI + A$. If $\mu$ is an eigenvalue of $B$, then $\mu - c$ is an eigenvalue of $A$.*
>
> *Proof.* Let $v \neq 0$ satisfy $Bv = \mu v$. Then
>
> $$Av = (B - cI)v = Bv - cv = \mu v - cv = (\mu - c)v ,$$
>
> and $v \neq 0$, so $\mu - c$ is an eigenvalue of $A$. $\blacksquare$
>
> **Proof of the Theorem.**
>
> If $W = \{0\}$ the statement is trivial, so assume $W \neq \{0\}$.
>
> Now we prove $0$ is not an eigenvalue of $cI + N$. Suppose toward a contradiction that it is. Applying the Lemma with $A = N$, $B = cI + R$, and $\mu = 0$, we conclude that $0 - c = -c$ is an eigenvalue of $N$. The only eigenvalue of $N$ is $0$, hence $-c = 0$, i.e. $c = 0$ — contradicting the hypothesis $c \neq 0$.
>
> Therefore, $cI + N$ is injective and thus invertible. $\blacksquare$

## Commuting Operators

### Parametrization

Suppose $SN = NS$. Then $SN^j = N^jS$ for all $j$, so for every basis vector

$$S(N^j v_i) = N^j (S v_i).$$

The left side is $S$ on an arbitrary *basis* vector; the right side only involves the $k$ vectors $Sv_1,\dots,Sv_k$. So a commuting $S$ is pinned down by its values on the $k$ tops.

Write $w_i = Sv_i$. Since $N^{m_i}v_i = 0$,

$$0 = S(N^{m_i}v_i) = N^{m_i}(Sv_i) = N^{m_i}w_i,$$

so the constraint is

$$\boxed{\,w_i \in \operatorname{null} N^{m_i}\,}$$

Each top may be sent anywhere killed by $N^{m_i}$.

Conversely, pick any $w_1,\dots,w_k$ with $w_i \in \operatorname{null}N^{m_i}$ and *define* $S$ on the basis by

$$S(N^jv_i) := N^j w_i, \qquad 0 \le j \le m_i - 1.$$

$S$ commutes with $N$. Check on a basis vector $N^jv_i$:

- If $j < m_i - 1$: $\;SN(N^jv_i) = S(N^{j+1}v_i) = N^{j+1}w_i$, while $NS(N^jv_i) = N(N^jw_i) = N^{j+1}w_i$. Equal.
- If $j = m_i-1$ (bottom of the chain): $\;SN(N^{m_i-1}v_i) = S(0) = 0$, while $NS(N^{m_i-1}v_i) = N^{m_i}w_i = 0$ by the constraint.

So the constraint is precisely what's needed at the *bottom* of each chain, and nowhere else. Conclusion:

$$\{S : SN = NS\} \;\cong\; \operatorname{null}N^{m_1} \times \cdots \times \operatorname{null}N^{m_k}, \qquad S \mapsto (Sv_1,\dots,Sv_k),$$

$$\dim\{S : SN = NS\} = \sum_{i,j}\min(m_i,m_j).$$

{: .prompt-warning }
> Nothing above says $Sv_i$ has to lie in chain $i$. The reason is visible in the constraint itself: $\operatorname{null}N^{m_i}$ is not a subspace of chain $i$. It is spanned by the bottom $\min(m_i, m_j)$ vectors of **every** chain $j$, so it reaches across the whole array. In the extreme case where all lengths equal $p$, we get $\operatorname{null}N^{m_i} = \operatorname{null}N^p = V$ and there is no constraint whatsoever.
>
> Take $\mathbb{C}^4$ with two chains of length $2$:
>
> $$e_2 \to e_1 \to 0, \qquad e_4 \to e_3 \to 0.$$
>
> Here $N^2 = 0$, so both constraints read $w_i \in \mathbb{C}^4$ and any pair of tops is legal. Swapping the chains is the choice $w_1 = e_4$, $w_2 = e_2$, which forces
>
> $$Se_2 = e_4, \quad Se_1 = Ne_4 = e_3, \qquad Se_4 = e_2, \quad Se_3 = Ne_2 = e_1.$$
>
> No verification is needed, since the construction guarantees $SN = NS$, but as a spot-check: $SNe_2 = Se_1 = e_3$ and $NSe_2 = Ne_4 = e_3$. So $S$ commutes with $N$ while carrying chain $1$ onto chain $2$.
>
> Shearing works the same way. Take $w_1 = e_2 + e_4$ and $w_2 = e_4$, giving
>
> $$Se_2 = e_2 + e_4, \quad Se_1 = e_1 + e_3, \qquad Se_4 = e_4, \quad Se_3 = e_3.$$
>
> This blends chain $1$ into chain $2$ without being a permutation of the basis.
>
> The most extreme case is $N = 0$ on $\mathbb{C}^2$, two chains of length $1$. Every operator commutes, so the commutant is all of $\mathcal{L}(\mathbb{C}^2)$, of dimension $4$; the ones preserving $\operatorname{span}(e_1)$ are the upper triangular matrices, a proper subspace of dimension $3$. Preserving a chosen block is a genuine restriction that most commuting operators fail.
>
> Unequal lengths are where the constraint bites. Take $\mathbb{C}^3$ with chains
>
> $$e_2 \to e_1 \to 0, \qquad f \to 0,$$
>
> so $m_1 = 2$ and $m_2 = 1$. The top of the long chain is unconstrained, since $\operatorname{null}N^2 = \mathbb{C}^3$, so $Se_2$ may be $f$. But the top of the short chain satisfies $Sf \in \operatorname{null}N = \operatorname{span}(e_1, f)$, so $Sf$ cannot be $e_2$: a short chain can only be sent into the bottom portion of a longer one, never onto its top. The freedom is **asymmetric**, and $\min(m_i, m_j)$ is exactly the bookkeeping for it.

**Worked example**

Take $N = \partial/\partial x$ on $V = \operatorname{span}(1,x,y,x^2,xy)$ again, with the Jordan basis found earlier: chain $x^2 \to 2x \to 2$ and chain $xy \to y$. So $v_1 = x^2$ with $m_1 = 3$, and $v_2 = xy$ with $m_2 = 2$.

Constraints: $w_1 \in \operatorname{null}N^3 = V$, unrestricted; $w_2 \in \operatorname{null}N^2 = \operatorname{span}(1,y,x,xy)$.

The asymmetry is visible: $w_1$ may be $xy$, but $w_2$ may **not** be $x^2$, since $N^2x^2 = 2 \neq 0$. The short chain cannot be sent to the top of the long one.

Choose $w_1 = xy$ and $w_2 = 2x$. Then

$$S(x^2) = xy, \quad S(2x) = N(xy) = y, \quad S(2) = N^2(xy) = 0,$$
$$S(xy) = 2x, \quad S(y) = N(2x) = 2.$$

Spot-check: $SN(x^2) = S(2x) = y$ and $NS(x^2) = N(xy) = y$; also $SN(xy) = S(y) = 2$ and $NS(xy) = N(2x) = 2$.

Dimension check: $\dim\operatorname{null}N^3 + \dim\operatorname{null}N^2 = 5 + 4 = 9$, agreeing with $\sum_{i,j}\min(m_i,m_j) = 3+2+2+2$, out of $25$ for all of $\mathcal{L}(V)$.

The isomorphism also handles the **basis** of the commutant for free: run over a basis of $\operatorname{null}N^{m_i}$ for one index with the other $w$'s set to $0$. Here that gives $5 + 4 = 9$ explicit commuting operators spanning the commutant.

### Single-chain Case

If $k = 1$ with length $m$, then $\operatorname{null}N^m = V$, so there is **no constraint at all** and $S$ is determined by an arbitrary $w = Sv$. Since $k = 1$, the chain

$$v,\; Nv,\; \dots,\; N^{m-1}v$$

is a basis of $V$, so expand $w$ in it:

$$w = c_0v + c_1Nv + \dots + c_{m-1}N^{m-1}v = q(N)v, \qquad q(z) = c_0 + c_1z + \dots + c_{m-1}z^{m-1}.$$

Now compare $S$ with $q(N)$ on each basis vector $N^tv$, for $0 \le t \le m-1$:

$$S(N^tv) \;=\; N^t(Sv) \;=\; N^t\big(q(N)v\big) \;=\; q(N)(N^tv).$$

The first equality is $SN^t = N^tS$, from $SN = NS$. The second substitutes $Sv = w = q(N)v$. The third is $N^tq(N) = q(N)N^t$, since polynomials in $N$ commute with powers of $N$.

So $S$ and $q(N)$ agree on a basis of $V$, hence are equal as operators:

$$S = c_0I + c_1N + \dots + c_{m-1}N^{m-1}.$$

In the basis $(N^{m-1}v,\dots,Nv,v)$ this is the upper-triangular [Toeplitz matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix) with $c_0$ down the diagonal, $c_1$ on the next diagonal, and so on.

{: .prompt-warning }
> Polynomials in $N$ always commute with $N$, so $\{\text{polynomials in }N\} \subseteq \{S : SN = NS\}$ for every nilpotent $N$. What the one-chain case gives is *equality*, and that is what fails when $k \ge 2$: the inclusion becomes strict, so "commutes with $N$" is no longer the same condition as "is a polynomial in $N$".

{: .prompt-proof }
> Write $\mathcal{C}(N) = \\{S \in \mathcal{L}(V) : SN = NS\\}$ for the commutant of $N$, and $\mathcal{P}(N) = \\{q(N) : q \in \mathcal{P}(\mathbf{F})\\}$ for the operators expressible as polynomials in $N$. Let $p = \max_i m_i$ be the index of nilpotency, so $N^p = 0$, and recall $\dim V = \sum_i m_i$.
>
> **The inclusion always holds.** $N$ commutes with $I$ and with itself, hence with every polynomial in itself, so $\mathcal{P}(N) \subseteq \mathcal{C}(N)$. Only equality is at issue.
>
> **$\dim\mathcal{P}(N) = p$.** Spanning: any term of degree $\ge p$ vanishes on substituting $N$, since writing $q(z) = z^pa(z) + r(z)$ with $\deg r < p$ gives $q(N) = r(N)$. So $I, N, \dots, N^{p-1}$ spans $\mathcal{P}(N)$. Independence: choose $i_0$ with $m_{i_0} = p$, so the chain of $v_{i_0}$ has exactly $p$ entries. If $\sum_{t<p} c_tN^t = 0$, applying it to $v_{i_0}$ gives $\sum_{t<p} c_tN^tv_{i_0} = 0$, a relation among $p$ distinct members of the Jordan basis, so every $c_t = 0$.
>
> **$\dim\mathcal{C}(N) \ge \dim V$.** By the parametrization above, $\dim\mathcal{C}(N) = \sum_{i,j}\min(m_i,m_j)$. View that as a $k \times k$ table with entry $(i,j)$ equal to $\min(m_i,m_j)$. Its diagonal entries are $\min(m_i,m_i) = m_i$, summing to $\dim V$, and every remaining entry is positive. So the bound holds, with equality exactly when the table has no off-diagonal entries, i.e. when $k = 1$.
>
> **Conclusion.** If $k \ge 2$, then $\dim V = \sum_i m_i > \max_i m_i = p$, since the omitted lengths are positive, so
>
> $$\dim\mathcal{C}(N) \ \ge\ \dim V \ >\ p \ =\ \dim\mathcal{P}(N)$$
>
> and the inclusion is strict. If $k = 1$, the double sum has the single term $\min(m_1,m_1) = m_1 = \dim V = p$, so both spaces have dimension $p$, and an inclusion of subspaces of equal finite dimension is an equality. Hence
>
> $$\mathcal{C}(N) = \mathcal{P}(N) \iff k = 1,$$
>
> that is, iff $N$ has a single Jordan block, equivalently $\dim\operatorname{null}N = 1$.

**Worked example**

Take $\mathbf{C}^3$ with chain $e_3 \to e_2 \to e_1 \to 0$; that is, $Ne_3 = e_2$, $Ne_2 = e_1$, $Ne_1 = 0$, so

$$N = \begin{pmatrix} 0&1&0\\ 0&0&1\\ 0&0&0\end{pmatrix}.$$

Suppose $SN = NS$. Whatever $S$ does to the top vector, say $Se_3 = w$, everything else is forced:

$$Se_2 = S(Ne_3) = N(Se_3) = Nw, \qquad Se_1 = S(Ne_2) = N(Se_2) = N^2w.$$

So $S$ is completely determined by the single vector $w$. Conversely, any $w$ works: the only condition left to check is $S(Ne_1) = N(Se_1)$, i.e. $0 = N^3w$, which is automatic.

Write $w = a e_1 + b e_2 + c e_3$. Then $Se_3 = ae_1+be_2+ce_3$, $Se_2 = be_1 + ce_2$, $Se_1 = ce_1$, so

$$S = \begin{pmatrix} c&b&a\\ 0&c&b\\ 0&0&c\end{pmatrix} = cI + bN + aN^2.$$
