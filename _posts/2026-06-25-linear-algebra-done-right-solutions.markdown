---
title:  "Linear Algebra Done Right (4th Edition) Solutions"
category: math
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## 3E

{: .prompt-info }
> 11\. Suppose $ U = \\{ (x1,x2,\dots) \in \mathbf{F}^\infty : x_k \ne 0 \ \text{for only finitely many k} \\} $.
>
> (b) Prove that $ \mathbf{F}^\infty/U $ is infinite-dimensional.

**Goal and strategy**

We show $\mathbf{F}^\infty/U$ is infinite-dimensional, i.e. **not** finite-dimensional. We use the standard criterion:

> A vector space is finite-dimensional of dimension $n$ only if every linearly independent list has length $\le n$. So a space is infinite-dimensional if, for every positive integer $m$, it contains a linearly independent list of length $m$.

Accordingly, we build one infinite list of vectors $v_1, v_2, \dots$ in $\mathbf{F}^\infty$ such that for **every** $m$, the cosets $v_1 + U, \dots, v_m + U$ are linearly independent in the quotient. That gives independent lists of every length, so the quotient cannot be finite-dimensional.

**Translating independence in the quotient**

Before constructing anything, let's record exactly what we must verify. For scalars $a_1, \dots, a_m$,

$$a_1(v_1 + U) + \cdots + a_m(v_m + U) = (a_1 v_1 + \cdots + a_m v_m) + U,$$

and a coset $x + U$ is the zero element of $\mathbf{F}^\infty/U$ precisely when $x \in U$. Therefore

$$v_1 + U, \dots, v_m + U \text{ are linearly independent} \iff \Big(\, a_1 v_1 + \cdots + a_m v_m \in U \ \Rightarrow\ a_1 = \cdots = a_m = 0 \,\Big). \tag{$\ast$}$$

Since $U$ is the set of sequences with only finitely many nonzero entries, the right-hand side says: *no nontrivial linear combination of the $v_j$ is finitely supported.* That is the property our vectors must have.

**Constructing the vectors**

The idea is to give each $v_j$ an **infinite** "footprint" of slots, with the footprints of different vectors **disjoint**, so that combinations cannot cancel down to finitely many nonzero terms.

Partition the index set $\{1, 2, 3, \dots\}$ into infinitely many pairwise disjoint infinite subsets $S_1, S_2, S_3, \dots$. One explicit choice: write each positive integer uniquely as $n = 2^{\,j-1}(2k-1)$ with $j, k \ge 1$, and let

$$S_j = \{\, 2^{\,j-1}(2k-1) : k \ge 1 \,\}.$$

Each $S_j$ is infinite, the $S_j$ are pairwise disjoint, and they cover every index. (Any partition into infinitely many infinite blocks works; only these three properties matter.)

Define $v_j \in \mathbf{F}^\infty$ to be the **indicator sequence** of $S_j$: its entry in slot $n$, written $(v_j)_n$, is

$$(v_j)_n = \begin{cases} 1 & n \in S_j, \\ 0 & n \notin S_j. \end{cases}$$

Each $v_j$ has infinitely many nonzero entries (one for each element of the infinite set $S_j$), so $v_j \notin U$.

**Verifying independence of the cosets**

Fix any $m$, and suppose some combination lies in $U$:

$$w := a_1 v_1 + a_2 v_2 + \cdots + a_m v_m \in U.$$

Here $w$ is itself a sequence in $\mathbf{F}^\infty$; write $w_n$ for its entry in slot $n$, so $w_n = a_1 (v_1)_n + \cdots + a_m (v_m)_n$.

**Each block carries a single coefficient.** Fix $j$ with $1 \le j \le m$ and let $n \in S_j$. Because the supports are disjoint, $n$ belongs to $S_j$ and to no other $S_i$; hence $(v_j)_n = 1$ while $(v_i)_n = 0$ for all $i \ne j$. The sum defining $w_n$ collapses to its single surviving term:

$$w_n = a_j \qquad \text{for every } n \in S_j. \tag{$\dagger$}$$

**Membership in $U$ forces each coefficient to vanish.** Suppose, toward a contradiction, that $a_j \ne 0$ for some $j \le m$. By $(\dagger)$, $w_n = a_j \ne 0$ at *every* index $n \in S_j$. Since $S_j$ is infinite, $w$ then has infinitely many nonzero entries — contradicting $w \in U$. Hence $a_j = 0$, and as $j \le m$ was arbitrary,

$$a_1 = a_2 = \cdots = a_m = 0.$$

By the criterion $(\ast)$, the cosets $v_1 + U, \dots, v_m + U$ are linearly independent.

**Conclusion**

For every positive integer $m$, the quotient $\mathbf{F}^\infty/U$ contains a linearly independent list of length $m$, namely $v_1 + U, \dots, v_m + U$. No finite-dimensional space has independent lists of arbitrary length, so

$$\mathbf{F}^\infty / U \text{ is infinite-dimensional.} \qquad \blacksquare$$

**Remark**

Conceptually, $U$ is the subspace of sequences that eventually vanish, so passing to $\mathbf{F}^\infty/U$ discards everything except a sequence's long-run "tail." The vectors $v_j$ were designed to have persistent, non-overlapping tails: each lives forever on its own block $S_j$, and disjointness — distilled in equation $(\dagger)$ — means a linear combination reads off coefficient $a_j$ across the whole of $S_j$, with no way for different vectors to interfere and cancel. So no nontrivial combination can die out, giving infinitely many independent directions in the quotient.

---

## 4

{: .prompt-info }
> 12\. Suppose $m$ is a nonnegative integer and $ p \in \mathcal{p}(\mathbb{C}) $ is such that there are distinct real numbers $ x_0, x_1, \dots, x_m $ with $ p(x_k) \in \mathbb{R} $ for each $ k = 0, 1, \dots, m $. Prove that all coefficients of $p$ are real.

Write $p(z) = a_0 + a_1 z + \cdots + a_m z^m$ with $a_j \in \mathbf{C}$. Define its **coefficient-conjugate**

$$
\bar p(z) := \overline{a_0} + \overline{a_1} z + \cdots + \overline{a_m} z^m,
$$

For any $z$,

$$
\overline{p(z)} = \overline{\sum_j a_j z^j} = \sum_j \overline{a_j}\,\overline{z}^{\,j} = \bar p(\overline{z}).
$$

Now specialize to a **real** input $x$, where $\overline{x} = x$:

$$
\bar p(x) = \overline{p(x)} \qquad \text{for every } x \in \mathbf{R}.
$$

By assumption $p(x_k) \in \mathbf{R}$, which means $\overline{p(x_k)} = p(x_k)$. Feeding each real point $x_k$ into the identity above:

$$
\bar p(x_k) = \overline{p(x_k)} = p(x_k) \qquad k = 0, 1, \dots, m.
$$

So $p$ and $\bar p$ **agree at the $m+1$ distinct points** $x_0,\dots,x_m$.

Consider the difference $q := p - \bar p$. It lies in $\mathcal{P}_m(\mathbf{C})$, so $\deg q \le m$. But $q(x_k) = 0$ for all $m+1$ distinct values $x_0,\dots,x_m$ — that's $m+1$ distinct zeros. By 4.8, a nonzero polynomial of degree $\le m$ has *at most $m$* zeros. Having $m+1$ is one too many, so the only escape is

$$
q = 0, \qquad\text{i.e.}\qquad p = \bar p.
$$

Hence $a_j = \overline{a_j}$ for every $j$, so all coefficients of $p$ are real. $\blacksquare$
