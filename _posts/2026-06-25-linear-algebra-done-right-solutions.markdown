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

$$a_1(v_1 + U) + \dots + a_m(v_m + U) = (a_1 v_1 + \dots + a_m v_m) + U,$$

and a coset $x + U$ is the zero element of $\mathbf{F}^\infty/U$ precisely when $x \in U$. Therefore

$$v_1 + U, \dots, v_m + U \text{ are linearly independent} \iff \Big(\, a_1 v_1 + \dots + a_m v_m \in U \ \Rightarrow\ a_1 = \dots = a_m = 0 \,\Big). \tag{$\ast$}$$

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

$$w := a_1 v_1 + a_2 v_2 + \dots + a_m v_m \in U.$$

Here $w$ is itself a sequence in $\mathbf{F}^\infty$; write $w_n$ for its entry in slot $n$, so $w_n = a_1 (v_1)_n + \dots + a_m (v_m)_n$.

**Each block carries a single coefficient.** Fix $j$ with $1 \le j \le m$ and let $n \in S_j$. Because the supports are disjoint, $n$ belongs to $S_j$ and to no other $S_i$; hence $(v_j)_n = 1$ while $(v_i)_n = 0$ for all $i \ne j$. The sum defining $w_n$ collapses to its single surviving term:

$$w_n = a_j \qquad \text{for every } n \in S_j. \tag{$\dagger$}$$

**Membership in $U$ forces each coefficient to vanish.** Suppose, toward a contradiction, that $a_j \ne 0$ for some $j \le m$. By $(\dagger)$, $w_n = a_j \ne 0$ at *every* index $n \in S_j$. Since $S_j$ is infinite, $w$ then has infinitely many nonzero entries — contradicting $w \in U$. Hence $a_j = 0$, and as $j \le m$ was arbitrary,

$$a_1 = a_2 = \dots = a_m = 0.$$

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

Write $p(z) = a_0 + a_1 z + \dots + a_m z^m$ with $a_j \in \mathbb{C}$. Define its **coefficient-conjugate**

$$
\bar p(z) := \overline{a_0} + \overline{a_1} z + \dots + \overline{a_m} z^m,
$$

For any $z$,

$$
\overline{p(z)} = \overline{\sum_j a_j z^j} = \sum_j \overline{a_j}\,\overline{z}^{\,j} = \bar p(\overline{z}).
$$

Now specialize to a **real** input $x$, where $\overline{x} = x$:

$$
\bar p(x) = \overline{p(x)} \qquad \text{for every } x \in \mathbb{R}.
$$

By assumption $p(x_k) \in \mathbb{R}$, which means $\overline{p(x_k)} = p(x_k)$. Feeding each real point $x_k$ into the identity above:

$$
\bar p(x_k) = \overline{p(x_k)} = p(x_k) \qquad k = 0, 1, \dots, m.
$$

So $p$ and $\bar p$ **agree at the $m+1$ distinct points** $x_0,\dots,x_m$.

Consider the difference $q := p - \bar p$. It lies in $\mathcal{P}_m(\mathbb{C})$, so $\deg q \le m$. But $q(x_k) = 0$ for all $m+1$ distinct values $x_0,\dots,x_m$ — that's $m+1$ distinct zeros. By 4.8, a nonzero polynomial of degree $\le m$ has *at most $m$* zeros. Having $m+1$ is one too many, so the only escape is

$$
q = 0, \qquad\text{i.e.}\qquad p = \bar p.
$$

Hence $a_j = \overline{a_j}$ for every $j$, so all coefficients of $p$ are real. $\blacksquare$

---

## 5D

{: .prompt-info }
> 16\. Suppose that $ T \in \mathcal{L}(V) $ is diagonalizable. Let $ \lambda_1, \dots, \lambda_m $ denote the distinct eigenvalues of $ T $. Prove that a subspace $ U $ of $ V $ is invariant under $ T $ if and only if there exist subspaces 𝑈1,…,𝑈𝑚 of 𝑉 such that 𝑈𝑘 ⊆ 𝐸(𝜆𝑘,𝑇) for each $ k $ and $ U = U_1 \oplus \dots \oplus U_m $.

{: .prompt-proof }
> *($\Rightarrow$)*
>
> **Step 1: the projections, built out of $T$.**
>
> For each $k$ define
>
> $$q_k = \prod_{j \neq k} (T - \lambda_j I) \in \mathcal{L}(V)$$
>
> (the product over all $j \in \{1,\dots,m\}$ with $j \neq k$; if $m = 1$ this is the empty product, $q_1 = I$). If $v \in E(\lambda_i, T)$, then each factor acts on $v$ as multiplication by $\lambda_i - \lambda_j$, so
>
> $$q_k v = \Big(\prod_{j\neq k}(\lambda_i - \lambda_j)\Big) v .$$
>
> When $i \neq k$ the index $j = i$ occurs in the product, so the scalar is $0$ and $q_k v = 0$. When $i = k$ the scalar is
>
> $$c_k = \prod_{j \neq k}(\lambda_k - \lambda_j) \neq 0,$$
>
> nonzero because the eigenvalues are distinct. So $q_k$ annihilates every eigenspace but the $k$-th, and acts as multiplication by $c_k$ on $E(\lambda_k, T)$.
>
>**Step 2: the construction.**
>
> Assume $U$ is invariant under $T$, and set
>
> $$U_k = U \cap E(\lambda_k, T).$$
>
> Each $U_k$ is a subspace of $V$ (intersection of subspaces) with $U_k \subseteq E(\lambda_k,T)$, as required.
>
> *$U \subseteq U_1 + \dots + U_m$.* Let $u \in U$. Since $T$ is diagonalizable, $V = E(\lambda_1,T) \oplus \dots \oplus E(\lambda_m,T)$, so we may write
>
> $$u = u_1 + \dots + u_m, \qquad u_k \in E(\lambda_k, T).$$
>
> Apply $q_k$ and use Step 1:
>
> $$q_k u = \sum_{i=1}^m q_k u_i = c_k u_k .$$
>
> $U$ is invariant under every polynomial in $T$, so $q_k u \in U$; since $c_k \neq 0$, this gives $u_k = c_k^{-1} q_k u \in U$. As also $u_k \in E(\lambda_k,T)$, we conclude $u_k \in U_k$. Hence $u \in U_1 + \dots + U_m$.
>
> *$U_1 + \dots + U_m \subseteq U$.* Each $U_k \subseteq U$ and $U$ is a subspace.
>
> *The sum is direct.* Suppose $u_1 + \dots + u_m = 0$ with $u_k \in U_k \subseteq E(\lambda_k, T)$. Directness of $E(\lambda_1,T) \oplus \dots \oplus E(\lambda_m,T)$ forces every $u_k = 0$.
>
> Therefore $U = U_1 \oplus \dots \oplus U_m$.
>
> *($\Leftarrow$)* If $U = U_1 \oplus \dots \oplus U_m$ with $U_k \subseteq E(\lambda_k,T)$, take $u \in U$ and write $u = \sum u_k$; then $Tu = \sum \lambda_k u_k \in U_1 + \dots + U_m = U$. $\blacksquare$

{: .prompt-tip }
> The operator $q_k/c_k$ is the projection onto $E(\lambda_k,T)$ along the other eigenspaces, expressed as a *polynomial in $T$*. It's Lagrange interpolation in disguise: $p_k(x) = \prod_{j\neq k}\frac{x - \lambda_j}{\lambda_k - \lambda_j}$ is the polynomial with $p_k(\lambda_i) = \delta_{ik}$, and $p_k(T)$ is the projection.

{: .prompt-tip }
> The $T$-invariant subspaces of a diagonalizable operator are exactly the spans of sets of eigenvectors.

{: .prompt-info }
> 17\. Suppose $ V $ is finite-dimensional. Prove that $ \mathcal{L}(V) $ has a basis consisting of diagonalizable operators.

{: .prompt-proof }
> **Setup.** Assume $n = \dim V \geq 1$ (if $V = \{0\}$ then $\mathcal{L}(V) = \{0\}$ and the empty list is a basis). Fix a basis $v_1, \dots, v_n$ of $V$ and let $E_{j,k} \in \mathcal{L}(V)$ be the operator determined by
>
> $$E_{j,k} v_k = v_j, \qquad E_{j,k}v_i = 0 \ \text{ for } i \neq k,$$
>
> i.e. its matrix has a $1$ in row $j$, column $k$, and zeros elsewhere. These $n^2$ operators are the standard basis of $\mathcal{L}(V)$, so $\dim \mathcal{L}(V) = n^2$.
>
> **The candidate list.** Define
>
> $$\mathcal{B} = \{\, E_{j,j} : 1 \le j \le n \,\} \cup \{\, E_{j,j} + E_{j,k} : j \neq k \,\}.$$
>
> It has $n + n(n-1) = n^2$ elements.
>
> **Step 1: every element of $\mathcal{B}$ is diagonalizable.**
>
> Each $E_{j,j}$ is already diagonal with respect to $v_1,\dots,v_n$: it fixes $v_j$ and kills the other basis vectors.
>
> Now fix $j \neq k$ and put $T = E_{j,j} + E_{j,k}$, so that
>
> $$Tv_j = v_j, \qquad Tv_k = v_j, \qquad Tv_i = 0 \ \text{ for } i \notin \{j,k\}.$$
>
> Consider the list
>
> $$v_j, \quad v_j - v_k, \quad (v_i)_{i \notin \{j,k\}}.$$
>
> Every vector in it is an eigenvector of $T$: the first has eigenvalue $1$, and the rest have eigenvalue $0$, since $T(v_j - v_k) = v_j - v_j = 0$ and $Tv_i = 0$ for $i \notin \{j,k\}$. The list has $n$ vectors and spans $V$, because $v_k = v_j - (v_j - v_k)$ recovers the one missing basis vector. So it is a basis of $V$ consisting of eigenvectors of $T$, and $T$ is diagonalizable.
>
> **Step 2: $\mathcal{B}$ spans $\mathcal{L}(V)$.**
>
> For $j \neq k$,
>
> $$E_{j,k} = (E_{j,j} + E_{j,k}) - E_{j,j} \in \operatorname{span}\mathcal{B},$$
>
> and each $E_{j,j} \in \mathcal{B}$. So $\operatorname{span}\mathcal{B}$ contains all $n^2$ matrix units, hence equals $\mathcal{L}(V)$.
>
> **Step 3: conclude.** $\mathcal{B}$ is a spanning list of $\mathcal{L}(V)$ of length $n^2 = \dim\mathcal{L}(V)$, so it is a basis (a spanning list of the right length is automatically independent). Its elements are diagonalizable by Step 1. $\blacksquare$

## 5E

{: .prompt-info }
> 6\. Suppose that $V$ is a nonzero finite-dimensional complex vector space and $S,T \in \mathcal{L}(V) $ commute. Prove that there exist $\alpha, \lambda \in \mathbb{C}$ such that
>
> $$ \operatorname{range}(S - \alpha I) + \operatorname{range}(T - \lambda I) \ne V. $$

{: .prompt-proof }
> $S'$ and $T'$ commute on $V'$, which is a nonzero finite-dimensional complex vector space. By the common-eigenvector theorem for commuting operators, there exist $\varphi \in V'$ with $\varphi \neq 0$ and scalars $\alpha, \lambda \in \mathbb{C}$ with
>
> $$S'\varphi = \alpha\varphi, \qquad T'\varphi = \lambda\varphi.$$
>
> Now for every $v \in V$,
>
> $$\varphi\big((S-\alpha I)v\big) = (S'\varphi)(v) - \alpha\varphi(v) = \alpha\varphi(v) - \alpha\varphi(v) = 0,$$
>
> so $\operatorname{range}(S - \alpha I) \subseteq \operatorname{null}\varphi$, and likewise $\operatorname{range}(T-\lambda I) \subseteq \operatorname{null}\varphi$. Since $\varphi \neq 0$, $\operatorname{null}\varphi$ has dimension $n-1$, so
>
> $$\operatorname{range}(S-\alpha I) + \operatorname{range}(T-\lambda I) \subseteq \operatorname{null}\varphi \neq V. \qquad \blacksquare$$
