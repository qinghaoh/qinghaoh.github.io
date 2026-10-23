---
title:  "Similarity"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
---

Two operators are **similar** when they are the same operator seen in different coordinates. This post makes that precise through change of basis, then asks the natural question: given $S$ and $T$, how can you tell whether they are similar?

The earlier posts collected the pieces of the answer. Here they are put together: first the quantities similarity preserves, then the theorem that some of them suffice when the minimal polynomial splits, and finally what changes over $\mathbb{R}$ and over other fields.

## Notation

{% include notation-table.md keys="F n LV MT eigenspace gen_eigenspace spectrum geo_mult alg_mult min_poly min_poly_exp jordan_block partition conj_partition commutant complexification_op" %}

Throughout, $S, T \in \mathcal{L}(V)$ and $n = \dim V$.

## Change of Basis and Similarity

Write $\mathcal{M}(T, (u_1, \dots, u_n), (v_1, \dots, v_n))$ for the [matrix]({% post_url 2026-06-11-linear-algebra %}#matrices) of $T$ with the $u$'s as the basis of the domain and the $v$'s as the basis of the target, and $\mathcal{M}(T, (u_1, \dots, u_n))$ when both are the $u$'s.

{: .prompt-info }
> *Change of basis.* Let $u_1, \dots, u_n$ and $v_1, \dots, v_n$ be bases of $V$, and let $C = \mathcal{M}(I, (u_1, \dots, u_n), (v_1, \dots, v_n))$, the matrix whose $k$-th column expresses $u_k$ in the $v$'s. Then $C$ is invertible with $C^{-1} = \mathcal{M}(I, (v_1, \dots, v_n), (u_1, \dots, u_n))$, and for every $T \in \mathcal{L}(V)$
>
> $$\mathcal{M}(T, (u_1, \dots, u_n)) = C^{-1}\,\mathcal{M}(T, (v_1, \dots, v_n))\,C.$$

{: .prompt-proof }
> Matrices multiply like the maps they represent: the matrix of a composition, with matching bases in the middle, is the product of the matrices. So $\mathcal{M}(I, (v), (u))\,\mathcal{M}(I, (u), (v)) = \mathcal{M}(I, (u)) = I$, and similarly in the other order, which proves the claim about $C^{-1}$. Writing $T = ITI$,
>
> $$\mathcal{M}(T, (u)) = \mathcal{M}(I, (v), (u))\;\mathcal{M}(T, (v))\;\mathcal{M}(I, (u), (v)) = C^{-1}\,\mathcal{M}(T, (v))\,C. \qquad \blacksquare$$

{: .prompt-info }
> *Similarity.* $A, B \in \mathbf{F}^{n,n}$ are **similar**, written $A \sim B$, if $A = C^{-1}BC$ for some invertible $C \in \mathbf{F}^{n,n}$. Likewise $S, T \in \mathcal{L}(V)$ are **similar** if $S = C^{-1}TC$ for some invertible $C \in \mathcal{L}(V)$.
>
> The two readings agree: $S \sim T$ iff they have the same matrix with respect to some pair of bases,
>
> $$\mathcal{M}(S, (u_1, \dots, u_n)) = \mathcal{M}(T, (v_1, \dots, v_n)).$$

{: .prompt-proof }
> ($\Rightarrow$) Suppose $S = C^{-1}TC$. Take any basis $v_1, \dots, v_n$ and set $u_i = C^{-1}v_i$. If $Tv_k = \sum_j A_{j,k}v_j$, then $Su_k = C^{-1}Tv_k = \sum_j A_{j,k}C^{-1}v_j = \sum_j A_{j,k}u_j$, so both matrices are $A$.
>
> ($\Leftarrow$) Suppose both matrices are $A$, and let $C$ be the invertible operator with $Cu_i = v_i$. Then $C^{-1}TCu_k = C^{-1}Tv_k = \sum_j A_{j,k}C^{-1}v_j = \sum_j A_{j,k}u_j = Su_k$ for every $k$, so $S = C^{-1}TC$. $\blacksquare$

By the change-of-basis formula, two matrices are similar exactly when they are the matrices of one operator with respect to two bases. So everything below can be read for matrices or for operators.

## Similarity Invariants

A quantity attached to operators is a **similarity invariant** if similar operators always share it.

{: .prompt-info }
> Suppose $S = C^{-1}TC$ with $C$ invertible. Then:
>
> (a) $q(S) = C^{-1}q(T)C$ for every polynomial $q$, so $S$ and $T$ have the same minimal polynomial.
>
> (b) For every $\lambda \in \mathbf{F}$ and $j \ge 0$, $C^{-1}$ maps $\operatorname{null}(T - \lambda I)^j$ isomorphically onto $\operatorname{null}(S - \lambda I)^j$, so
>
> $$\dim\operatorname{null}(S - \lambda I)^j = \dim\operatorname{null}(T - \lambda I)^j.$$

{: .prompt-proof }
> (a) is the [conjugation row]({% post_url 2026-07-22-polynomials %}#homomorphism-gives-divisibility-injectivity-gives-equality) of the table in Polynomials: $T \mapsto C^{-1}TC$ is an algebra isomorphism, so it commutes with substituting into $q$, and $q(S) = 0$ iff $q(T) = 0$.
>
> (b) By (a) with $q(z) = (z - \lambda)^j$, $(S - \lambda I)^j = C^{-1}(T - \lambda I)^jC$. So $(S - \lambda I)^jx = 0$ iff $(T - \lambda I)^j(Cx) = 0$, that is, $x \in \operatorname{null}(S - \lambda I)^j$ iff $Cx \in \operatorname{null}(T - \lambda I)^j$. Hence $C^{-1}$ maps the second null space onto the first, injectively since $C^{-1}$ is invertible. $\blacksquare$

Every invariant met so far in the series is read off from (a) and (b):

| invariant | read off from | defined in |
| --- | --- | --- |
| eigenvalues $\sigma(T)$ | $\lambda$ with $\dim\operatorname{null}(T - \lambda I) > 0$ | [Eigenvalues and Eigenvectors]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#eigenvalues-and-the-minimal-polynomial) |
| geometric multiplicity $g_\lambda$ | $j = 1$ | [Generalized Eigenspaces]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#generalized-eigenspaces) |
| algebraic multiplicity $d_\lambda$ | $j = n$ | [Generalized Eigenspaces]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#generalized-eigenspaces) |
| minimal polynomial $p_T$ and its exponents $e_\lambda$ | (a) | [Polynomials]({% post_url 2026-07-22-polynomials %}#minimal-polynomial) |
| characteristic polynomial $\prod_\lambda (z - \lambda)^{d_\lambda}$ (split case) | the $d_\lambda$ | [Generalized Eigenspaces]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#generalized-eigenspaces) |
| diagram $\mu^{(\lambda)}$ of the nilpotent part at each $\lambda$ | all $j$: $\mu^{(\lambda)\prime}_j$ is the jump from $j-1$ to $j$ | [The Nilpotent Part]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#the-nilpotent-part) |
| $\dim\mathbf{F}[T] = \deg p_T$ | (a) | [Commuting Operators]({% post_url 2026-08-01-commuting-operators %}#basic-facts) |
| $\dim\mathcal{C}(T)$ | $\mathcal{C}(C^{-1}TC) = C^{-1}\mathcal{C}(T)C$ | [Commuting Operators]({% post_url 2026-08-01-commuting-operators %}#back-to-arbitrary-operators) |

The diagram row is the finest: every other row in the split case is computed from the eigenvalues and their diagrams. The diagram of the nilpotent part $N_\lambda$ is determined by $\dim\operatorname{null}N_\lambda^j$, and $\operatorname{null}N_\lambda^j = \operatorname{null}(T - \lambda I)^j$ because these null spaces already lie in $G(\lambda, T)$.

{: .prompt-tip }
> The trace and the determinant are also similarity invariants: in the split case they are the sum and the product of the eigenvalues, each repeated $d_\lambda$ times, so they are read off from the characteristic polynomial. Like the characteristic polynomial, they are far from complete: $I$ and $J_2(1)$ on $\mathbf{F}^2$ share both, yet $I$ is similar only to itself.

## Classification When the Minimal Polynomial Splits

The diagrams are not only the finest invariant on the list. Together with the eigenvalues, they are a **complete** invariant: they decide similarity.

{: .prompt-info }
> Suppose the minimal polynomials of $S$ and $T$ split over $\mathbf{F}$ (automatic when $\mathbf{F} = \mathbb{C}$). The following are equivalent:
>
> (a) $S \sim T$.
>
> (b) $\dim\operatorname{null}(S - \lambda I)^j = \dim\operatorname{null}(T - \lambda I)^j$ for every $\lambda \in \mathbf{F}$ and every $j \ge 1$.
>
> (c) $\sigma(S) = \sigma(T)$, and for each $\lambda$ the nilpotent parts of $S$ and $T$ at $\lambda$ have the same diagram.

{: .prompt-proof }
> (a) $\Rightarrow$ (b) is part (b) of the previous theorem.
>
> (b) $\Rightarrow$ (c): taking $j = 1$, the same $\lambda$ have nonzero eigenspaces, so $\sigma(S) = \sigma(T)$. For each $\lambda$, the nilpotent parts have the same $d_j = \dim\operatorname{null}(\,\cdot\, - \lambda I)^j$ for every $j$, and the diagram is determined by these through [$\mu'_j = d_j - d_{j-1}$]({% post_url 2026-07-25-nilpotent-operators %}#jordan-basis).
>
> (c) $\Rightarrow$ (a): since the minimal polynomials split, $V = \bigoplus_{\lambda \in \sigma(T)} G(\lambda, S) = \bigoplus_{\lambda \in \sigma(T)} G(\lambda, T)$. Fix $\lambda$ with diagram $\mu = (m_1, \dots, m_k)$. The nilpotent parts of $S$ on $G(\lambda, S)$ and of $T$ on $G(\lambda, T)$ have [Jordan bases]({% post_url 2026-07-25-nilpotent-operators %}#jordan-basis) of the same shape, and with respect to them both have the matrix $J_{m_1}(0) \oplus \dots \oplus J_{m_k}(0)$. Adding $\lambda I$, the restrictions of $S$ and $T$ both have the matrix $J_{m_1}(\lambda) \oplus \dots \oplus J_{m_k}(\lambda)$. Concatenating these bases over all $\lambda$ gives a basis of $V$ for $S$ and one for $T$ with respect to which the two operators have the same block diagonal matrix, so $S \sim T$ by the [characterization of similarity](#change-of-basis-and-similarity). $\blacksquare$

Condition (b) is a finite computation: $\lambda$ only needs to range over the eigenvalues, and $j$ only up to $n$, after which the null spaces stop growing. In matrix terms, it compares the ranks of $(A - \lambda I)^j$ and $(B - \lambda I)^j$.

The proof of (c) $\Rightarrow$ (a) also produces a canonical representative of each similarity class.

{: .prompt-info }
> *Jordan form.* If the minimal polynomial of $T$ splits, then $T$ has a matrix of the form
>
> $$\bigoplus_{\lambda \in \sigma(T)} \Big( J_{m^{(\lambda)}_1}(\lambda) \oplus \dots \oplus J_{m^{(\lambda)}_{k_\lambda}}(\lambda) \Big),$$
>
> where $\mu^{(\lambda)} = \big(m^{(\lambda)}_1 \ge \dots \ge m^{(\lambda)}_{k_\lambda}\big)$ is the diagram of the nilpotent part at $\lambda$. Two such operators are similar iff their Jordan forms agree up to the order of the eigenvalues.

### When the Polynomials Suffice

The classification explains when the cheap invariants, the two polynomials, already decide similarity. The characteristic polynomial records the eigenvalues and each $d_\lambda$, the number of boxes in the diagram at $\lambda$; the minimal polynomial records each $e_\lambda$, the [height of its tallest column]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#the-nilpotent-part). So the two polynomials are a complete invariant exactly when no two diagrams with the same number of boxes have the same tallest column.

{: .prompt-info }
> Suppose the minimal polynomials of $S$ and $T$ split.
>
> (a) If $S$ and $T$ are diagonalizable, then $S \sim T$ iff they have the same characteristic polynomial.
>
> (b) If $n \le 3$, then $S \sim T$ iff they have the same characteristic and minimal polynomials.

{: .prompt-proof }
> (a) A diagonalizable operator has every diagram a [single level]({% post_url 2026-07-25-nilpotent-operators %}#one-level-the-zero-operator), $\mu^{(\lambda)} = (1, \dots, 1)$ with $d_\lambda$ boxes. So the diagrams are determined by the $d_\lambda$, which the characteristic polynomial records.
>
> (b) Each diagram has $d_\lambda \le 3$ boxes. The partitions of $1$, $2$, $3$ are $(1)$; $(2), (1,1)$; $(3), (2,1), (1,1,1)$, and within each size the largest part tells them apart. So each diagram is determined by $d_\lambda$ and $e_\lambda$. $\blacksquare$

At $n = 4$ this fails for the first time: the nilpotent diagrams $(2,2)$ and $(2,1,1)$ have the same number of boxes and the same tallest column, so they share both polynomials, but they are [not similar]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#the-nilpotent-part).

### A Matrix Is Similar to Its Transpose

{: .prompt-info }
> If the minimal polynomial of $A \in \mathbf{F}^{n,n}$ splits, then $A \sim A^{t}$.

{: .prompt-proof }
> By the Jordan form, $A = C^{-1}JC$ with $J = \bigoplus J_m(\lambda)$, so $A^{t} = C^{t}J^{t}(C^{t})^{-1}$ is similar to $J^{t} = \bigoplus J_m(\lambda)^{t}$. It is enough to show $J_m(\lambda)^{t} \sim J_m(\lambda)$ for each block. Let $P$ be the $m \times m$ matrix with $1$s on the antidiagonal, so $P^{-1} = P$ and $PXP$ has entries $(PXP)_{i,j} = X_{m+1-i,\,m+1-j}$: it reverses the order of the basis. The superdiagonal entries $(i, i+1)$ of $J_m(\lambda)$ move to positions $(m+1-i, m-i)$, which is the subdiagonal, and the diagonal stays put. So $PJ_m(\lambda)P = J_m(\lambda)^{t}$. $\blacksquare$

For operators this says $T$ and its [dual map]({% post_url 2026-07-10-dual-space %}) $T'$ have similar matrices, since [$\mathcal{M}(T') = \mathcal{M}(T)^t$]({% post_url 2026-06-11-linear-algebra %}#matrices) for dual bases. The statement holds over every field; that needs the rational canonical form below.

## Real Operators

Over $\mathbb{R}$ the minimal polynomial need not split, and rotations have no Jordan form. [Complexification]({% post_url 2026-08-03-complexification %}) resolves this: it does not change whether two real operators are similar.

{: .prompt-info }
> Suppose $A, B \in \mathbb{R}^{n,n}$. If $A$ and $B$ are similar over $\mathbb{C}$, they are similar over $\mathbb{R}$.

{: .prompt-proof }
> Let $P \in \mathbb{C}^{n,n}$ be invertible with $PA = BP$, and write $P = X + iY$ with $X, Y$ real. Since $A$ and $B$ are real, comparing real and imaginary parts gives $XA = BX$ and $YA = BY$, so $(X + tY)A = B(X + tY)$ for every real $t$.
>
> It remains to make $X + tY$ invertible for some real $t$. The function $f(t) = \det(X + tY)$ is a polynomial in $t$, and $f(i) = \det P \ne 0$, so $f$ is not the zero polynomial and has finitely many zeros. Choose a real $t_0$ that is not one of them. Then $Q = X + t_0Y$ is real and invertible with $QA = BQ$, so $A = Q^{-1}BQ$. $\blacksquare$

With respect to a basis of $V$, a real operator $T$ and its complexification $T_{\mathbb{C}}$ have the [same matrix]({% post_url 2026-08-03-complexification %}#linear-map). So two real operators are similar iff their complexifications are, and the classification above, applied over $\mathbb{C}$, decides it: compare the eigenvalues of $S_{\mathbb{C}}$ and $T_{\mathbb{C}}$, now possibly non-real, and their diagrams.

{: .prompt-tip }
> For a real $T$, the complex Jordan form is symmetric under conjugation. Conjugation on $V_{\mathbb{C}}$ satisfies $C(T_{\mathbb{C}} - \lambda I)^j = (T_{\mathbb{C}} - \bar\lambda I)^jC$, as in [Complexification]({% post_url 2026-08-03-complexification %}#eigenvalue), so it carries each $\operatorname{null}(T_{\mathbb{C}} - \lambda I)^j$ onto $\operatorname{null}(T_{\mathbb{C}} - \bar\lambda I)^j$ and the diagrams at $\lambda$ and $\bar\lambda$ agree. Pairing each block $J_m(\lambda)$ with its conjugate $J_m(\bar\lambda)$ and choosing real bases of the paired spaces gives the **real Jordan form**, where each pair becomes a $2m \times 2m$ block built from rotation-scaling $2 \times 2$ blocks.

## Other Fields

Over a field where the minimal polynomial need not split and no algebraically closed field is at hand, the eigenvalues are replaced by the monic irreducible factors of the minimal polynomial. The classification keeps the same shape.

{: .prompt-tip }
> *Rational canonical form.* For arbitrary $\mathbf{F}$, $S \sim T$ iff
>
> $$\dim\operatorname{null}q(S)^j = \dim\operatorname{null}q(T)^j \quad\text{for every monic irreducible } q \in \mathcal{P}(\mathbf{F}) \text{ and every } j \ge 1.$$
>
> For $q = z - \lambda$ these are the conditions in (b) above. The canonical representative replaces Jordan blocks by companion matrices of the powers $q^m$, using the [primary decomposition]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#eigenvalues-and-the-minimal-polynomial) $V = \bigoplus_q \operatorname{null}q(T)^{r_q}$ in place of the generalized eigenspaces. The proof goes through the structure theorem for finitely generated modules over the polynomial ring $\mathbf{F}[z]$, which is beyond this series. Two consequences: $A \sim A^t$ over every field, and two matrices over $\mathbf{F}$ that are similar over a larger field are already similar over $\mathbf{F}$, generalizing the real case above.

## Similarity Classes

The **similarity class** of $T$ is the set of all operators similar to it, $\\{C^{-1}TC : C \text{ invertible}\\}$. The classification says the classes are indexed by the data in (c): a set of eigenvalues with a diagram for each. [Commuting Operators]({% post_url 2026-08-01-commuting-operators %}#freedom-in-a-jordan-basis) measures the size of a class by a dimension count: the invertible $C$ have $n^2$ degrees of freedom, and $C$ and $D$ give the same operator exactly when $D^{-1}C$ commutes with $T$, so the class has dimension

$$n^2 - \dim\mathcal{C}(T).$$

Since $\dim\mathcal{C}(T)$ is computed from the diagrams, so is the size of each class. A scalar operator $\lambda I$ has the smallest class, the single point $\\{\lambda I\\}$; an operator whose minimal and characteristic polynomials coincide has the largest, of dimension $n^2 - n$.
