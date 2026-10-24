---
title:  "Polynomials"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
---

## Notation

{% include notation-table.md keys="F V n LV PF min_poly poly_alg dual_op quotient_op complexification_op" %}

## Polynomials

Two polynomials are **coprime** if their only common divisors are the nonzero constants.

{: .prompt-info }
> _Bézout for several polynomials_
>
> Suppose $ p_1, \dots, p_m \in \mathcal{P}(\mathbf{F}) $ have no common nonconstant divisor. Then there exist $ r_1, \dots, r_m \in \mathcal{P}(\mathbf{F}) $ such that
>
> $$ r_1p_1 + \dots + r_mp_m = 1. $$
>
> Over $\mathbb{C}$, the hypothesis says exactly that $p_1, \dots, p_m$ have no zero common to all of them.

{: .prompt-proof }
> Let $I$ be the set of all combinations $r_1p_1 + \dots + r_mp_m$. It is closed under addition and under multiplication by any polynomial, and it contains a nonzero polynomial, since the hypothesis rules out $p_1 = \dots = p_m = 0$. Let $d$ be a monic polynomial of smallest degree in $I$. Divide each $p_i$ by $d$: $p_i = a_id + s_i$ with $\deg s_i < \deg d$. Then $s_i = p_i - a_id \in I$, so $s_i = 0$ by the choice of $d$. Thus $d$ divides every $p_i$, so $d$ is constant by hypothesis, that is, $1 = d \in I$.
>
> Over $\mathbb{C}$, a common nonconstant divisor has a zero by the fundamental theorem of algebra, and that zero is common to all the $p_i$. Conversely, a common zero $\lambda$ gives the common divisor $z - \lambda$. $\blacksquare$

{: .prompt-tip }
> *Euclid's lemma.* If $p$ and $q$ are coprime and $p \mid qr$, then $p \mid r$. Indeed, write $ap + bq = 1$; then $r = a\,rp + b\,qr$, and $p$ divides both terms.

For two polynomials over $\mathbb{C}$, linear algebra gives Bézout's identity with control over the degrees.

{: .prompt-info }
> _Bézout with degree bounds_
>
> Suppose $ p, q \in \mathcal{P}(\mathbb{C}) $ are nonconstant polynomials with no zeros in common. Let $ m = \deg p $ and $ n = \deg q $. There exist unique $$ r \in \mathcal{P}_{n - 1}(\mathbb{C}) $$ and $$ s \in \mathcal{P}_{m - 1}(\mathbb{C}) $$ such that
>
> $$ rp + sq = 1. $$

{: .prompt-proof }
> Define $$ \Gamma: \mathcal{P}_{n - 1}(\mathbb{C}) \times \mathcal{P}_{m - 1}(\mathbb{C}) \to \mathcal{P}_{m + n - 1}(\mathbb{C}) $$ by
>
> $$ \Gamma(r,s) = rp + sq. $$
>
> It lands in the stated codomain because $\deg(rp) \le (n - 1) + m$ and $\deg(sq) \le (m - 1) + n$, and it is linear in $(r,s)$ since $p, q$ are fixed. Domain and codomain have the same dimension:
>
> $$
> \dim\big(\mathcal{P}_{n-1}(\mathbb{C}) \times \mathcal{P}_{m-1}(\mathbb{C})\big) = n + m = \dim \mathcal{P}_{m+n-1}(\mathbb{C}).
> $$
>
> So $\Gamma$ is bijective as soon as it is injective. Suppose $\Gamma(r,s) = 0$, that is, $rp = -sq$. Then $q \mid rp$, and $q$ is coprime to $p$ because they have no common zeros, so $q \mid r$ by Euclid's lemma. But $\deg r \le n - 1 < \deg q$, and the only multiple of $q$ of degree below $\deg q$ is $0$. Hence $r = 0$, then $sq = 0$ with $q \neq 0$, so $s = 0$.
>
> Since $m, n \ge 1$, the constant polynomial $1$ lies in the codomain, so $1 = \Gamma(r,s)$ for exactly one pair $(r,s)$. $\blacksquare$

{: .prompt-info }
> Suppose $ p \in \mathcal{P}(\mathbb{C}) $ has degree $ m \ge 1 $. The following are equivalent:
>
> (a) $ p $ has $ m $ distinct zeros.
>
> (b) $ p $ and its derivative $ p' $ have no zeros in common.
>
> (c) $ p $ and $ p' $ are coprime.

{: .prompt-proof }
> (b) $\iff$ (c) is the $\mathbb{C}$ case of Bézout above.
>
> (a) $\iff$ (b). Let $\lambda$ be a zero of $p$ of multiplicity $k \ge 1$, so $p = (z - \lambda)^k h$ with $h(\lambda) \neq 0$. Then
>
> $$ p' = (z - \lambda)^{k-1}\big(k\,h + (z - \lambda)h'\big), $$
>
> so $p'(\lambda) = 0$ iff $k \ge 2$. Thus (b) says every zero of $p$ is simple. Since $p$ has $m$ zeros counted with multiplicity, that is (a). $\blacksquare$

## Minimal Polynomial

From now on, $V$ is finite-dimensional, $T \in \mathcal{L}(V)$, and $n = \dim V$.

{: .prompt-info }
> The **minimal polynomial** of $ T $ is the _unique_ monic polynomial $ p_T \in \mathcal{P}(\mathbf{F}) $ of smallest degree such that $ p_T(T) = 0 $.
>
> Moreover, $ \deg p_T \le n $ (proved [below](#local-minimal-polynomial)).

{: .prompt-proof }
> *Existence.* $\dim \mathcal{L}(V) = n^2$, so the $n^2 + 1$ operators $I, T, \dots, T^{n^2}$ are linearly dependent. A nontrivial dependence is a nonzero polynomial $q$ with $q(T) = 0$. Among such polynomials, take one of smallest degree and divide it by its leading coefficient.
>
> *Uniqueness.* If $p$ and $\tilde p$ both qualify, then $p - \tilde p$ annihilates $T$ and has smaller degree, since the leading terms cancel. Dividing by its leading coefficient would give a monic annihilator of smaller degree, so $p - \tilde p = 0$. $\blacksquare$

{: .prompt-info }
> For $ q \in \mathcal{P}(\mathbf{F}) $, $ q(T) = 0 \iff p_T \mid q $.

{: .prompt-proof }
> ($\Leftarrow$) If $q = a\,p_T$, then $q(T) = a(T)\,p_T(T) = 0$.
>
> ($\Rightarrow$) Divide: $q = a\,p_T + r$ with $\deg r < \deg p_T$. Then $r(T) = q(T) - a(T)p_T(T) = 0$. If $r \neq 0$, dividing $r$ by its leading coefficient gives a monic annihilator of degree less than $\deg p_T$, which is impossible. So $r = 0$. $\blacksquare$

{: .prompt-info }
> Let $m = \deg p_T$. Then $I, T, \dots, T^{m-1}$ is a basis of $\mathbf{F}[T]$. In particular, $\dim \mathbf{F}[T] = \deg p_T$.

{: .prompt-proof }
> $\mathbf{F}[T]$ is a subspace, because $q_1(T) + q_2(T) = (q_1 + q_2)(T)$ and $c\,q(T) = (cq)(T)$.
>
> *Spanning.* For $q \in \mathcal{P}(\mathbf{F})$, divide: $q = a\,p_T + r$ with $\deg r < m$. Then $q(T) = r(T)$, a combination of $I, T, \dots, T^{m-1}$.
>
> *Independence.* A nontrivial relation $\sum_{t<m} c_tT^t = 0$ would be a nonzero polynomial of degree less than $m$ annihilating $T$, contradicting minimality. $\blacksquare$

{: .prompt-tip }
> In the language of [quotient spaces]({% post_url 2026-07-04-quotient-spaces %}): the map $\mathcal{P}(\mathbf{F}) \to \mathcal{L}(V)$, $q \mapsto q(T)$, is linear with range $\mathbf{F}[T]$, and by the previous box its null space is the set of multiples of $p_T$. So $\mathbf{F}[T]$ is isomorphic to the quotient of $\mathcal{P}(\mathbf{F})$ by the multiples of $p_T$, and the remainders of degree less than $m$ are canonical representatives of the translates.

## Local Minimal Polynomial

The minimal polynomial kills every vector. Asking instead for the polynomials that kill one vector $v$ leads to the span of $v, Tv, T^2v, \dots$. The spans $\operatorname{span}(v, Tv, \dots, T^mv)$ are the [**Krylov subspaces**](https://en.wikipedia.org/wiki/Krylov_subspace) of $T$ at $v$, and the largest of them, which the next box shows is $\operatorname{span}(v, Tv, \dots, T^{n-1}v)$, is also called the **cyclic subspace** generated by $v$.

{: .prompt-info }
> Suppose $ v \in V $. Then
>
> $$\operatorname{span}(v, Tv, \dots, T^m v) = \operatorname{span}(v, Tv, \dots, T^{n - 1}v)$$
>
> for all integers $ m \ge n - 1 $.

{: .prompt-proof }
> If $v = 0$, every span is $\\{0\\}$. So assume $v \neq 0$, and write
>
> $$U_m := \operatorname{span}(v, Tv, \dots, T^m v), \qquad U_0 \subseteq U_1 \subseteq U_2 \subseteq \dots$$
>
> Each term is built from the previous one by a fixed rule,
>
> $$U_{m+1} = \operatorname{span}(v) + T(U_m),$$
>
> so the [stabilizing chains lemma]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#null-space-and-range-chains) applies: the chain stalls, and once it stalls it stays. Let $k \ge 1$ be the first index with $U_k = U_{k-1}$, that is, $T^kv \in U_{k-1}$. Before that, each step adds a single vector and the dimension grows, so it grows by exactly $1$. Hence $\dim U_{k-1} = k$, which says that $v, Tv, \dots, T^{k-1}v$ is linearly independent and $k \le n$.
>
> By (a) of the lemma, $U_m = U_{k-1}$ for every $m \ge k - 1$. Since $k - 1 \le n - 1$, this covers $m = n - 1$ and every $m \ge n - 1$, so $U_m = U_{k-1} = U_{n-1}$. $\blacksquare$

{: .prompt-tip }
> $U_{n-1}$ is $T$-invariant, since $T(U_{n-1}) \subseteq U_n = U_{n-1}$. It is the *smallest* $T$-invariant subspace containing $v$, because any such subspace contains every $T^jv$. So closing $v$ up under $T$ never takes more than $n$ terms.

{: .prompt-info }
> *Local minimal polynomial*
>
> Suppose $ v \in V $. There is a unique monic polynomial $ p_v $ of smallest degree such that $ p_v(T)v = 0 $.
>
> Let $U$ be the smallest $T$-invariant subspace containing $v$. Then $p_v$ is the minimal polynomial of $\left.T\right\rvert_U$, and $\deg p_v = \dim U$.

{: .prompt-proof }
> For $v = 0$, $p_v = 1$ and $U = \\{0\\}$. Otherwise take $k$ as in the proof above, so $U = U_{k-1}$ has basis $v, Tv, \dots, T^{k-1}v$ and $\dim U = k$.
>
> The relation $T^kv = \sum_{j<k} a_j T^j v$ rearranges to $q(T)v = 0$ with $q(z) = z^k - a_{k-1}z^{k-1} - \dots - a_0$, monic of degree $k$. A nonzero polynomial of degree less than $k$ cannot kill $v$, since that would be a nontrivial relation among $v, Tv, \dots, T^{k-1}v$. So the smallest degree is $k$, and uniqueness follows as for $p_T$: the difference of two candidates kills $v$ and has degree less than $k$.
>
> Finally, $p_v(T)$ commutes with each $T^j$, so $p_v(T)T^jv = T^jp_v(T)v = 0$, and $p_v(T)$ vanishes on $U$. Hence $p_{T\vert_U} \mid p_v$. Conversely $p_{T\vert_U}(T)v = 0$, so $\deg p_{T\vert_U} \ge k = \deg p_v$. Both are monic, so they are equal. $\blacksquare$

{: .prompt-info }
> For $ q \in \mathcal{P}(\mathbf{F}) $, $ q(T)v = 0 \iff p_v \mid q $. In particular, $ p_v \mid p_T $.

{: .prompt-proof }
> The same division argument as for $p_T$: if $q = a\,p_v + r$ with $\deg r < \deg p_v$ and $q(T)v = 0$, then $r(T)v = 0$, so $r = 0$. Since $p_T(T)v = 0$, $p_v \mid p_T$. $\blacksquare$

{: .prompt-info }
> If $p_u$ and $p_w$ are coprime, then $p_{u+w} = p_u p_w$.

{: .prompt-proof }
> First, $(p_up_w)(T)(u+w) = p_w(T)p_u(T)u + p_u(T)p_w(T)w = 0$, so $p_{u+w} \mid p_up_w$.
>
> Conversely, suppose $r(T)(u+w) = 0$. Apply $p_w(T)$, which kills $w$ and commutes with $r(T)$:
>
> $$(p_w r)(T)u = p_w(T)r(T)(u+w) - r(T)p_w(T)w = 0.$$
>
> So $p_u \mid p_w r$, and Euclid's lemma gives $p_u \mid r$. Symmetrically $p_w \mid r$. Since $p_u$ and $p_w$ are coprime, $p_up_w \mid r$ (write $r = p_u s$; then $p_w \mid p_u s$ gives $p_w \mid s$). Taking $r = p_{u+w}$ finishes it. $\blacksquare$

{: .prompt-info }
> There exists $v \in V$ with $p_v = p_T$. Consequently $\deg p_T \le n$.

{: .prompt-proof }
> Factor $p_T = q_1^{m_1}\cdots q_k^{m_k}$ into powers of distinct monic irreducibles. Fix $i$.
>
> Since $\deg(p_T/q_i) < \deg p_T$, minimality gives $(p_T/q_i)(T) \neq 0$, so choose $u_i$ with $(p_T/q_i)(T)u_i \neq 0$. Set
>
> $$w_i = \big(p_T/q_i^{m_i}\big)(T)\,u_i.$$
>
> Then $q_i^{m_i}(T)w_i = p_T(T)u_i = 0$, so $p_{w_i} \mid q_i^{m_i}$. And $q_i^{m_i-1}(T)w_i = (p_T/q_i)(T)u_i \neq 0$, so $p_{w_i} \nmid q_i^{m_i-1}$. As $q_i$ is irreducible, the monic divisors of $q_i^{m_i}$ are the powers $q_i^j$ with $j \le m_i$, so $p_{w_i} = q_i^{m_i}$.
>
> The polynomials $q_1^{m_1},\dots,q_k^{m_k}$ are pairwise coprime, and a product of some of them is coprime to any other. So applying the previous box $k - 1$ times, $v = w_1 + \dots + w_k$ gives
>
> $$p_v = p_{w_1}\cdots p_{w_k} = q_1^{m_1}\cdots q_k^{m_k} = p_T.$$
>
> Finally $\deg p_T = \deg p_v = \dim U \le n$, where $U$ is the smallest $T$-invariant subspace containing $v$. $\blacksquare$

### Cyclic Vectors

A vector $v$ is a **cyclic vector** for $T$ if $v, Tv, \dots, T^{n-1}v$ is a basis of $V$, that is, if the cyclic subspace generated by $v$ (the smallest $T$-invariant subspace containing $v$) is all of $V$. In other words, the Krylov subspaces of $T$ at $v$ grow all the way to $V$, and $v, Tv, \dots, T^{n-1}v$ is then called a **Krylov basis**. With respect to it, the matrix of $T$ is the companion matrix of $p_T$ (see the box below): $T$ shifts each basis vector to the next, and $T(T^{n-1}v) = T^nv$ is expanded using the coefficients of $p_v = p_T$.

{: .prompt-info }
> (a) $v$ is a cyclic vector $\iff \deg p_v = n$, and then $p_v = p_T$.
>
> (b) $T$ has a cyclic vector $\iff \deg p_T = n$.

{: .prompt-proof }
> (a) $\deg p_v$ is the dimension of the smallest $T$-invariant subspace containing $v$, which is $V$ iff $\deg p_v = n$. In that case $p_v \mid p_T$ and $\deg p_T \le n$, so the two monic polynomials are equal.
>
> (b) ($\Rightarrow$) is (a). ($\Leftarrow$) Take $v$ with $p_v = p_T$; then $\deg p_v = n$, and (a) applies. $\blacksquare$

{: .prompt-tip }
> **Over $\mathbb{C}$, (b) is a statement about Jordan blocks.** This looks ahead to [Nilpotent Operators]({% post_url 2026-07-25-nilpotent-operators %}) and [Eigenvalues and Eigenvectors]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#the-nilpotent-part). There $q_i = z - \lambda_i$ and $m_i = e_{\lambda_i}$, so $\deg p_T = \sum_i e_{\lambda_i}$, while $n = \sum_i d_{\lambda_i}$ with $d_{\lambda} = \dim G(\lambda,T)$. Since always $e_\lambda \le d_\lambda$, the condition $\deg p_T = n$ forces $e_\lambda = d_\lambda$ for every $\lambda$: the tallest column of each diagram holds all its boxes, so there is **one Jordan block per eigenvalue**. In that case the $w_i$ in the proof above is a top of the $i$-th chain, and the cyclic vector $v$ is the sum of the tops.

Every monic polynomial arises as a minimal polynomial, and the operator can be chosen with a cyclic vector.

{: .prompt-info }
> *Companion matrix*
>
> Let $p(z) = a_0 + a_1z + \dots + a_{n-1}z^{n-1} + z^n$ with $n \ge 1$. Define $T \in \mathcal{L}(\mathbf{F}^n)$ by
>
> $$Te_k = e_{k+1} \quad (1 \le k < n), \qquad Te_n = -a_0e_1 - a_1e_2 - \dots - a_{n-1}e_n.$$
>
> Then $e_1$ is a cyclic vector for $T$, and $p_T = p$.

{: .prompt-proof }
> $T^je_1 = e_{j+1}$ for $j < n$, so $e_1$ is cyclic. Moreover $T^ne_1 = Te_n = -\sum_{j<n} a_jT^je_1$, that is, $p(T)e_1 = 0$. So $p_{e_1} \mid p$, and both are monic of degree $n$, so $p_{e_1} = p$. By (a) of the previous box, $p_T = p_{e_1} = p$. $\blacksquare$

{: .prompt-tip }
> The matrix of $T$ is the [companion matrix](https://en.wikipedia.org/wiki/Companion_matrix) of $p$ (see also [MathWorld](https://mathworld.wolfram.com/CompanionMatrix.html)). Since $\deg p_T = n$, $p$ is also the characteristic polynomial of $T$. In the split case this is the Jordan tip above: $e_\lambda = d_\lambda$ for every $\lambda$, so $\prod_\lambda(z - \lambda)^{e_\lambda} = \prod_\lambda(z - \lambda)^{d_\lambda}$.

{: .prompt-tip }
> *Computing $p_T$*
>
> **Directly.** Find the smallest positive integer $m$ such that
>
> $$ c_0I + c_1T + \dots + c_{m-1}T^{m-1} = -T^m $$
>
> has a solution $c_0, \dots, c_{m-1} \in \mathbf{F}$. In a basis, this is a system of $n^2$ linear equations (one per matrix entry) in $m$ unknowns. Solve it by Gaussian elimination for $m = 1, 2, \dots$; a solution appears by $m = n$, and then $p_T = z^m + c_{m-1}z^{m-1} + \dots + c_0$.
>
> **Via one vector: Krylov's method (usually faster).** This is how [A. N. Krylov](https://webspace.science.uu.nl/~vorst102/kryl.html) computed characteristic polynomials in 1931, and the reason Krylov subspaces carry his name. Pick $v \neq 0$ and solve the $n \times n$ system, whose coefficient matrix has the Krylov vectors $v, Tv, \dots, T^{n-1}v$ as its columns:
>
> $$ c_0v + c_1Tv + \dots + c_{n-1}T^{n-1}v = -T^nv. $$
>
> It has a unique solution exactly when $v, Tv, \dots, T^{n-1}v$ is a basis, that is, when $v$ is a cyclic vector. This is the typical case. Then $p_T = p_v = z^n + c_{n-1}z^{n-1} + \dots + c_0$ by the box above. Otherwise, fall back on the direct method, or use $p_T = \operatorname{lcm}(p_{v_1}, \dots, p_{v_n})$ for a basis $v_1, \dots, v_n$: $q(T) = 0$ iff $q(T)v_i = 0$ for every $i$, iff every $p_{v_i}$ divides $q$.

## Minimal Polynomials of Related Operators

Many operators are built from $T$: its restriction to an invariant subspace, the quotient operator, the complexification, the dual, a conjugate $S^{-1}TS$, the inverse, and the products $ST$ and $TS$. In most of these cases a map that respects the algebra of operators carries $T$ to the new operator, and that map controls the minimal polynomial.

### Algebra Homomorphisms

{: .prompt-info }
> Let $\mathcal{A} \subseteq \mathcal{L}(V)$ be a subspace closed under composition with $I \in \mathcal{A}$, and $T \in \mathcal{A}$. Let $\Phi: \mathcal{A} \to \mathcal{L}(W)$ be a unital [algebra homomorphism](https://en.wikipedia.org/wiki/Algebra_over_a_field#Algebra_homomorphisms) (linear, $\Phi(I) = I$, $\Phi(AB) = \Phi(A)\Phi(B)$) or anti-homomorphism ($\Phi(AB) = \Phi(B)\Phi(A)$). Then
>
> (a) $q(\Phi(T)) = \Phi(q(T))$ for every $q \in \mathcal{P}(\mathbf{F})$;
>
> (b) $p_{\Phi(T)} \mid p_T$;
>
> (c) if $\Phi$ is injective, $p_{\Phi(T)} = p_T$.

{: .prompt-proof }
> (a) $\Phi(T^k) = \Phi(T)^k$ by multiplicativity; for an anti-homomorphism too, since all the factors are equal. With $\Phi(I) = I$ and linearity, $\Phi\big(\sum_k a_kT^k\big) = \sum_k a_k\Phi(T)^k$.
>
> (b) By (a), $p_T(\Phi(T)) = \Phi(p_T(T)) = \Phi(0) = 0$, so $p_{\Phi(T)} \mid p_T$.
>
> (c) By (a), $\Phi\big(p_{\Phi(T)}(T)\big) = p_{\Phi(T)}(\Phi(T)) = 0$, so injectivity gives $p_{\Phi(T)}(T) = 0$ and $p_T \mid p_{\Phi(T)}$. Two monic polynomials that divide each other are equal. $\blacksquare$

In short: a homomorphism gives divisibility, and injectivity on top of that gives equality. Below, $\mathcal{L}_U(V)$ denotes the operators on $V$ for which $U$ is invariant, and $U$ is assumed invariant under $T$.

| $\Phi(T)$                                   | $q(\Phi(T))$                                    | Type of $\Phi$                                                                                          | $\Phi$ injective? | Minimal polynomial                    |
| ------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------- |
| $T\vert_U$                                  | $q(T\vert_U) = q(T)\vert_U$                     | unital algebra hom. $\mathcal{L}_U(V) \to \mathcal{L}(U)$                                               | no                | $p_{T\vert_U} \mid p_T$               |
| $T/U$                                       | $q(T/U) = q(T)/U$                               | unital algebra hom. $\mathcal{L}_U(V) \to \mathcal{L}(V/U)$                                             | no                | $p_{T/U} \mid p_T$                    |
| $\mathbf{F} = \mathbb{R}$, $T_{\mathbb{C}}$ | $q(T_\mathbb{C}) = (q(T))_\mathbb{C}$, $q$ real | unital $\mathbb{R}$-algebra hom. $$\mathcal{L}_\mathbb{R}(V) \to \mathcal{L}_\mathbb{C}(V_\mathbb{C})$$ | yes               | $p_{T_\mathbb{C}} = p_T$              |
| $T'$                                        | $q(T') = (q(T))'$                               | unital **anti**-homomorphism $\mathcal{L}(V) \to \mathcal{L}(V')$                                       | yes               | $p_{T'} = p_T$                        |
| $S^{-1}TS$, $S$ invertible                  | $q(S^{-1}TS) = S^{-1}q(T)S$                     | unital algebra **iso** $\mathcal{L}(V) \to \mathcal{L}(V)$                                              | yes (bijective)   | $p_{S^{-1}TS} = p_T$                  |

{: .prompt-warning }
> The complexification row needs more than (c). The map $T \mapsto T_{\mathbb{C}}$ is only $\mathbb{R}$-linear, so (a) holds only for real $q$, and (c) only shows that $T$ and $T_{\mathbb{C}}$ have the same *real* annihilators. That $p_{T_{\mathbb{C}}}$ is real to begin with is the content of the theorem in [Complexification]({% post_url 2026-07-30-complexification %}#polynomial).

The restriction and the quotient each lose information, but together they recover $T$ up to a product.

{: .prompt-info }
> If $U$ is invariant under $T$, then $p_T \mid p_{T\vert_U}\, p_{T/U}$.

{: .prompt-proof }
> Write $a = p_{T\vert_U}$ and $b = p_{T/U}$. For $v \in V$, $b(T)v + U = b(T/U)(v + U) = 0$ in $V/U$, so $b(T)v \in U$. Then $a(T)b(T)v = a(T\vert_U)\,b(T)v = 0$. So $(ab)(T) = 0$, and $p_T \mid ab$. $\blacksquare$

### Inverse

No unital algebra homomorphism sends $T$ to $T^{-1}$ in general: by (a) it would force $q(T) = 0 \Rightarrow q(T^{-1}) = 0$, which fails already for $T = 2I$ and $q = z - 2$. The substitute is the **reversal** $q^{\mathrm{rev}}(z) = z^{\deg q}\,q(1/z)$, which is $q$ with its coefficients read backwards.

{: .prompt-info }
> Suppose $ T $ is invertible. Then $ p_T(0) \ne 0 $ and
>
> $$p_{T^{-1}} = \frac{p_T^{\mathrm{rev}}}{p_T(0)}.$$

{: .prompt-proof }
> For $q = \sum_{k=0}^d a_k z^k$ of degree $d$, $q^{\mathrm{rev}}(T^{-1}) = \sum_k a_k T^{k-d} = T^{-d}q(T)$, so $q(T) = 0 \iff q^{\mathrm{rev}}(T^{-1}) = 0$.
>
> If $p_T(0) = 0$, then $p_T = zh$, and $T\,h(T) = 0$ gives $h(T) = 0$ because $T$ is invertible, contradicting minimality. So $p_T^{\mathrm{rev}}$ has degree $\deg p_T$ and leading coefficient $p_T(0)$, and $p_T^{\mathrm{rev}}/p_T(0)$ is a monic annihilator of $T^{-1}$. Hence $\deg p_{T^{-1}} \le \deg p_T$. Swapping $T$ and $T^{-1}$ gives the reverse inequality, so the monic $p_{T^{-1}}$ divides a monic polynomial of the same degree, and they are equal. $\blacksquare$

{: .prompt-tip }
> $ p_T(0) \ne 0 $ is exactly the statement that $0$ is not an eigenvalue. Taking $q = (z - \lambda)^j$ with $\lambda \ne 0$, the reversal is $(1 - \lambda z)^j = (-\lambda)^j (z - \tfrac{1}{\lambda})^j$, so the identity reads $(T^{-1} - \tfrac{1}{\lambda}I)^j = (-\lambda)^{-j}\,T^{-j}(T - \lambda I)^j$. The two sides have the same null space, so [$E$ and $G$]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#generalized-eigenspaces) of $T^{-1}$ at $\tfrac{1}{\lambda}$ equal those of $T$ at $\lambda$ ($j = 1$ and $j = n$).

### $ST$ and $TS$

Let $S, T \in \mathcal{L}(V)$. The map $ST \mapsto TS$ is not a homomorphism, but an intertwining identity still transfers annihilating polynomials, at the cost of one factor of $z$.

{: .prompt-info }
> *Intertwining identity*
>
> For every polynomial $q$,
>
> $$T\, q(ST) \;=\; q(TS)\, T.$$

{: .prompt-proof }
> First for monomials: $T(ST)^k = (TS)^k T$, by induction on $k$. The case $k = 0$ is $T = T$. Assuming it for $k$,
>
> $$T(ST)^{k+1} = (TS)\,T\,(ST)^{k} = (TS)\,(TS)^k T = (TS)^{k+1}T,$$
>
> where the first step regroups $T(ST)(ST)^k$. Both sides of the identity are linear in $q$, so it extends from monomials to all polynomials. $\blacksquare$

{: .prompt-info }
> If $q$ annihilates $ST$, then $z\,q(z)$ annihilates $TS$. Consequently
>
> $$p_{TS} \;\big|\; z\,p_{ST} \qquad\text{and}\qquad p_{ST} \;\big|\; z\,p_{TS}.$$

{: .prompt-proof }
> Suppose $q(ST) = 0$. By the identity, $q(TS)\,T = T\,q(ST) = 0$. Multiply on the right by $S$:
>
> $$q(TS)\,TS = 0.$$
>
> This says exactly that $z\,q(z)$ evaluated at $TS$ is $0$. Applying this to $q = p_{ST}$ gives $p_{TS} \mid z\,p_{ST}$. Swapping the roles of $S$ and $T$ gives the other divisibility. $\blacksquare$

{: .prompt-tip }
> $ST$ and $TS$ have the same nonzero eigenvalues. The eigenvalues are the [zeros of the minimal polynomial]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#eigenvalues-and-the-minimal-polynomial), and by the divisibilities a nonzero zero of either of $p_{ST}, p_{TS}$ is a zero of the other.

{: .prompt-info }
> $ST$ is nilpotent $\iff$ $TS$ is nilpotent. Writing $i(\cdot)$ for the index of nilpotency,
>
> $$\big|\, i(ST) - i(TS) \,\big| \;\le\; 1.$$

{: .prompt-proof }
> If $(ST)^k = 0$, then
>
> $$(TS)^{k+1} = \big((TS)^k T\big) S = \big(T (ST)^k\big) S = 0.$$
>
> This is the previous box with $q = z^k$. So $i(TS) \le i(ST) + 1$, and symmetrically $i(ST) \le i(TS) + 1$. $\blacksquare$

{: .prompt-warning }
> The factor of $z$ cannot be dropped, and the bound on the indices is attained. Take
>
> $$S = \begin{pmatrix} 0 & 0 \\ 0 & 1\end{pmatrix}, \qquad
>   T = \begin{pmatrix} 0 & 1 \\ 0 & 0\end{pmatrix}.$$
>
> Then $ST = 0$, so $p_{ST} = z$ and $i(ST) = 1$, while $$TS = \begin{pmatrix} 0 & 1 \\ 0 & 0\end{pmatrix} \ne 0$$ with $(TS)^2 = 0$, so $p_{TS} = z^2$ and $i(TS) = 2$.
