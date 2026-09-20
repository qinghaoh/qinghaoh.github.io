---
title:  "Polynomials"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Notation

{% include notation-table.md keys="F V LV PF min_poly dual_op quotient_op complexification_op" %}

## Polynomials

{: .prompt-info }
> _Bézout identity_
>
> Suppose $ p, q \in \mathcal{P}(\mathbb{C}) $ are nonconstant polynomials with no zeros in common. Let $ m = \deg p $ and $ n = \deg q $. There exist $$ r \in \mathcal{P}_{n - 1}(\mathbb{C}) $$ and $$ s \in \mathcal{P}_{m - 1}(\mathbb{C}) $$ such that
>
> $$ rp + sq = 1.$$

{: .prompt-proof }
> Define $$ T: \mathcal{P}_{n - 1}(\mathbb{C}) \times \mathcal{P}_{m - 1}(\mathbb{C}) \to \mathcal{P}_{m + n - 1}(\mathbb{C}) $$ by
>
> $$ T(r,s) = rp + sq $$.
>
> $T$ is a *square* map:
>
> $$
> \dim\big(\mathcal{P}_{n-1}(\mathbb{C}) \times \mathcal{P}_{m-1}(\mathbb{C})\big) = n + m,
> $$
>
> $$
> \dim \mathcal{P}_{m+n-1}(\mathbb{C}) = m+n.
> $$
>
> Equal. And $T$ is linear: $T(r,s) = rp + sq$ is linear in $(r,s)$ since $p, q$ are fixed.
>
> Next, show $ T $ is injective. Suppose $T(r,s) = 0$, i.e.
>
> $$
> rp + sq = 0, \qquad\text{so}\qquad rp = -sq. \tag{$\ast$}
> $$
>
> We want to force $r = 0$ and $s = 0$. The **no-common-zeros** hypothesis enters here, through unique factorization / the divisibility structure of $\mathbb{C}[z]$.
>
> From $(\ast)$, $q$ divides $rp$. Now list the zeros of $q$: by the FTA, $q$ factors as $q(z) = c\prod_{i}(z - \mu_i)$ over its roots $\mu_i$ (with multiplicity). Each root $\mu_i$ of $q$ is a zero of the left side $rp$, hence a zero of $r$ or of $p$. But $p$ and $q$ share **no** zeros, so $\mu_i$ is *not* a zero of $p$ — therefore $\mu_i$ must be a zero of $r$, and by matching multiplicities (a root of $q$ of multiplicity $t$ is not absorbed by $p$ at all, so all $t$ copies must come from $r$), the **full factor** $q$ divides $r$:
>
> $$
> q \mid r.
> $$
>
> But now degrees: $r \in \mathcal{P}_{n-1}(\mathbb{C})$ so $\deg r \le n - 1 < n = \deg q$. The only multiple of $q$ with degree below $\deg q$ is the zero polynomial. Hence
>
> $$
> r = 0.
> $$
>
>Plugging back into $(\ast)$: $sq = 0$ with $q \neq 0$, so $s = 0$. Thus $\ker T = \{0\}$ and $T$ is injective.
>
> It's easy to show $T$ is invertible since $T$ is an operator.
>
> The constant polynomial $1$ lives in the codomain $$ \mathcal{P}_{m+n-1}(\mathbb{C}) $$ (its degree $0$ is $\le m + n - 1$, using that $p, q$ nonconstant gives $m, n \ge 1$, so $m + n - 1 \ge 1 \ge 0$). By surjectivity from (b), $1$ is hit: there exist $$ r \in \mathcal{P}_{n-1}(\mathbb{C}) $$ and $s \in \mathcal{P}_{m-1}(\mathbb{C})$ with
>
> $$
> T(r,s) = rp + sq = 1. \qquad\blacksquare
> $$

{: .prompt-tip }
> "no common zeros" is the $\mathbb{C}[z]$-analogue of "coprime," and $rp + sq = 1$ is exactly the statement that the gcd is a unit.

{: .prompt-info }
> Suppose $ p \in \mathcal{P}(\mathbb{C}) $ has degree $ m $.
>
> $ p $ has $ m $ distinct zeros $ \iff p $ and its derivative $ p' $ have no zeros in common $ \iff $ The greatest common divisor of $ p $ and $ p' $ is the constant polynomial $1$.

## Minimal Polynomial

{: .prompt-info }
> Suppose $ V $ is finite-dimensional and $ T \in \mathcal{L}(V) $. The minimal polynomial of $ T $ is the _unique_ monic polynomial $ p \in \mathcal{P}(\mathbf{F}) $ of smallest degree such that $ p(T) = 0 $.
>
> $ \deg p \le \dim V $.

{: .prompt-tip }
> Computation ($O((\dim V)^2)$)
>
> Find the smallest positive integer $m such that the equation
>
> $$ c_0I + c_1T + \dots + c_{m-1}T^{m-1} = -T^m $$
>
> has a solution $c_0, c_1, \dots, c_{m-1} \in \mathbf{F}$.
>
> Pick a basis of $V$ and replace $T$ in the equation above with the matrix of $T$, then the equation above can be thought of as a system of $(\dim V)^2$ linear equations in the $m$ unknowns $c_0, c_1, \dots, c_{m-1} \in \mathbf{F}$.
>
> Use Gaussian elimination or another fast method of solving systems of linear equations can tell us whether a solution exists, testing successive values $m = 1, 2, \dots, \dim V $ until a solution exists.
>
> A _usually_ faster way ($O((\dim V))$):
>
> Pick $v \in V$ with $v \ne 0$ and consider the equation
>
> to check whether the following system of $\dim V$ linear equations has a unique solution:
>
> $$ c_0v + c_1Tv + \dots + c_{\dim V-1}T^{\dim V-1}v = -T^{\dim V}v. $$
>
> Use a basis of $V$ to convert the equation above to a system of $\dim V$ linear equations in $\dim V$ unknowns $c_0, c_1, \dots, c_{\dim V -1} $.
>
> If this system of equations has a _unique_ solution $\dim V$ unknowns $c_0, c_1, \dots, c_{\dim V -1} $ (as happens most of the time), then the scalars $\dim V$ unknowns $c_0, c_1, \dots, c_{\dim V -1}, 1 $ are the coefficients of the minimal polynomial of $T$.

{: .prompt-info }
> Every monic polynomial is the minimal polynomial of some operator.

{: .prompt-tip }
> See
> * https://en.wikipedia.org/wiki/Companion_matrix
> * https://mathworld.wolfram.com/CompanionMatrix.html

{: .prompt-info }
> Every monic polynomial is the characteristic polynomial of some operator.

{: .prompt-info }
> Suppose $ V$ is finite-dimensional and $ T \in \mathcal{L}(V) $. Let $ \mathcal{E} $ be the subspace of $ \mathcal{L}(V) $ defined by
>
> $$ \mathcal{E} = \{ q(T) : q \in \mathcal{P}(\mathbf{F}) \} $$.
>
> Then the list $I, T, T^2, \dots, T^{m-1}$ is a basis of $\mathcal{E}$, where $m = \deg p_T$.

{: .prompt-proof }
> $\mathcal{E}$ is indeed a subspace: it's closed under addition and scalar multiplication because $q_1(T) + q_2(T) = (q_1 + q_2)(T)$ and $c\,q(T) = (cq)(T)$. In fact $\mathcal{E}$ is the image of the linear map $\mathcal{P}(\mathbf{F}) \to \mathcal{L}(V)$, $q \mapsto q(T)$.
>
> Consider the linear map $\Phi : \mathcal{P}(\mathbf{F}) \to \mathcal{L}(V)$, $\Phi(q) = q(T)$. Then $\operatorname{range}\Phi = \mathcal{E}$ and $\operatorname{null}\Phi = \{q : q(T) = 0\}$ — exactly the multiples of $p$. So $\mathcal{E} \cong \mathcal{P}(\mathbf{F})/\langle p\rangle$, and the quotient has the remainders of degree $< m$ as canonical representatives — $m$ dimensions' worth.

## Local Minimal Polynomial

{: .prompt-info }
> Suppose $V$ is finite-dimensional, $ T \in \mathcal{L}(V) $, and $ v \in V $. Then
>
> $$\operatorname{span}(v, Tv, \dots, T^m v) = \operatorname{span}(v, Tv, \dots, T^{\dim V - 1}v)$$
>
> for all integers $ m \ge \dim V - 1 $.

{: .prompt-proof }
> Write $n = \dim V$ and
>
> $$U_m := \operatorname{span}(v, Tv, \dots, T^m v).$$
>
> Since each list extends the previous one, the chain is increasing:
>
> $$U_0 \subseteq U_1 \subseteq U_2 \subseteq \dots$$
>
> The goal is: $U_m = U_{n-1}$ for all $m \geq n-1$.
>
> **Claim 1 (stabilizing chain).** If $U_k = U_{k-1}$ for some $k \geq 1$, then $U_m = U_{k-1}$ for all $m \geq k-1$.
>
> *Proof.* It suffices to show $U_{k+1} = U_k$, then induct.
>
> $U_k = U_{k-1}$ says $T^k v \in U_{k-1} = \operatorname{span}(v, Tv, \dots, T^{k-1}v)$, so write
>
> $$T^k v = a_0 v + a_1 Tv + \dots + a_{k-1}T^{k-1}v.$$
>
> Apply $T$ to both sides:
>
> $$T^{k+1}v = a_0 Tv + a_1 T^2 v + \dots + a_{k-1}T^{k}v \in U_k.$$
>
> So the one new vector in the list for $U_{k+1}$ already lies in $U_k$, giving $U_{k+1} \subseteq U_k$, hence $U_{k+1} = U_k$. Induction extends this to all $m \geq k$. $\square$
>
> **Claim 2.** There exists $k$ with $1 \leq k \leq n$ and $U_k = U_{k-1}$. (Treat the case $v = 0$ separately: then every $U_m = \{0\}$ and the exercise is trivial. So assume $v \neq 0$.)
>
> *Proof.* The list $v, Tv, \dots, T^n v$ has $n+1$ vectors in the $n$-dimensional space $V$, so it is **linearly dependent**. Therefore, some vector in the list lies in the span of the ones preceding it: there is $k$ with $0 \le k \leq n$ and
>
> $$T^k v \in \operatorname{span}(v, Tv, \dots, T^{k-1}v) = U_{k-1}.$$
>
> Since $v \ne 0$, we have $k \geq 1$. And $T^kv \in U_{k-1}$ gives $U_k \subseteq U_{k-1}$, i.e. $U_k = U_{k-1}$. $\square$
>
> Take the $k \leq n$ from Claim 2. By Claim 1, $U_m = U_{k-1}$ for **all** $m \geq k-1$. Since $k - 1 \leq n-1$, the index $n-1$ is itself in that range, so $U_{n-1} = U_{k-1}$. Therefore, for every $m \geq n-1 \; (\geq k-1)$,
>
> $$U_m = U_{k-1} = U_{n-1},$$
>
> which is exactly
>
> $$\operatorname{span}(v, Tv, \dots, T^m v) = \operatorname{span}(v, Tv, \dots, T^{\dim V - 1}v). \qquad \blacksquare$$

{: .prompt-tip }
> **$U_{n-1}$ is $T$-invariant.** Since $T(U_{n-1}) \subseteq U_n = U_{n-1}$. In fact $U_{n-1}$ is the *smallest* $T$-invariant subspace containing $v$ — any such subspace must contain all $T^jv$. So, closing $v$ up under $T$ never requires more than $\dim V$ terms.

{: .prompt-info }
> *Local minimal polynomial*
>
> Suppose $ V $ is finite-dimensional, $ T \in \mathcal{L}(V) $, and $ v \in V $. Then there exists a unique monic polynomial $ p_v $ of smallest degree such that $ p_v(T)v = 0 $.
>
> $ p_v $ is the minimal polynomial of the smallets $T$-invariant subspace $U$ containing $v$, and $ \deg p_v = k = \dim U $.

{: .prompt-proof }
> In Claim 2, $T^kv = \sum_{j<k} a_j T^j v$, rearranges to $q(T)v = 0$ with $q(z) = z^k - a_{k-1}z^{k-1} - \dots - a_0$ monic of degree $k$ — and minimality of $k$ makes $q$ the least-degree monic polynomial with $q(T)v = 0$.

{: .prompt-tip }
> Suppose $ V $ is finite-dimensional, $ T \in \mathcal{L}(V) $, $ q \in \mathcal{P}(\mathbf{F}) $ and $ q(T) = 0 $, then
>
> $$ p_v \mid p_T \mid q. $$

{: .prompt-tip }
> If $p_u$ and $p_w$ are coprime, then $p_{u+w} = p_u p_w$.

{: .prompt-proof }
> First, $(p_up_w)(T)(u+w) = p_w(T)p_u(T)u + p_u(T)p_w(T)w = 0$, so $p_{u+w} \mid p_up_w$.
>
> Conversely suppose $r(T)(u+w) = 0$. Applying $p_w(T)$ and using $p_w(T)w = 0$:
$$(p_w r)(T)u = p_w(T)r(T)u + p_w(T)r(T)w = p_w(T)r(T)(u+w) = 0,$$
so $p_u \mid p_w r$, and coprimality gives $p_u \mid r$. Symmetrically $p_w \mid r$, hence $p_up_w \mid r$. Taking $r = p_{u+w}$ finishes it. $\square$

{: .prompt-info }
> There exists $v \in V$ with $p_v = p$.

{: .prompt-proof }
> Factor $p = q_1^{m_1}\dots q_k^{m_k}$ into powers of distinct monic irreducibles. Fix $i$.
>
> Since $\deg(p/q_i) < \deg p$, minimality of $p$ gives $(p/q_i)(T) \neq 0$, so choose $u_i$ with $(p/q_i)(T)u_i \neq 0$. Set
$$w_i = \big(p/q_i^{m_i}\big)(T)\,u_i.$$
>
> Then $q_i^{m_i}(T)w_i = p(T)u_i = 0$, so $p_{w_i} \mid q_i^{m_i}$; and $q_i^{m_i-1}(T)w_i = (p/q_i)(T)u_i \neq 0$, so $p_{w_i} \nmid q_i^{m_i-1}$. As $q_i$ is irreducible, the only possibility is $p_{w_i} = q_i^{m_i}$.
>
> The polynomials $q_1^{m_1},\dots,q_k^{m_k}$ are pairwise coprime, so $v = w_1 + \dots + w_k$ gives
>
> $$p_v = p_{w_1}\dots p_{w_k} = q_1^{m_1}\dots q_k^{m_k} = p. \qquad \blacksquare$$

{: .prompt-tip }
> **Over $\mathbb{C}$ this is the Jordan statement in disguise.** There $q_i = z - \lambda_i$, and $\deg p = \sum m_i$ while $n = \sum \dim G(\lambda_i, T)$. Since always $m_i \le \dim G(\lambda_i,T)$, the hypothesis $\deg p = n$ forces $m_i = \dim G(\lambda_i,T)$ for every $i$ — that is, **one Jordan block per eigenvalue**. The $w_i$ above is precisely a vector at the *top* of the $i$-th block's chain, and $v$ is their sum.

{: .prompt-info }
> If $p_v = p$ and $\deg p = n$, then $v, Tv, \dots, T^{n-1}v$ is a basis.

{: .prompt-proof }
> Suppose $a_0 v + a_1 Tv + \dots + a_{n-1}T^{n-1}v = 0$ with the $a_j$ not all zero. Then $q(z) = a_0 + a_1 z + \dots + a_{n-1}z^{n-1}$ is a nonzero polynomial with $q(T)v = 0$, so $p_v \mid q$ — impossible, since $\deg q < n = \deg p_v$. So the list is linearly independent, and $n$ independent vectors in an $n$-dimensional space form a basis. $\square$

## Homomorphism gives divisibility; injectivity gives equality

| $T \in \mathcal{L}(V)$                      | $q \in \mathcal{P}(\mathbf{F})$                 | Type of $\Phi$                                                                                          | $\Phi$ injective? | Minimal polynomial                    |
| ------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------- |
| $T\vert_U$                                  | $q(T\vert_U) = q(T)\vert_U$                     | unital algebra hom. $\mathcal{L}(V) \to \mathcal{L}(U)$                                                 | no                | $p_{T\vert_U} \mid p_T$               |
| $T/U$                                       | $q(T/U) = q(T)/U$                               | unital algebra hom. $\mathcal{L}(V) \to \mathcal{L}(V/U)$                                               | no                | $p_{T/U} \mid p_T$                    |
|                                             |                                                 |                                                                                                         |                   | $p_T \mid p_{T\vert_U} \cdot p_{T/U}$ |
| $\mathbf{F} = \mathbb{R}$, $T_{\mathbb{C}}$ | $q(T_\mathbb{C}) = (q(T))_\mathbb{C}$, $q$ real | unital $\mathbb{R}$-algebra hom. $$\mathcal{L}_\mathbb{R}(V) \to \mathcal{L}_\mathbb{C}(V_\mathbb{C})$$ | yes               | $p_{T_\mathbb{C}} = p_T$              |
| $T'$                                        | $q(T') = (q(T))'$                               | unital **anti**-homomorphism $\mathcal{L}(V) \to \mathcal{L}(V')$                                       | yes               | $p_{T'} = p_T$                        |
| $S$ invertible, $S^{-1}TS$                  | $q(S^{-1}TS) = S^{-1}q(T)S$                     | unital algebra **iso** $\mathcal{L}(V) \to \mathcal{L}(V)$                                              | yes (bijective)   | $p_{S^{-1}TS} = p_T$                  |

{: .prompt-info }
> $\Phi$ is a unital [algebra homomorphism](https://en.wikipedia.org/wiki/Algebra_over_a_field#Algebra_homomorphisms) $\mathcal{L}(V) \to \mathcal{L}(W)$ for the relevant $W$ $\implies$ $q(\Phi(T)) = \Phi(q(T))$.

{: .prompt-tip }
> The map preserves powers by multiplicativity, scalars by linearity and unitality, and sums by linearity.

{: .prompt-tip }
> Homomorphism $\Rightarrow$ divisibility; injective on top of that $\Rightarrow$ equality.** Because $p_{\Phi(T)} \mid p_T$ is just "$q(T) = 0 \Rightarrow q(\Phi(T)) = 0$", and injectivity supplies the converse.

## $ST$ and $TS$: what transfers, and the cost of $z$

{: .prompt-info }
> *Commutation identity*
>
> For every polynomial $q$,
>
> $$T\, q(ST) \;=\; q(TS)\, T.$$

{: .prompt-proof }
> First for monomials: $T(ST)^k = (TS)^k T$, by induction on $k$. The case $k = 0$ is $T = T$. Assuming it for $k$,
>
> $$T(ST)^{k+1} = (TS)\,T\,(ST)^{k} = (TS)\,(TS)^k T = (TS)^{k+1}T,$$
>
> where the first step just regroups $T(ST)(ST)^k$. Both sides of the identity are linear in $q$, so it extends from monomials to all polynomials. $\blacksquare$

The therom below is about annihilating polynomials transfer, at the cost of one factor of $z$.

{: .prompt-info }
> If $q$ annihilates $ST$, then $z\,q(z)$ annihilates $TS$. Consequently
>
> $$p_{TS} \;\big|\; z\,p_{ST} \qquad\text{and}\qquad p_{ST} \;\big|\; z\,p_{TS}.$$

{: .prompt-proof }
> Suppose $q(ST) = 0$. By the identity, $q(TS)\,T = T\,q(ST) = 0$. Multiply on the right by $S$:
>
> $$q(TS)\,TS = 0.$$
>
> Since $q(TS)$ is a polynomial in $TS$, this says exactly that $z\,q(z)$ evaluated at $TS$ is $0$. Applying this to $q = p_{ST}$ gives $p_{TS} \mid z\,p_{ST}$, since the minimal polynomial divides every annihilating polynomial. Swapping the roles of $S$ and $T$ gives the other divisibility. $\blacksquare$

{: .prompt-warning }
> The factor of $z$ cannot be dropped: $ST$ and $TS$ need **not** have the same minimal polynomial. See the example below, where $p_{ST} = z$ and $p_{TS} = z^2$.

{: .prompt-tip }
> $ST$ and $TS$ have the same nonzero eigenvalues.

The following theorem is about nilpotency transfers.

{: .prompt-info }
> $ST$ is nilpotent $\iff$ $TS$ is nilpotent.
>
> Writing $i(\cdot)$ for the nilpotency index,
>
> $$\big|\, i(ST) - i(TS) \,\big| \;\le\; 1.$$

{: .prompt-proof }
> If $(ST)^k = 0$, $q = z^k$ annihilates $ST$: then $z^{k+1}$ annihilates $TS$, i.e. $(TS)^{k+1} = 0$. So $i(TS) \le i(ST) + 1$, and symmetrically $i(ST) \le i(TS) + 1$. $\blacksquare$

{: .prompt-tip }
> Unwound, the computation is one line:
>
> $$(TS)^{k+1} = \big((TS)^k T\big) S = \big(T (ST)^k\big) S = T \cdot 0 \cdot S = 0.$$

{: .prompt-warning }
> The bound is attained, so the indices genuinely need not be equal. Take
>
> $$S = \begin{pmatrix} 0 & 0 \\ 0 & 1\end{pmatrix}, \qquad
>   T = \begin{pmatrix} 0 & 1 \\ 0 & 0\end{pmatrix}.$$
>
> Then $ST = 0$ with $i(ST) = 1$, while $$TS = \begin{pmatrix} 0 & 1 \\ 0 & 0\end{pmatrix} \ne 0$$ with $i(TS) = 2$.
