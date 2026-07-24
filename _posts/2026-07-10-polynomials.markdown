---
title:  "Polynomials"
category: math
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Polynomials

{: .prompt-info }
> _Bézout identity_
>
> Suppose $ p, q \in \mathcal{P}(\mathbf{C}) $ are nonconstant polynomials with no zeros in common. Let $ m = \deg p $ and $ n = \deg q $. There exist $ r \in \mathcal{P}_{n - 1}(\mathbf{C}) $ and $ s \in \mathcal{P}_{m - 1}(\mathbf{C}) $ such that
>
> $$ rp + sq = 1.$$

{: .prompt-proof }
> Define $ T: \mathcal{P}_{n - 1}(\mathbf{C}) \times \mathcal{P}_{m - 1}(\mathbf{C}) \to \mathcal{P}_{m + n - 1}(\mathbf{C}) $ by
>
> $$ T(r,s) = rp + sq $$.
>
> $T$ is a *square* map:
>
> $$
> \dim\big(\mathcal{P}_{n-1}(\mathbf{C}) \times \mathcal{P}_{m-1}(\mathbf{C})\big) = n + m,
> $$
>
> $$
> \dim \mathcal{P}_{m+n-1}(\mathbf{C}) = m+n.
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
> We want to force $r = 0$ and $s = 0$. The **no-common-zeros** hypothesis enters here, through unique factorization / the divisibility structure of $\mathbf{C}[z]$.
>
> From $(\ast)$, $q$ divides $rp$. Now list the zeros of $q$: by the FTA, $q$ factors as $q(z) = c\prod_{i}(z - \mu_i)$ over its roots $\mu_i$ (with multiplicity). Each root $\mu_i$ of $q$ is a zero of the left side $rp$, hence a zero of $r$ or of $p$. But $p$ and $q$ share **no** zeros, so $\mu_i$ is *not* a zero of $p$ — therefore $\mu_i$ must be a zero of $r$, and by matching multiplicities (a root of $q$ of multiplicity $t$ is not absorbed by $p$ at all, so all $t$ copies must come from $r$), the **full factor** $q$ divides $r$:
>
> $$
> q \mid r.
> $$
>
> But now degrees: $r \in \mathcal{P}_{n-1}(\mathbf{C})$ so $\deg r \le n - 1 < n = \deg q$. The only multiple of $q$ with degree below $\deg q$ is the zero polynomial. Hence
>
> $$
> r = 0.
> $$
>
>Plugging back into $(\ast)$: $sq = 0$ with $q \neq 0$, so $s = 0$. Thus $\ker T = \{0\}$ and $T$ is injective.
>
> It's easy to show $T$ is invertible since $T$ is an operator.
>
> The constant polynomial $1$ lives in the codomain $\mathcal{P}_{m+n-1}(\mathbf{C})$ (its degree $0$ is $\le m + n - 1$, using that $p, q$ nonconstant gives $m, n \ge 1$, so $m + n - 1 \ge 1 \ge 0$). By surjectivity from (b), $1$ is hit: there exist $r \in \mathcal{P}_{n-1}(\mathbf{C})$ and $s \in \mathcal{P}_{m-1}(\mathbf{C})$ with
>
> $$
> T(r,s) = rp + sq = 1. \qquad\blacksquare
> $$

{: .prompt-tip }
> "no common zeros" is the $\mathbf{C}[z]$-analogue of "coprime," and $rp + sq = 1$ is exactly the statement that the gcd is a unit.

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
> The minimal polynomial of $ T $ is a polynomial multiple of the minimal polynomial of $ \left. T \right\rvert u $.

{: .prompt-info }
> Every monic polynomial is the minimal polynomial of some operator.

{: .prompt-tip }
> See
> * https://en.wikipedia.org/wiki/Companion_matrix
> * https://mathworld.wolfram.com/CompanionMatrix.html

{: .prompt-info }
> For every polynomial $q$,
>
> $$T\, q(ST) \;=\; q(TS)\, T.$$

{: .prompt-proof }
> First for monomials: $T(ST)^k = (TS)^k T$, by induction on $k$. The case $k=0$ is $T = T$. Assuming it for $k$,
>
> $$T(ST)^{k+1} = \big(T S\big) T (ST)^{k} = (TS)\,(TS)^k T = (TS)^{k+1}T,$$
>
> where the first step just regroups $T(ST)(ST)^k$. Both sides of the identity are linear in $q$, so it extends from monomials to all polynomials. $\square$

{: .prompt-info }
> If $q$ annihilates $ST$, then $z\,q(z)$ annihilates $TS$.

{: .prompt-proof }
> Suppose $q(ST) = 0$. By the identity, $q(TS)\,T = T\,q(ST) = 0$. Multiply on the right by $S$:
>
> $$q(TS)\,TS = 0.$$
>
> Since $q(TS)$ is a polynomial in $TS$, it commutes with $TS$, so this says exactly that the polynomial $z\,q(z)$ evaluated at $TS$ is zero. $\square$

{: .prompt-info }
> Suppose $V$ is finite-dimensional, $ T \in \mathcal{L}(V) $, and $ v \in V $. Then
>
> $$\operatorname{span}(v, Tv, \dots, T^m v) = \operatorname{span}(v, Tv, \dots, T^{\dim V - 1}v). \qquad \blacksquare$$
>
> for all integers $ m \ge \dim V - 1 $.

{: .prompt-proof }
> Write $n = \dim V$ and
>
> $$U_m := \operatorname{span}(v, Tv, \dots, T^m v).$$
>
> Since each list extends the previous one, the chain is increasing:
>
> $$U_0 \subseteq U_1 \subseteq U_2 \subseteq \cdots$$
>
> The goal is: $U_m = U_{n-1}$ for all $m \geq n-1$.
>
> **Claim 1 (stabilizing chain).** If $U_k = U_{k-1}$ for some $k \geq 1$, then $U_m = U_{k-1}$ for all $m \geq k-1$.
>
> *Proof.* It suffices to show $U_{k+1} = U_k$, then induct.
>
> $U_k = U_{k-1}$ says $T^k v \in U_{k-1} = \operatorname{span}(v, Tv, \dots, T^{k-1}v)$, so write
>
> $$T^k v = a_0 v + a_1 Tv + \cdots + a_{k-1}T^{k-1}v.$$
>
> Apply $T$ to both sides:
>
> $$T^{k+1}v = a_0 Tv + a_1 T^2 v + \cdots + a_{k-1}T^{k}v \in U_k.$$
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
>
> **$k$ is the degree of the minimal polynomial of $T$ relative to $v$.** The dependence found in Claim 2, $T^kv = \sum_{j<k} a_j T^j v$, rearranges to $q(T)v = 0$ with $q(z) = z^k - a_{k-1}z^{k-1} - \cdots - a_0$ monic of degree $k$ — and minimality of $k$ makes $q$ the least-degree monic polynomial with $q(T)v = 0$.

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
