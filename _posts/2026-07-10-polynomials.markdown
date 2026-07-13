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
