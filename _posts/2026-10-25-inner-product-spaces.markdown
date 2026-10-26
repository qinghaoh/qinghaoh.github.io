---
title:  "Inner Product Spaces"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
---

Notes on inner products: the definition over $\mathbb{R}$ and $\mathbb{C}$, the parallelogram equality and which norms come from an inner product, and two applications of the Cauchy–Schwarz inequality: the classical mean inequalities and a bound on eigenvalues.

## Notation

{% include notation-table.md keys="F V n LV MT" %}

## Inner Products

An **inner product** on $V$ is

- over $\mathbb{R}$: a positive-definite symmetric bilinear form;
- over $\mathbb{C}$: a positive-definite Hermitian form.

It defines the norm $\lVert v \rVert = \sqrt{\langle v, v \rangle}$.

### The Parallelogram Equality

{: .prompt-info }
> In an inner product space, for all $u, v \in V$,
>
> $$\lVert u + v \rVert^2 + \lVert u - v \rVert^2 = 2\big(\lVert u \rVert^2 + \lVert v \rVert^2\big).$$

{: .prompt-proof }
> Expanding gives $\lVert u \pm v \rVert^2 = \lVert u \rVert^2 \pm 2\operatorname{Re}\langle u, v \rangle + \lVert v \rVert^2$. Adding the two cancels the cross terms. $\blacksquare$

{: .prompt-info }
> [*Polarization identities.*](https://en.wikipedia.org/wiki/Polarization_identity) In an inner product space, the norm determines the inner product: for all $u, v \in V$,
>
> $$\langle u, v \rangle = \frac{\lVert u + v \rVert^2 - \lVert u - v \rVert^2}{4} \quad \text{over } \mathbb{R},$$
>
> $$\langle u, v \rangle = \frac{\lVert u + v \rVert^2 - \lVert u - v \rVert^2 + \lVert u + iv \rVert^2\, i - \lVert u - iv \rVert^2\, i}{4} \quad \text{over } \mathbb{C}.$$

{: .prompt-proof }
> Subtracting the two expansions in the previous proof gives $\lVert u + v \rVert^2 - \lVert u - v \rVert^2 = 4\operatorname{Re}\langle u, v \rangle$, which is the real case. Over $\mathbb{C}$, replacing $v$ by $iv$ and using $\operatorname{Re}\langle u, iv \rangle = \operatorname{Re}\big(\overline{i}\langle u, v \rangle\big) = \operatorname{Im}\langle u, v \rangle$ gives $\lVert u + iv \rVert^2 - \lVert u - iv \rVert^2 = 4\operatorname{Im}\langle u, v \rangle$. So the right side is $\operatorname{Re}\langle u, v \rangle + i\operatorname{Im}\langle u, v \rangle$. $\blacksquare$

A **norm** on $V$ is a function $\lVert \cdot \rVert : V \to [0, \infty)$ such that

- $\lVert v \rVert = 0$ if and only if $v = 0$;
- $\lVert \alpha v \rVert = \lvert \alpha \rvert\,\lVert v \rVert$ for all $\alpha \in \mathbf{F}$ and all $v \in V$;
- $\lVert u + v \rVert \le \lVert u \rVert + \lVert v \rVert$ for all $u, v \in V$.

A norm **comes from an inner product** if there is an inner product $\langle \cdot, \cdot \rangle$ on $V$ such that $\lVert v \rVert = \langle v, v \rangle^{1/2}$ for all $v \in V$.

{: .prompt-info }
> *Jordan–von Neumann theorem.* A norm on $V$ comes from an inner product if and only if it satisfies the parallelogram equality. The inner product is then given by the polarization identities.

{: .prompt-proof }
> *($\Rightarrow$)* The two boxes above.
>
> *($\Leftarrow$)* Define $\langle \cdot, \cdot \rangle$ by the polarization identity and check that it is an inner product. Write $R(u, v) = \frac14\big(\lVert u + v \rVert^2 - \lVert u - v \rVert^2\big)$.
>
> *Real case.* $R$ is symmetric, $R(-u, v) = -R(u, v)$, and $R(v, v) = \lVert v \rVert^2$. Apply the parallelogram equality to $x + v, y + v$ and to $x - v, y - v$, and subtract:
>
> $$R(x + y, 2v) = 2R(x, v) + 2R(y, v).$$
>
> With $y = 0$ this gives $R(x, 2v) = 2R(x, v)$, so the left side is $2R(x + y, v)$, and $R$ is additive in the first slot. Additivity gives $R(qu, v) = qR(u, v)$ for rational $q$, and continuity of the norm ($\big\lvert \lVert a \rVert - \lVert b \rVert \big\rvert \le \lVert a - b \rVert$) extends this to real $q$.
>
> *Complex case.* $R$ is a real inner product on $V$ viewed as a real vector space, and $\langle u, v \rangle = R(u, v) + iR(u, iv)$. Since $\lVert iw \rVert = \lVert w \rVert$, $R(iu, iv) = R(u, v)$. Hence $R(iu, v) = R(-u, iv) = -R(u, iv)$, and $R(v, iv) = R(iv, -v) = -R(v, iv)$, so $R(v, iv) = 0$. These give $\langle iu, v \rangle = i\langle u, v \rangle$, $\langle v, u \rangle = \overline{\langle u, v \rangle}$, and $\langle v, v \rangle = \lVert v \rVert^2$. $\blacksquare$

### Which $p$-Norms Come from an Inner Product

{: .prompt-info }
> Suppose $p > 0$ and $n \ge 2$. There is an inner product on $\mathbf{F}^n$ whose norm is
>
> $$\lVert (x_1, \dots, x_n) \rVert = \big(\lvert x_1 \rvert^p + \cdots + \lvert x_n \rvert^p\big)^{1/p}$$
>
> if and only if $p = 2$.

{: .prompt-proof }
> If $p = 2$, this is the norm of the Euclidean inner product. Conversely, suppose an inner product has this norm. Apply the parallelogram equality to the first two standard basis vectors $u = e_1 = (1, 0, 0, \dots, 0)$ and $v = e_2 = (0, 1, 0, \dots, 0)$, which exist because $n \ge 2$. Then $u \pm v = (1, \pm 1, 0, \dots, 0)$, so
>
> $$\lVert u \rVert = \lVert v \rVert = 1, \qquad \lVert u \pm v \rVert = 2^{1/p}.$$
>
> The parallelogram equality gives $2 \cdot 2^{2/p} = 4$, so $2^{2/p} = 2$ and $p = 2$. $\blacksquare$

{: .prompt-tip }
> - **The case $n = 1$.** The formula becomes $\lvert x \rvert$ for every $p$, which is the norm of the inner product $\langle x, y \rangle = x\overline{y}$. So every $p$ works, and the hypothesis $n \ge 2$ is needed: the proof uses two independent directions to form a parallelogram.
> - **Infinite dimensions.** The same computation shows that among the sequence spaces $\ell^p$ ($p \ge 1$), only $\ell^2$ has a norm that comes from an inner product: use the sequences $(1, 0, 0, \dots)$ and $(0, 1, 0, \dots)$. The same holds for $L^p[0,1]$: use the indicator functions of $[0, \tfrac12)$ and $[\tfrac12, 1]$.

## The Cauchy–Schwarz Inequality

{: .prompt-info }
> For all $u, v$ in an inner product space, $\lvert \langle u, v \rangle \rvert \le \lVert u \rVert\,\lVert v \rVert$, with equality if and only if one of $u, v$ is a scalar multiple of the other.

### Mean Inequalities

{: .prompt-info }
> For positive real numbers $a_1, \dots, a_n$:
>
> $$
> \frac{n}{\frac{1}{a_1} + \cdots + \frac{1}{a_n}}
> \;\le\;
> \sqrt[n]{a_1 a_2 \cdots a_n}
> \;\le\;
> \frac{a_1 + \cdots + a_n}{n}
> \;\le\;
> \sqrt{\frac{a_1^2 + \cdots + a_n^2}{n}}
> $$
>
> From left to right, these are HM, GM, AM, and QM. Every inequality in the chain is an equality if and only if $a_1 = a_2 = \cdots = a_n$.

{: .prompt-tip }
> The last inequality, AM $\le$ QM, is Cauchy–Schwarz in $\mathbb{R}^n$ applied to $(a_1, \dots, a_n)$ and $(1, \dots, 1)$: $\sum_i a_i \le \sqrt{n}\,\big(\sum_i a_i^2\big)^{1/2}$. The equality case is $(a_1, \dots, a_n)$ being a multiple of $(1, \dots, 1)$.

### Eigenvalues and the Euclidean Norm

The [size of eigenvalues]({% post_url 2026-07-28-eigenvalues-and-eigenvectors %}#size-of-eigenvalues) bound uses the max-norm on coordinates. Using the Euclidean norm on $\mathbf{F}^n$ instead gives a bound in terms of all the entries at once.

{: .prompt-info }
> Suppose $v_1, \dots, v_n$ is a basis of $V$ and $T \in \mathcal{L}(V)$. If $\lambda$ is an eigenvalue of $T$, then
>
> $$\lvert \lambda \rvert^2 \le \sum_{j=1}^n \sum_{k=1}^n \lvert \mathcal{M}(T)_{j,k} \rvert^2,$$
>
> where $\mathcal{M}(T)$ is the matrix of $T$ with respect to $v_1, \dots, v_n$.

{: .prompt-proof }
> Let $A = \mathcal{M}(T)$, let $v$ be an eigenvector for $\lambda$, and let $x = \mathcal{M}(v) \in \mathbf{F}^n$ be its coordinates, so $x \ne 0$ and $Ax = \lambda x$. Use the Euclidean norm $\lVert x \rVert^2 = \sum_k \lvert x_k \rvert^2$ on $\mathbf{F}^n$. No inner product on $V$ is involved. By the Cauchy–Schwarz inequality on each row,
>
> $$\lvert\lambda\rvert^2 \lVert x \rVert^2 = \lVert Ax \rVert^2 = \sum_j \Big\lvert \sum_k A_{j,k} x_k \Big\rvert^2 \le \sum_j \Big(\sum_k \lvert A_{j,k} \rvert^2\Big) \lVert x \rVert^2.$$
>
> Dividing by $\lVert x \rVert^2$ gives the bound. $\blacksquare$

{: .prompt-tip }
> The right side is the squared Frobenius norm of $A$, which is at least the squared operator norm induced by the Euclidean norm. So this is another case of "$\lvert\lambda\rvert \le \lVert T \rVert$ for every induced norm", made weaker, by up to a factor of $\sqrt{n}$ (for $A = I$, the Frobenius norm is $\sqrt{n}$ and the operator norm is $1$).
>
> - **Comparison with row sums.** Since $\sum_{j,k}\lvert A_{j,k}\rvert^2 \le n^2 \max_{j,k}\lvert A_{j,k}\rvert^2$, this bound also gives $\lvert\lambda\rvert \le n\max_{j,k}\lvert A_{j,k}\rvert$. Neither it nor the max row sum bound always wins: the identity gives $1$ for row sums vs. $\sqrt{n}$ here, and a matrix whose only nonzero row is all ones gives $n$ vs. $\sqrt{n}$.
