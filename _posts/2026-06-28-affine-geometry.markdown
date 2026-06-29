---
title:  "Affine Geometry"
category: math
tags: math
mathjax_font: mathjax-pagella
mermaid: true
---

## Affine subspace

{: .prompt-tip }
> *affine* = linear structure with the origin forgotten
>
> An operation on points commutes with every translation $ x \mapsto x + t $ if and only if its coefficients sum to 1.

{: .prompt-proof }
> Take a combination $\sum_i \lambda_i v_i$ and translate every input point by a fixed $t$ (i.e. move the origin by $-t$). The output translates by:
>
> $$\sum_i \lambda_i (v_i + t) = \sum_i \lambda_i v_i + \Big(\sum_i \lambda_i\Big) t.$$
>
> For the whole construction to **shift by exactly $t$** — so that the pattern of points looks identical regardless of where the origin sits — we need the extra term to be $\big(\sum_i \lambda_i\big)t = t$ for all $t$, i.e.
>
> $$\sum_i \lambda_i = 1.$$

{: .prompt-tip }
> *affine subspace* (closed under lines):
>
> For any two points of $ A $, the entire line through them lies in $ A $.
>
> $ \lambda v + (1 - \lambda)w \in A \forall v,w \in A $ and $ \forall \lambda \in \mathbf{F} $

{: .prompt-info }
> *Affine subspaces are exactly translates of linear subspaces*

{: .prompt-proof }
> ($\Rightarrow$) Suppose $A = x + U$ for a subspace $U$. For $v, w \in A$, write $v = x + u_1$, $w = x + u_2$ with $u_1, u_2 \in U$. Then for any $\lambda \in \mathbf{F}$,
>
> $$\lambda v + (1-\lambda)w = x + \big(\lambda u_1 + (1-\lambda)u_2\big) \in x + U = A,$$
>
> since $\lambda u_1 + (1-\lambda)u_2 \in U$.
>
> ($\Leftarrow$) Suppose $A \neq \emptyset$ satisfies $\lambda v + (1-\lambda)w \in A$ for all $v, w \in A$, $\lambda \in \mathbf{F}$. Fix $p \in A$ and define $U := A - p$. We show $U$ is a subspace; then $A = p + U$.
>
> *Zero:* $0 = p - p \in U$.
>
> *Scalar multiplication:* Let $a - p \in U$ (with $a \in A$) and $\mu \in \mathbf{F}$. Then
>
> $$\mu(a-p) + p = \mu a + (1-\mu)p \in A,$$
>
> by the hypothesis applied to $a, p$ with $\lambda = \mu$. Hence $\mu(a - p) \in A - p = U$.
>
> *Addition:* Let $a_1 - p,\ a_2 - p \in U$. By the hypothesis with $\lambda = \tfrac12$, the point $c := \tfrac12 a_1 + \tfrac12 a_2 \in A$. By scalar closure just proved, $2(c - p) + p = 2c - p \in A$. Since $2c - p = a_1 + a_2 - p$,
>
> $$(a_1 - p) + (a_2 - p) = (a_1 + a_2 - p) - p \in A - p = U.$$
>
> So $U$ is closed under addition and scalar multiplication and contains $0$: it is a subspace, and $A = p + U$. $\blacksquare$

{: .prompt-info }
> *An affine subspace is closed under every finite affine combination* ("Closed under lines" is the $ k = 2 $ case).
>
> If $ a_1, \dots, a_k \in A $ and $ \sum \lambda_i = 1 $, then $ \sum \lambda_i a_i \in A $.

{: .prompt-proof }
> *Proof by induction on $k$.*
>
> For $ k = 1 $, $ \lambda_1 = 1 $ gives $ a_1 \in A $. For the step, given $ \sum_{i=1}^{k}\lambda_i = 1 $ with $ k \ge 2 $, at least one coefficient — say $ \lambda_k $ — satisfies $ \lambda_k \neq 1 $. Set $ s = \lambda_1 + \dots + \lambda_{k-1} = 1 - \lambda_k \neq 0 $. Then
>
> $$\sum_{i=1}^k \lambda_i a_i = s\underbrace{\left(\sum_{i=1}^{k-1}\tfrac{\lambda_i}{s} a_i\right)}_{=:b} + \lambda_k a_k.$$
>
> The inner combination $c$ has coefficients $ \tfrac{\lambda_i}{s} $ summing to $ \tfrac{s}{s} = 1 $, so by induction $ b \in A $. Then $ s\,b + \lambda_k a_k $ is an affine combination of the two points $ b, a_k \in A $ (coefficients $ s + \lambda_k = 1 $), so it lies in $ A $ by the line condition. $\blacksquare$.

## Affine hull

{: .prompt-tip }
> *affine null* (closed under *affine combinations*):
>
> $ A = \\{\lambda_1 v_1 + \dots + \lambda_m v_m : v_1, \dots, v_m \in V, \lambda_1, \dots, \lambda_m \in \mathbf{F} \ \text{and} \ \lambda_1 + \dots + \lambda_m = 1 \\} $

{: .prompt-info }
> An affine null is an affine subspace.

{: .prompt-proof }
> Take two elements of $A$:
>
> $$x = \sum_{i=1}^m \alpha_i v_i \ \Big(\textstyle\sum_i \alpha_i = 1\Big), \qquad y = \sum_{i=1}^m \beta_i v_i \ \Big(\textstyle\sum_i \beta_i = 1\Big),$$
>
> and any $\lambda \in \mathbf{F}$. Form the line combination and group by each $v_i$:
>
> $$\lambda x + (1-\lambda)y = \sum_{i=1}^m \big(\lambda \alpha_i + (1-\lambda)\beta_i\big) v_i.$$
>
> This is a linear combination of $v_1, \dots, v_m$, so it is a candidate member of $A$; we just need its coefficients to sum to $1$. Sum them:
>
> $$\sum_{i=1}^m \big(\lambda \alpha_i + (1-\lambda)\beta_i\big) = \lambda \underbrace{\sum_i \alpha_i}_{=1} + (1-\lambda)\underbrace{\sum_i \beta_i}_{=1} = \lambda + (1-\lambda) = 1.$$
>
> The coefficients sum to $1$, so $\lambda x + (1-\lambda)y \in A$. Thus $A$ satisfies the definition of an affine subspace. $\blacksquare$

Since an affine subspace is equivalent to a translate of a subspace, another proof is to prove that an affine hull is a translate of a subspace:

{: .prompt-proof }
> If $A$ is empty there's nothing to translate, but $A \neq \emptyset$: taking $\lambda_1 = 1$ and the rest $0$ gives $v_1 \in A$. Fix the basepoint $v_1$ and define
>
$$U := \operatorname{span}(v_2 - v_1,\ v_3 - v_1,\ \dots,\ v_m - v_1).$$
>
> We claim $A = v_1 + U$.
>
> **$A \subseteq v_1 + U$.** Take $\lambda_1 v_1 + \cdots + \lambda_m v_m \in A$ with $\sum_i \lambda_i = 1$. Use $\lambda_1 = 1 - (\lambda_2 + \cdots + \lambda_m)$ to eliminate $\lambda_1$:
>
> $$\sum_{i=1}^m \lambda_i v_i = v_1 + \sum_{i=2}^m \lambda_i (v_i - v_1).$$
>
> The sum $ \sum_{i=2}^m \lambda_i (v_i - v_1)$ lies in $U$, so the point is in $v_1 + U$.
>
> **$v_1 + U \subseteq A$.** A general element is $v_1 + \sum_{i=2}^m c_i(v_i - v_1)$. Reversing the computation, this equals $\sum_i \lambda_i v_i$ with $\lambda_i = c_i$ for $i \ge 2$ and $\lambda_1 = 1 - \sum_{i\ge2} c_i$, whose coefficients sum to $1$. So it's in $A$.
>
> Hence $A = v_1 + U$ is a translate of the subspace $U$. $\blacksquare$

{: .prompt-tip }
> The affine hull $ A = \\{\lambda_1 v_1 + \dots + \lambda_m v_m : v_1, \dots, v_m \in V, \lambda_1, \dots, \lambda_m \in \mathbf{F} \ \text{and} \ \lambda_1 + \dots + \lambda_m = 1 \\} $ is a translate of a subspace of dimension less than $m$.

{: .prompt-info }
> The affine hull $ A = \\{\lambda_1 v_1 + \dots + \lambda_m v_m : v_1, \dots, v_m \in V, \lambda_1, \dots, \lambda_m \in \mathbf{F} \ \text{and} \ \lambda_1 + \dots + \lambda_m = 1 \\} $ is the smallest affine subspace containing the the specific finite set $ v_1, \dots, v_m $.

{: .prompt-proof }
> This is a direct conclusion from the closure property of an affine subspace.

### Hull

A "hull" is always a **closure operation toward a property**: given a set $S$ and a class of "nice" sets (subspaces, affine subspaces, convex sets...), the corresponding hull is

$$\text{hull}(S) = \text{the smallest nice set containing } S = \bigcap \{\,N : N \text{ is nice and } S \subseteq N\,\}.$$

It's the tightest-fitting enclosure of $S$ from the chosen family:

- **It contains $S$** (it encloses).
- **It is itself nice** (the shell is made of the right material).
- **It is the smallest such** (the shell is tight, no slack).

| Hull                              | Family it's smallest within | Defining combinations | Coefficient constraint                         |
| --------------------------------- | --------------------------- | --------------------- | ---------------------------------------------- |
| **Linear span** (= "linear hull") | subspaces                   | $\sum \lambda_i v_i$  | none                                           |
| **Affine hull**                   | affine subspaces            | $\sum \lambda_i v_i$  | $\sum \lambda_i = 1$                           |
| **Convex hull**                   | convex sets                 | $\sum \lambda_i v_i$  | $\sum \lambda_i = 1$ **and** $\lambda_i \ge 0$ |
| **Conical hull**                  | convex cones                | $\sum \lambda_i v_i$  | $\lambda_i \ge 0$                              |

The unifying idea is that **"hull" names a closure operator**. Abstractly, any map $S \mapsto \overline{S}$ that is

- *extensive* ($S \subseteq \overline{S}$ — it encloses),
- *monotone* ($S \subseteq T \Rightarrow \overline{S} \subseteq \overline{T}$), and
- *idempotent* ($\overline{\overline{S}} = \overline{S}$ — re-enclosing an already-enclosed set adds nothing),
