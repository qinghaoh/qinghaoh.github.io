---
title:  "Quotient Spaces"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Notation

{% include notation-table.md keys="V LV quotient_space quotient_op" %}

## Translate

{: .prompt-info }
> *Re-basing*
>
> Suppose $ U $ is a subspace of $ V $ and $ v,w \in V $. Then
>
> $$ x \in v + U \Leftrightarrow x + U = v + U. $$

{: .prompt-tip }
> Corollary: Two translates of a subspace are equal or disjoint.
>
> $$ v - w \in U \Leftrightarrow v + U = w + U \Leftrightarrow (v + U) \cap (w + U) \ne \emptyset $$
>
> $$ v \in U \Leftrightarrow v + U = 0 + U. $$

{: .prompt-tip }
>  Suppose $ A_1 = v + U_1 $ and $ A_2 = w + U_2 $ for some $ v,w \in V $ and some subspaces $ U_1,U_2 $ of $ V $.
>
> $$ \forall x \in A_1 \cap A_2, \ A_1 \cap A_2 = (x + U_1) \cap (x + U_2) = x + (U_1 \cap U_2). $$

### Bases

{: .prompt-info }
> Suppose $ U $ and $ W $ are subspaces of $ V $ and $ V = U \oplus W $. $\pi\|_W : W \to V/U$ is an **isomorphism**.

{: .prompt-info }
> *Duality (Project, function)*
>
> Suppose $ U $ and $ W $ are subspaces of $ V $ and $ V = U \oplus W $. Suppose $ w_1, \dots, w_m $ is a basis of $ W $. Then $ w_1 + U, \dots, w_m + U $ is a basis of $ V/U $.

{: .prompt-info }
> *Duality (Lift, one-to-many)*
>
> Suppose that $ U $ is a subspace of $ V $ such that $ V/U $ is finite-dimensional.
> There exists a finite-dimensional subspace $ W $ of $ V $ such that $ \dim W = \dim V/U $ and $ V = U \oplus W $.

{: .prompt-proof }
> Since $V/U$ is finite-dimensional, pick a basis
>
> $$v_1 + U,\ \dots,\ v_m + U \quad (m = \dim V/U),$$
>
> with chosen representatives $v_1, \dots, v_m \in V$. **Define**
>
> $$W := \operatorname{span}(v_1, \dots, v_m).$$
>
> We show this $W$ works: it's finite-dimensional, $\dim W = m = \dim V/U$, and $V = U \oplus W$.
>
> **The $v_k$ are linearly independent (so $\dim W = m$).** Suppose $\sum_k a_k v_k = 0$. Apply $\pi$:
>
> $$0 + U = \pi\Big(\sum_k a_k v_k\Big) = \sum_k a_k (v_k + U).$$
>
> Since $v_1 + U, \dots, v_m + U$ is a basis of $V/U$, it's independent, so all $a_k = 0$. Thus $v_1, \dots, v_m$ is independent and $\dim W = m = \dim V/U$.
>
> **$U + W = V$.** Let $v \in V$. Expand its coset in the basis: $v + U = \sum_k a_k (v_k + U) = \big(\sum_k a_k v_k\big) + U$. Equal cosets differ by an element of $U$:
>
> $$v - \sum_k a_k v_k \in U \quad\Longrightarrow\quad v = \underbrace{\Big(v - \sum_k a_k v_k\Big)}_{\in\, U} + \underbrace{\sum_k a_k v_k}_{\in\, W} \in U + W.$$
>
> **$U \cap W = \{0\}$.** Let $x \in U \cap W$. Since $x \in W$, write $x = \sum_k a_k v_k$. Since $x \in U$, $\pi(x) = 0$:
>
> $$0 + U = \pi(x) = \sum_k a_k (v_k + U).$$
>
> Independence of the basis forces all $a_k = 0$, so $x = 0$.
>
> Therefore $V = U \oplus W$ with $\dim W = \dim V/U$. $\blacksquare$

{: .prompt-info }
> Suppose $ U $ is a subspace of $ V $ and $ v_1 + U, \dots, v_m + U $ is a basis of $ V/U $ and $ u_1, \dots, u_n $ is a basis of $ U $. Then $ v_1, \dots, v_m, u_1, \dots, u_n $ is a basis of 𝑉.

## Quotient Operator

{: .prompt-info }
> Suppose $ V $ is finite-dimensional, $ T \in \mathcal{L}(V) $, and $ U $ is a subspace of $ V $ invariant under $ T $. The quotient operator $ T/U \in \mathcal{L}(V/U) $ is defined by
>
> $$ (T/U)(v + U) = Tv + U $$
>
> for each $ v \in V $.
