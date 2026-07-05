---
title:  "Dual Space"
category: math
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Dual Space and Dual Map

![dual map](/assets/img/math/dual_map.png){: w="600" h="300" }

* $ \operatorname{null} T' = (\operatorname{range} T)^0 $
* $ \operatorname{range} T' = (\operatorname{null} T)^0 $

* $ \dim \operatorname{null} T' = \dim \operatorname{null} T + \dim W - \dim V $
* $ \dim \operatorname{range} T' = \dim \operatorname{range} T $

## Annihilator

{: .prompt-tip }
> Denote the collection of all subspaces of $V$ by $\mathrm{Sub}(V)$.
>
> $(\mathrm{Sub}(V), \subseteq)$ is a *lattice*, with $(\wedge, \vee) = (\cap, +)$.
>
> The annihilator map $ U \mapsto U^0 $ is an order-reversing *bijection* between $ \mathrm{Sub}(V) $ and $ \mathrm{Sub}(V') $.

{: .prompt-info }
> Suppose $ V $ is finite-dimensional and $ U $ and $ W $ are subspaces of $ V $.
>
> * $ (U^0)^0 = U $
> * Antitone: $ W^0 \subseteq U^0 \Leftrightarrow U \subseteq W $

{: .prompt-proof }
> $W^0 \subseteq U^0 \implies U \subseteq W$
>
> Prove the contrapositive: if $U \not\subseteq W$, then $W^0 \not\subseteq U^0$.
>
> Suppose $U \not\subseteq W$, so there exists $u \in U$ with $u \notin W$. The goal is to produce a functional that lives in $W^0$ but not $U^0$ — one that kills all of $W$ yet doesn't kill $u$.
>
> Let $w_1, \dots, w_k$ be a basis of $W$. Since $u \notin W = \operatorname{span}(w_1,\dots,w_k)$, the list
>
> $$w_1, \dots, w_k,\, u$$
>
> is linearly independent. Using finite-dimensionality, extend it to a basis $w_1, \dots, w_k,\, u,\, v_1, \dots, v_j$ of $V$. Now define $\varphi \in V'$ on this basis by
>
> $$\varphi(w_i) = 0 \ (\text{all } i), \qquad \varphi(u) = 1, \qquad \varphi(v_\ell) = 0 \ (\text{all } \ell),$$
>
> extended linearly. Since $\varphi$ vanishes on a basis of $W$, it vanishes on all of $W$, so $\varphi \in W^0$. But $u \in U$ and $\varphi(u) = 1 \neq 0$, so $\varphi \notin U^0$. Therefore $W^0 \not\subseteq U^0$, completing the contrapositive. $\blacksquare$

{: .prompt-info }
> Lattice anti-homomorphism:
>
> * $ (U + W)^0 = U^0 \cap W^0 $
> * $ (U \cap W)^0 = U^0 + W^0 $

{: .prompt-proof }
> By the double-annihilator fact $(A^0)^0 = A$ (under $V \cong V''$). Apply (a) *inside $V'$* to the subspaces $U^0, W^0$:
>
> $$(U^0 + W^0)^0 = (U^0)^0 \cap (W^0)^0 = U \cap W.$$
>
> Now annihilate both sides and use $(A^0)^0 = A$ once more on the left:
>
> $$U^0 + W^0 = \big((U^0 + W^0)^0\big)^0 = (U \cap W)^0.$$

## Duality Swaps Spanning and Independence

Given $v_1,\dots,v_m \in V$, define

$$S: \mathbb{F}^m \to V, \qquad S(a_1,\dots,a_m) = a_1 v_1 + \cdots + a_m v_m \quad (\text{so } e_i \mapsto v_i).$$

Under this re-encoding:

{: .prompt-info }
> * $ S \text{ surjective} \iff (v_i) \text{ spans } V $
> * $ S \text{ injective} \iff (v_i) \text{ linearly independent} $

Both equivalences hold in **any** dimension — no finiteness needed.

The dual map $S': V' \to (\mathbb{F}^m)' \cong \mathbb{F}^m$ is given by $S'(\varphi) = \varphi \circ S$, so its $i$-th coordinate is $(\varphi \circ S)(e_i) = \varphi(S e_i) = \varphi(v_i)$. That is,

$$S'(\varphi) = \big(\varphi(v_1),\dots,\varphi(v_m)\big).$$

Now suppose $V$ is finite-dimensional, so the **dual-map theorem** applies: $T$ injective $\iff T'$ surjective, and $T$ surjective $\iff T'$ injective.

| map property    | reason              | list statement               |
| --------------- | ------------------- | ---------------------------- |
| $S$ surjective  | definition          | $(v_i)$ spans $V$            |
| $S$ injective   | definition          | $(v_i)$ linearly independent |
| $S'$ injective  | $\iff S$ surjective | $(v_i)$ spans $V$            |
| $S'$ surjective | $\iff S$ injective  | $(v_i)$ linearly independent |

{: .prompt-warning }
> Only the last two rows use finite-dimensionality (through the dual-map theorem). The first two are dimension-free.

### Dualizing: replace $V$ by $V'$

The identical construction, with the base space changed from $V$ to $V'$. The list $\varphi_1,\dots,\varphi_m$ now lives in $V'$, so its encoding map is

$$ R: \mathbb{F}^m \to V', \quad e_i \mapsto \varphi_i. $$

Its transpose is $R': V'' \to (\mathbb{F}^m)' \cong \mathbb{F}^m$. Under the canonical isomorphism $V \cong V''$, the element of $V''$ corresponding to $v \in V$ is evaluation-at-$v$, written $\hat v$ with $\hat v(\varphi) = \varphi(v)$; feeding that to $R'$ gives

$$ R'(\hat v) = (\varphi_1(v),\dots,\varphi_m(v)). $$

| map property    | reason              | list statement                     |
| --------------- | ------------------- | ---------------------------------- |
| $R$ surjective  | definition          | $(\varphi_i)$ spans $V'$           |
| $R$ injective   | definition          | $(\varphi_i)$ linearly independent |
| $R'$ injective  | $\iff R$ surjective | $(\varphi_i)$ spans $V'$           |
| $R'$ surjective | $\iff R$ injective  | $(\varphi_i)$ linearly independent |

### Back to the annihilator

The null space of $R'$ is the common null space of the functionals:

$$ \operatorname{null} R' = \{v : \varphi_i(v) = 0 \ \forall i\} = \operatorname{null}\varphi_1 \cap \cdots \cap \operatorname{null}\varphi_m. $$

By the dual-map identity $\operatorname{null} T' = (\operatorname{range} T)^0$ applied to $T = R$, and since $\operatorname{range} R = \operatorname{span}(\varphi_i)$,

$$\operatorname{null}(R') = (\operatorname{range} R)^0 = \big(\operatorname{span}(\varphi_i)\big)^0.$$

Annihilating both sides yields

$$\operatorname{span}(\varphi_i) = \big(\operatorname{null}\varphi_1 \cap \cdots \cap \operatorname{null}\varphi_m\big)^0.$$
