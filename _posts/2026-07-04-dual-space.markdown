---
title:  "Dual Space"
category: math
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Dual Space and Dual Map

![dual map](/assets/img/math/dual_map.png){: w="600" h="300" }

{: .prompt-tip }
> Suppose $V$ is finite-dimensional. Then $ V \cong V' $, but finding an isomorphism from $V$ onto $V'$ generally requires choosing a basis of $V$.

* $ \dim U^0 = \dim V - \dim U $

* $ \operatorname{null} T' = (\operatorname{range} T)^0 $
* $ \operatorname{range} T' = (\operatorname{null} T)^0 $

* $ \dim \operatorname{null} T' = \dim \operatorname{null} T + \dim W - \dim V $
* $ \dim \operatorname{range} T' = \dim \operatorname{range} T $

## Double Dual Space

{: .prompt-info }
> The _double dual space_ of $V$, denoted by $V''$, is defined to be the dual space of $V'$, i.e., $ V'' = (V')' $.
>
> Define $ \Lambda : V \to V'' $ by
>
> $$ (\Lambda v)(\varphi) = \varphi (v) $$
>
> for each $ v \in V $ and each $ \varphi \in V' $.

{: .prompt-info }
> If $ T \in \mathcal(V) $, then $ T'' \circ \Lambda = \Lambda \circ T $, where $ T'' = (T')' $.

{: .prompt-proof }
> Both sides are maps $V \to V''$, so fix $v \in V$ and show the two functionals $T''(\Lambda v)$ and $\Lambda(Tv)$ agree — i.e. give the same scalar on every $\varphi \in V'$.
>
> $$
> \big(T''(\Lambda v)\big)(\varphi)
> = \big((\Lambda v)\circ T'\big)(\varphi)
> = (\Lambda v)(T'\varphi)
> = (T'\varphi)(v)
> = (\varphi\circ T)(v)
> = \varphi(Tv).
> $$
>
> And the other side, straight from the definition of $\Lambda$:
>
> $$
> \big(\Lambda(Tv)\big)(\varphi) = \varphi(Tv).
> $$
>
> They match for all $\varphi$, so $T''(\Lambda v) = \Lambda(Tv)$ for all $v$, i.e. $T'' \circ \Lambda = \Lambda \circ T$. $\blacksquare$

{: .prompt-info }
> If $V$ is finite-dimensional, then $ V \cong_{\Lambda} V'' $. This isomorphism does not require a choice of basis and thus is considered more natural (_canonical_).

{: .prompt-proof }
> Suppose $\Lambda v = 0$, meaning $\varphi(v) = 0$ for *every* $\varphi \in V'$. So $v = 0$, and $\operatorname{null}\Lambda = \{0\}$: $\Lambda$ is _injective_.
>
> In finite dimensions $\dim V' = \dim V$, and applying this twice gives
>
> $$\dim V'' = \dim V' = \dim V.$$
>
> So $\Lambda$ is bijective, hence an isomorphism $V \xrightarrow{\sim} V''$. $\blacksquare$

![double dual space](/assets/img/math/double_dual_space.png)

## Annihilator

{: .prompt-info }
> _Annihilator_
>
> For $ U \subseteq V $,
>
> $$ U^0 = \{\varphi \in V' : \varphi(u) = 0 \ \forall u \in U \}. $$

{: .prompt-info }
> _Pre-annihilator_
>
> Suppose $V$ is finite-dimensional and $U$ is a subspace of $V$,
>
> $$ U = \{v \in V : \varphi(v) = 0 \ \forall \varphi \in U^0 \}. $$

{: .prompt-tip }
> Denote the collection of all subspaces of $V$ by $\mathrm{Sub}(V)$.
>
> $(\mathrm{Sub}(V), \subseteq)$ is a *lattice*, with $(\wedge, \vee) = (\cap, +)$.
>
> The annihilator map $ U \mapsto U^0 $ is an order-reversing *bijection* between $ \mathrm{Sub}(V) $ and $ \mathrm{Sub}(V') $.

{: .prompt-info }
> Suppose $ V $ is finite-dimensional and $ U $ and $ W $ are subspaces of $ V $.
>
> * Double annihilator: $ (U^0)^0 = U $
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
> $\blacksquare$

{: .prompt-info }
> Suppose $ V $ is finite-dimensional and $ \varphi_1, \dots, \varphi_m \in V' $. Let $ N = \operatorname{null}\varphi_1 \cap \cdots \cap \operatorname{null}\varphi_m \subseteq V$, $U = \operatorname{span}(\varphi_1,\dots,\varphi_m) \subseteq V'$. Then $ U = N^0$.

{: .prompt-proof }
> $N$ is the **pre-annihilator** of $U$ — the annihilator of $U$ taken back in $V$ under $V \cong V''$:
>
> $$v \in N \iff \varphi_i(v) = 0 \ \forall i \iff \varphi(v) = 0 \ \forall \varphi \in U \iff v \in U^0,$$
>
> So $N = U^0$ and $N^0 = U$. $\blacksquare$

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
