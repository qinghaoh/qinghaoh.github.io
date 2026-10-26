---
title:  "Dual Space"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Notation

{% include notation-table.md keys="V LVW dual_space dual_op double_dual Lambda annihilator" %}

## Dual Space

{: .prompt-info }
> A _linear functional_ on $V$ is an element of $\mathcal{L}(V, \mathbf{F})$. The _dual space_ of $V$ is $V' = \mathcal{L}(V, \mathbf{F})$.
>
> If $V$ is finite-dimensional, then $\dim V' = \dim V \cdot \dim \mathbf{F} = \dim V$.

{: .prompt-tip }
> Suppose $V$ is finite-dimensional. Then $ V \cong V' $, but finding an isomorphism from $V$ onto $V'$ generally requires choosing a basis of $V$.

### Dual Basis

{: .prompt-info }
> If $ v_1, \dots, v_n $ is a basis of $V$, its _dual basis_ is the list $ \varphi_1, \dots, \varphi_n \in V' $ defined by
>
> $$ \varphi_j(v_k) = \delta_{jk}. $$
>
> It is a basis of $V'$, and the two bases are *coordinate-readers for each other*:
>
> * vector: $ v = \varphi_1(v)\,v_1 + \dots + \varphi_n(v)\,v_n $ for every $ v \in V $
> * covector: $ \psi = \psi(v_1)\,\varphi_1 + \dots + \psi(v_n)\,\varphi_n $ for every $ \psi \in V' $

{: .prompt-proof }
> Vector: if $v = a_1 v_1 + \dots + a_n v_n$, then applying $\varphi_i$ gives $\varphi_i(v) = a_i$.
>
> Covector: both sides are linear functionals, and on each basis vector $v_j$ the right side gives $\sum_i \psi(v_i)\,\delta_{ij} = \psi(v_j)$. Two linear maps that agree on a basis are equal.
>
> The covector formula shows $\varphi_1, \dots, \varphi_n$ spans $V'$; since it has length $n = \dim V'$, it is a basis. $\blacksquare$

**Coordinate = evaluate the dual basis vector.** The $i$-th coordinate of $v$ is $\varphi_i(v)$, and the $i$-th coordinate of $\psi$ is $\psi(v_i)$.

## Dual Map

{: .prompt-info }
> Suppose $ T \in \mathcal{L}(V, W) $. The _dual map_ of $T$ is $ T' \in \mathcal{L}(W', V') $ defined by
>
> $$ T'(\varphi) = \varphi \circ T $$
>
> for each $ \varphi \in W' $.

![dual map](/assets/img/math/dual_map.png){: w="720" h="400" }

## Annihilator

{: .prompt-info }
> For $ U \subseteq V $, the _annihilator_ of $U$ is
>
> $$ U^0 = \{\varphi \in V' : \varphi(u) = 0 \ \forall u \in U \}. $$
>
> For $ \Gamma \subseteq V' $, the _pre-annihilator_ of $\Gamma$ is
>
> $$ \Gamma_0 = \{v \in V : \varphi(v) = 0 \ \forall \varphi \in \Gamma \}. $$

Both are subspaces, and both reverse inclusions: $U_1 \subseteq U_2 \implies U_2^0 \subseteq U_1^0$, and likewise for $(\cdot)_0$.

{: .prompt-info }
> Suppose $V$ is finite-dimensional and $U$ is a subspace of $V$. Then
>
> $$ \dim U^0 = \dim V - \dim U. $$

{: .prompt-proof }
> Extend a basis $u_1, \dots, u_m$ of $U$ to a basis $u_1, \dots, u_m, v_{m+1}, \dots, v_n$ of $V$, with dual basis $\varphi_1, \dots, \varphi_n$. By the covector expansion, $\psi \in U^0$ iff $\psi(u_1) = \dots = \psi(u_m) = 0$ iff $\psi \in \operatorname{span}(\varphi_{m+1}, \dots, \varphi_n)$. So $\dim U^0 = n - m$. $\blacksquare$

## Null Space and Range of the Dual Map

{: .prompt-info }
> Suppose $ T \in \mathcal{L}(V, W) $. Then
>
> * $ \operatorname{null} T' = (\operatorname{range} T)^0 $
>
> If, in addition, $V$ and $W$ are finite-dimensional, then
>
> * $ \operatorname{range} T' = (\operatorname{null} T)^0 $
> * $ \dim \operatorname{null} T' = \dim \operatorname{null} T + \dim W - \dim V $
> * $ \dim \operatorname{range} T' = \dim \operatorname{range} T $

{: .prompt-proof }
> Null space: $ \varphi \in \operatorname{null} T' \iff \varphi \circ T = 0 \iff \varphi(Tv) = 0 \ \forall v \iff \varphi \in (\operatorname{range} T)^0 $.
>
> Dimensions: by the annihilator dimension formula and rank-nullity,
>
> $$ \dim \operatorname{null} T' = \dim W - \dim \operatorname{range} T = \dim W - \dim V + \dim \operatorname{null} T, $$
>
> and then rank-nullity for $T'$ (using $\dim W' = \dim W$) gives $\dim \operatorname{range} T' = \dim W - \dim \operatorname{null} T' = \dim \operatorname{range} T$.
>
> Range: if $\psi = T'\varphi = \varphi \circ T$, then $\psi$ vanishes on $\operatorname{null} T$, so $\operatorname{range} T' \subseteq (\operatorname{null} T)^0$. Both sides have dimension $\dim V - \dim \operatorname{null} T = \dim \operatorname{range} T$, so they are equal. $\blacksquare$

{: .prompt-info }
> Suppose $V$ and $W$ are finite-dimensional and $ T \in \mathcal{L}(V, W) $. Then
>
> * $ T $ is injective $ \iff T' $ is surjective
> * $ T $ is surjective $ \iff T' $ is injective

{: .prompt-proof }
> $T' \text{ injective} \iff (\operatorname{range} T)^0 = \{0\} \iff \operatorname{range} T = W$, using $\dim (\operatorname{range} T)^0 = \dim W - \dim \operatorname{range} T$.
>
> $T' \text{ surjective} \iff (\operatorname{null} T)^0 = V' \iff \operatorname{null} T = \{0\}$, using $\dim (\operatorname{null} T)^0 = \dim V - \dim \operatorname{null} T$. $\blacksquare$

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
> If $ T \in \mathcal{L}(V) $, then $ T'' \circ \Lambda = \Lambda \circ T $, where $ T'' = (T')' $.

{: .prompt-proof }
> Both sides are maps $V \to V''$, so fix $v \in V$ and show the two functionals $T''(\Lambda v)$ and $\Lambda(Tv)$ agree, i.e., give the same scalar on every $\varphi \in V'$:
>
> $$
> \big(T''(\Lambda v)\big)(\varphi)
> = \big((\Lambda v)\circ T'\big)(\varphi)
> = (\Lambda v)(T'\varphi)
> = (T'\varphi)(v)
> = (\varphi\circ T)(v)
> = \varphi(Tv)
> = \big(\Lambda(Tv)\big)(\varphi).
> $$
>
> $\blacksquare$

{: .prompt-info }
> If $V$ is finite-dimensional, then $\Lambda$ is an isomorphism from $V$ onto $V''$. This isomorphism does not require a choice of basis and thus is considered more natural (_canonical_).

{: .prompt-proof }
> Suppose $v \ne 0$. Extend $v$ to a basis of $V$ and let $\varphi$ be the first vector of the dual basis; then $(\Lambda v)(\varphi) = \varphi(v) = 1 \ne 0$. So $\operatorname{null}\Lambda = \{0\}$ and $\Lambda$ is injective.
>
> Applying $\dim V' = \dim V$ twice gives $\dim V'' = \dim V' = \dim V$, so $\Lambda$ is an isomorphism. $\blacksquare$

![double dual space](/assets/img/math/double_dual_space.png)

{: .prompt-tip }
> Under $\Lambda$, the pre-annihilator is the annihilator taken one level up: for $\Gamma \subseteq V'$,
>
> $$ \Lambda(\Gamma_0) = \Gamma^0 \subseteq V'', $$
>
> since $\Lambda v \in \Gamma^0 \iff \varphi(v) = 0 \ \forall \varphi \in \Gamma$. This is why $(U^0)_0$ is often written $(U^0)^0$.

## Annihilator Correspondence

Denote the collection of all subspaces of $V$ by $\mathrm{Sub}(V)$. $(\mathrm{Sub}(V), \subseteq)$ is a *lattice*, with $(\wedge, \vee) = (\cap, +)$.

{: .prompt-info }
> Suppose $V$ is finite-dimensional, $U$ is a subspace of $V$ and $\Gamma$ is a subspace of $V'$. Then
>
> * $ (U^0)_0 = U $
> * $ (\Gamma_0)^0 = \Gamma $

{: .prompt-proof }
> $(U^0)_0 = U$: every $u \in U$ is killed by every $\varphi \in U^0$, so $U \subseteq (U^0)_0$. Conversely, suppose $v \notin U$. Extend a basis $u_1, \dots, u_m$ of $U$ to a basis $u_1, \dots, u_m, v, \dots$ of $V$ and let $\varphi$ be the dual basis vector with $\varphi(u_i) = 0$ and $\varphi(v) = 1$. Then $\varphi \in U^0$ but $\varphi(v) \ne 0$, so $v \notin (U^0)_0$.
>
> $(\Gamma_0)^0 = \Gamma$: every $\varphi \in \Gamma$ kills $\Gamma_0$, so $\Gamma \subseteq (\Gamma_0)^0$. By the tip above, $\dim \Gamma_0 = \dim \Gamma^0 = \dim V'' - \dim \Gamma = \dim V - \dim \Gamma$, so $\dim (\Gamma_0)^0 = \dim V - \dim \Gamma_0 = \dim \Gamma$. $\blacksquare$

{: .prompt-tip }
> So $ U \mapsto U^0 $ is an order-reversing *bijection* from $ \mathrm{Sub}(V) $ onto $ \mathrm{Sub}(V') $, with inverse $ \Gamma \mapsto \Gamma_0 $. In particular, for subspaces $U$, $W$ of $V$,
>
> $$ W^0 \subseteq U^0 \iff U \subseteq W, $$
>
> since applying $(\cdot)_0$ to $W^0 \subseteq U^0$ gives $(U^0)_0 \subseteq (W^0)_0$, i.e., $U \subseteq W$.

{: .prompt-info }
> Suppose $ V $ is finite-dimensional and $ U $ and $ W $ are subspaces of $ V $. The annihilator map is a lattice anti-homomorphism:
>
> * $ (U + W)^0 = U^0 \cap W^0 $
> * $ (U \cap W)^0 = U^0 + W^0 $

{: .prompt-proof }
> The first identity is direct: $\varphi$ kills $U + W$ iff it kills both $U$ and $W$. The same argument in $V'$ gives $(\Gamma + \Delta)_0 = \Gamma_0 \cap \Delta_0$ for subspaces $\Gamma, \Delta$ of $V'$. Apply it to $\Gamma = U^0$, $\Delta = W^0$:
>
> $$(U^0 + W^0)_0 = (U^0)_0 \cap (W^0)_0 = U \cap W.$$
>
> Taking annihilators of both sides and using $(\Gamma_0)^0 = \Gamma$ on the left gives $U^0 + W^0 = (U \cap W)^0$. $\blacksquare$

{: .prompt-info }
> Suppose $ V $ is finite-dimensional and $ \varphi_1, \dots, \varphi_m \in V' $. Then
>
> $$ \operatorname{span}(\varphi_1,\dots,\varphi_m) = \big(\operatorname{null}\varphi_1 \cap \dots \cap \operatorname{null}\varphi_m\big)^0. $$

{: .prompt-proof }
> Let $\Gamma = \operatorname{span}(\varphi_1,\dots,\varphi_m)$. Then
>
> $$ v \in \operatorname{null}\varphi_1 \cap \dots \cap \operatorname{null}\varphi_m \iff \varphi_i(v) = 0 \ \forall i \iff \varphi(v) = 0 \ \forall \varphi \in \Gamma \iff v \in \Gamma_0, $$
>
> so the intersection is $\Gamma_0$, and its annihilator is $(\Gamma_0)^0 = \Gamma$. $\blacksquare$

## Duality Swaps Spanning and Independence

Given $v_1,\dots,v_m \in V$, define

$$S: \mathbf{F}^m \to V, \qquad S(a_1,\dots,a_m) = a_1 v_1 + \dots + a_m v_m \quad (\text{so } e_i \mapsto v_i).$$

Its dual map $S': V' \to (\mathbf{F}^m)' \cong \mathbf{F}^m$ is $S'(\varphi) = \varphi \circ S$, whose $i$-th coordinate is $(\varphi \circ S)(e_i) = \varphi(v_i)$. That is,

$$S'(\varphi) = \big(\varphi(v_1),\dots,\varphi(v_m)\big).$$

If $V$ is finite-dimensional, the injective/surjective swap gives:

| map property                           | list statement               |
| -------------------------------------- | ---------------------------- |
| $S$ surjective $\iff S'$ injective     | $(v_i)$ spans $V$            |
| $S$ injective $\iff S'$ surjective     | $(v_i)$ linearly independent |

{: .prompt-warning }
> Only the equivalences with $S'$ use finite-dimensionality. The statements about $S$ hold in any dimension.

Running the same construction with $V'$ in place of $V$ gives a statement about lists of functionals. For $\varphi_1, \dots, \varphi_m \in V'$, let $R : \mathbf{F}^m \to V'$, $e_i \mapsto \varphi_i$. Then $R' : V'' \to \mathbf{F}^m$ and, through $\Lambda$, $R'(\Lambda v) = (\varphi_1(v), \dots, \varphi_m(v))$.

{: .prompt-info }
> Suppose $V$ is finite-dimensional, $\varphi_1, \dots, \varphi_m \in V'$, and
>
> $$ \Gamma:V\to\mathbf{F}^m,\qquad \Gamma(v)=\big(\varphi_1(v),\dots,\varphi_m(v)\big). $$
>
> * $\Gamma$ injective $\iff (\varphi_i)$ spans $V'$
> * $\Gamma$ surjective $\iff (\varphi_i)$ is linearly independent

{: .prompt-proof }
> $\Gamma = R' \circ \Lambda$ and $\Lambda$ is an isomorphism, so $\Gamma$ is injective (surjective) iff $R'$ is, iff $R$ is surjective (injective). $\blacksquare$

{: .prompt-info }
> Suppose $V$ is finite-dimensional and $ \varphi_1, \dots, \varphi_n $ is a basis of $V'$. Then there exists a basis of $V$ whose dual basis is $ \varphi_1, \dots, \varphi_n $.

{: .prompt-proof }
> By the previous result, $\Gamma(v) = (\varphi_1(v), \dots, \varphi_n(v))$ is an isomorphism $V \to \mathbf{F}^n$. Set $v_k=\Gamma^{-1}(e_k)$, where $e_1,\dots,e_n$ is the standard basis of $\mathbf{F}^n$. These form a basis of $V$, and $\Gamma(v_k)=e_k$ unpacks to $\varphi_j(v_k)=\delta_{jk}$, as required.
> $\blacksquare$
