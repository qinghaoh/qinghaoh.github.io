---
title:  "Extension and Lift: A Factoring Duality"
category: math
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Extension and Lift: A Factoring Duality

{: .prompt-info }
> *Extension (shared domain)*
>
> Suppose $ W_1 $ is finite-dimensional, $ S \in \mathcal{L}(V, W_1) $ and $ T \in \mathcal{L}(V, W_2) $. Then
> $$ \operatorname{null} S \subseteq \operatorname{null} T \iff \exists\, E \in \mathcal{L}(W_1, W_2) \ \text{s.t.}\ T = ES. $$

{: .prompt-proof }
> ($\Leftarrow$) If $T = ES$ and $Sv = 0$, then $Tv = E(Sv) = E0 = 0$, so $\operatorname{null} S \subseteq \operatorname{null} T$.
>
> ($\Rightarrow$) We build $E : W_1 \to W_2$ with $E(Sv) = Tv$ for all $v$.
>
> For $w \in \operatorname{range} S$, write $w = Sv$ and set $Ew := Tv$.
> This is well-defined: if $Sv_1 = Sv_2$, then
> $v_1 - v_2 \in \operatorname{null} S \subseteq \operatorname{null} T$, so $Tv_1 = Tv_2$.
> It is linear on $\operatorname{range} S$ since, writing $w_i = Sv_i$,
> $$ E(aw_1 + bw_2) = E\bigl(S(av_1+bv_2)\bigr) = T(av_1+bv_2) = aEw_1 + bEw_2. $$
>
> Because $W_1$ is finite-dimensional, $\operatorname{range} S$ has a complement:
> $$ W_1 = \operatorname{range} S \oplus U. $$
> Define $E$ to be $0$ on $U$ and extend linearly to all of $W_1$. Then $Ew \in W_2$ for every $w$, and $ES = T$.
> $\blacksquare$

{: .prompt-info }
> *Lift (shared codomain)*
>
> Suppose $ V_1 $ is finite-dimensional, $ S \in \mathcal{L}(V_1, W) $ and $ T \in \mathcal{L}(V_2, W) $. Then
> $$ \operatorname{range} S \subseteq \operatorname{range} T \iff \exists\, E \in \mathcal{L}(V_1, V_2) \ \text{s.t.}\ S = TE. $$

{: .prompt-proof }
> ($\Leftarrow$) If $S = TE$, then $Sv = T(Ev) \in \operatorname{range} T$ for all $v$, so $\operatorname{range} S \subseteq \operatorname{range} T$.
>
> ($\Rightarrow$) Because $V_1$ is finite-dimensional, pick a basis $v_1, \dots, v_n$ of $V_1$. For each $j$, the vector $Sv_j$ lies in $\operatorname{range} S \subseteq \operatorname{range} T$, so there exists $u_j \in V_2$ with
> $$ T u_j = S v_j. $$
> (The preimage $u_j$ lives in $V_2$ because $T$ starts at $V_2$ — which is exactly why $E$ ends up in $\mathcal{L}(V_1, V_2)$.) Define $E \in \mathcal{L}(V_1, V_2)$ to be the unique linear map with $E v_j = u_j$ for every $j$. Then for each basis vector,
> $$ (TE)(v_j) = T(E v_j) = T(u_j) = S v_j. $$
> So $TE$ and $S$ agree on a basis of $V_1$, hence $TE = S$.
> $\blacksquare$

![factoring](/assets/img/math/factoring.png)

{: .prompt-tip }
> *Extension and Lift are transposes.*
>
> The two laws are one statement seen through the dual map $M \mapsto M'$. Transpose the Extension data — $S : V \to W_1$ and $T : V \to W_2$, sharing the domain $V$ — to get $S' : W_1' \to V'$ and $T' : W_2' \to V'$, now sharing the *codomain* $V'$, which is a Lift configuration. The conclusion transposes with it, $T = ES \mapsto T' = S'E'$, turning "$T$ factors through $S$ on the left" into "$T'$ factors through $S'$ on the right."
>
> And the hypotheses correspond: since $\operatorname{range} M' = (\operatorname{null} M)^0$, and since $U \mapsto U^0$ is an inclusion-reversing **bijection** on subspaces (its inverse is $U^0 \mapsto U^{00} = U$, the double-annihilator identity), $$ \operatorname{null} S \subseteq \operatorname{null} T \iff (\operatorname{null} T)^0 \subseteq (\operatorname{null} S)^0 \iff \operatorname{range} T' \subseteq \operatorname{range} S'. $$ So the Extension problem for $(S, T)$ is exactly the Lift problem for its transposes $(T', S')$.

{: .prompt-tip }
> *Removing the finite-dimensional hypotheses*
>
> Both laws hold in **every** dimension under the Axiom of Choice; the
> finite-dimensional hypotheses are conveniences, not necessities.
>
> - **Extension** used $W_1$ finite-dimensional only to give $\operatorname{range} S$ a complement in $W_1$. Under AC, every subspace of every vector space has a complement, so the hypothesis can be dropped.
> - **Lift** used $V_1$ finite-dimensional only to supply a basis of $V_1$ and to choose a preimage $u_j$ for each basis vector. Under AC, every vector space has a (Hamel) basis and the preimages can be chosen simultaneously, so the hypothesis can be dropped.
>
> The reverse ($\Leftarrow$) directions use neither a complement nor a basis, so they are choice-free in all dimensions. Structurally, the two facts being invoked are that in $\mathbf{Vect}$ every object is both **injective** (extension along monos) and **projective** (lifting along epis).

## Left and Right Inverse

Setting one given map to an **identity** collapses each law into a one-sided inverse:

**Extension.** Put $T = I_V$. The hypothesis $\operatorname{null} S \subseteq \operatorname{null} I_V = \{0\}$ says $S$ is **injective**, and $I_V = ES$ makes $E$ a **left inverse** of $S$. A left inverse is called a **retraction** of $S$, and an injective map that admits one is a **split monomorphism**.

**Lift.** Put $S = I_W$. The hypothesis $W = \operatorname{range} I_W \subseteq \operatorname{range} T$ says $T$ is **surjective**, and $I_W = TE$ makes $E$ a **right inverse** of $T$. A right inverse is called a **section** of $T$, and a surjective map that admits one is a **split epimorphism**.

So *injective $\iff$ has a retraction* and *surjective $\iff$ has a section*.

A retraction is automatically surjective and a section automatically injective, so the two constructions say that in $\mathbf{Vect}$ **every injective map splits** (has a retraction) and **every surjective map splits** (has a section) — the linear-algebra form of "every mono and every epi splits."

{: .prompt-tip }
> *These two equivalences are transposes of each other*, inherited from the Extension↔Lift duality above.
>
> Since setting a map to the identity commutes with transposing, the retraction↔section pairing is that duality evaluated at $I$: a retraction $ST = I_V$ dualizes to $T'S' = I_{V'}$, a section of $T'$. Bringing in the second annihilator identity $\operatorname{null} M' = (\operatorname{range} M)^0$ alongside the first completes the dictionary between the two maps' properties:
>
> $$ T \text{ injective} \iff T' \text{ surjective}, \qquad T \text{ surjective} \iff T' \text{ injective}. $$
>
> So the transpose swaps injective with surjective in the very stroke that swaps retraction with section: "$T$ is injective, hence has a retraction" read on the dual side is exactly "$T'$ is surjective, hence has a section." The left/right asymmetry is one phenomenon, seen through $M \mapsto M'$.

| relationship                                              | factorization                               | $E$ acts on                |
| --------------------------------------------------------- | ------------------------------------------- | -------------------------- |
| $\operatorname{range} S \subseteq \operatorname{range} T$ | $S = TE$, $E \in \mathcal{L}(V)$            | domain (one-directional)   |
| $\operatorname{null} S \subseteq \operatorname{null} T$   | $T = ES$, $E \in \mathcal{L}(W)$            | codomain (one-directional) |
| $\operatorname{range} S = \operatorname{range} T$         | $S = TE$, $E \in \mathcal{L}(V)$ invertible | domain (isomorphism)       |
| $\operatorname{null} S = \operatorname{null} T$           | $S = ET$, $E \in \mathcal{L}(W)$ invertible | codomain (isomorphism)     |

## References

* [lift in nLab](https://ncatlab.org/nlab/show/lift)
* [extension in nLab](https://ncatlab.org/nlab/show/extension)
