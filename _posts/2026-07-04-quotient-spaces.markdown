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
> *Translate (coset)*
>
> Suppose $ U $ is a subspace of $ V $ and $ v \in V $. The **translate** of $ U $ by $ v $ is
>
> $$ v + U = \{ v + u : u \in U \}. $$
>
> In the language of groups, $ v + U $ is the **coset** of the subgroup $ U $ in the additive group $ (V, +) $.

There are three ways to see the same object.

### Geometric View

A translate is **a parallel copy of $ U $ through $ v $**, which is exactly an [affine subspace]({% post_url 2026-06-28-affine-geometry %}#affine-subspace): $ U $ with the origin forgotten. It has no distinguished zero, but it still contains the whole line through any two of its points.

The subspace $ U $ is the *direction* of $ v + U $, and affine subspaces with the same direction are *parallel*. So the translates of $ U $ are all the affine subspaces with direction $ U $, and they tile $ V $: every vector lies on exactly one of them. For a line $ U $ in $ \mathbb{R}^3 $ they form a 2-parameter family of parallel lines.

### Algebraic View

A translate is **an [equivalence class](https://en.wikipedia.org/wiki/Equivalence_class)**. Define $ v \sim w \Leftrightarrow v - w \in U $, read "$ v $ and $ w $ are equal up to $ U $". The equivalence class of $ v $ is $ v + U $.

{: .prompt-tip }
> $ \sim $ is an equivalence relation precisely because $ U $ is a subspace:
>
> - reflexive, because $ 0 \in U $;
> - symmetric, because $ U $ is closed under negation;
> - transitive, because $ U $ is closed under addition.
>
> Every result below about translates is a restatement of this.

### [Fiber](https://en.wikipedia.org/wiki/Fiber_(mathematics)) View

A translate is **a level set of a linear map**. If $ T $ is linear with $ \operatorname{null} T = U $, then $ Tv = Tw \Leftrightarrow v - w \in U $, so the translates of $ U $ are exactly the nonempty level sets of $ T $.

{: .prompt-tip }
> *Solution sets*
>
> Suppose $ T \in \mathcal{L}(V, W) $ and $ Tx_0 = b $. Then
>
> $$ \{ x \in V : Tx = b \} = x_0 + \operatorname{null} T. $$
>
> The solution set of a linear system is a translate of the solution set of the homogeneous system.

### Example

Let $ V = \mathbb{R}^2 $ and $ U = \operatorname{span}((1,1)) $. The translates of $ U $ are the lines of slope 1.

![Translates of a subspace](/assets/img/math/translates_of_a_subspace.png){: w="400" h="270" }

- **Equivalence class.** $ (a, b) \sim (a', b') \Leftrightarrow a - b = a' - b' $. The points $ v = (2, 1) $ and $ x = (0.5, -0.5) $ lie on the same line, so either one names it: $ v + U = x + U $.
- **Fiber.** Each line is a level set of $ T(a, b) = a - b $, whose null space is $ U $.
- **One point per line.** Each line crosses $ W = \operatorname{span}((1,0)) $ exactly once, at $ (c, 0) $ with $ c = a - b $. So the family of translates is labelled by a single number: it is 1-dimensional, which is $ \dim V - \dim U $.

In $ V/U $, each of these lines is a single point.

### Properties

Translates of $ U $ are the equivalence classes of $ v \sim w \Leftrightarrow v - w \in U $. Hence any element of a translate represents it, two translates are equal or disjoint, and $ U $ is the class of $ 0 $. The intersection of a translate of $ U_1 $ and a translate of $ U_2 $ is empty or a translate of $ U_1 \cap U_2 $.

{: .prompt-info }
> *Re-basing*
>
> Suppose $ U $ is a subspace of $ V $ and $ v, x \in V $. Then
>
> $$ x \in v + U \Leftrightarrow x + U = v + U. $$

{: .prompt-tip }
> Corollary: Two translates of a subspace are equal or disjoint. For $ v, w \in V $,
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
> *Complement $\Leftrightarrow$ isomorphism*
>
> Suppose $ U $ and $ W $ are subspaces of $ V $ and $ \pi : V \to V/U $ is the quotient map $ \pi(v) = v + U $. Then
>
> $$ V = U \oplus W \Leftrightarrow \pi|_W : W \to V/U \text{ is an isomorphism.} $$

{: .prompt-proof }
> - $ \operatorname{null}(\pi\|_W) = U \cap W $, so $ \pi\|_W $ is injective if and only if $ U \cap W = \\{0\\} $.
> - $ v + U \in \operatorname{range}(\pi\|_W) $ if and only if $ v = w + u $ for some $ w \in W $, $ u \in U $, so $ \pi\|_W $ is surjective if and only if $ U + W = V $. $\blacksquare$

Geometrically, $ W $ is a complement of $ U $ exactly when it crosses every translate of $ U $ once: it is a set of representatives that happens to be a subspace. The map $ \pi $ is canonical, but $ W $ is a choice.

{: .prompt-tip }
> *Project*
>
> Suppose $ V = U \oplus W $ and $ w_1, \dots, w_m $ is a basis of $ W $. Then $ w_1 + U, \dots, w_m + U $ is a basis of $ V/U $, because isomorphisms preserve bases.

{: .prompt-tip }
> *Lift*
>
> Suppose $ U $ is a subspace of $ V $ and $ v_1 + U, \dots, v_m + U $ is a basis of $ V/U $. Let $ W = \operatorname{span}(v_1, \dots, v_m) $. Then
>
> - $ V = U \oplus W $ and $ \dim W = \dim V/U $;
> - $ v_1, \dots, v_m $ is a basis of $ W $;
> - if $ u_1, \dots, u_n $ is a basis of $ U $, then $ v_1, \dots, v_m, u_1, \dots, u_n $ is a basis of $ V $.
>
> In particular, if $ V/U $ is finite-dimensional, then $ U $ has a finite-dimensional complement, even when $ V $ is not finite-dimensional.

{: .prompt-proof }
> $ \pi\|_W $ sends the spanning list $ v_1, \dots, v_m $ of $ W $ to a basis of $ V/U $.
>
> - **Surjective:** its range contains a basis of $ V/U $.
> - **Injective:** if $ \pi\big(\sum_k a_k v_k\big) = \sum_k a_k (v_k + U) = 0 $, then all $ a_k = 0 $.
>
> So $ \pi\|_W $ is an isomorphism, which gives $ V = U \oplus W $ and $ \dim W = \dim V/U = m $. A spanning list of $ W $ of length $ \dim W $ is a basis of $ W $, and a basis of $ W $ followed by a basis of $ U $ is a basis of $ U \oplus W = V $. $\blacksquare$

## Quotient Operator

{: .prompt-info }
> Suppose $ V $ is finite-dimensional, $ T \in \mathcal{L}(V) $, and $ U $ is a subspace of $ V $ invariant under $ T $. The quotient operator $ T/U \in \mathcal{L}(V/U) $ is defined by
>
> $$ (T/U)(v + U) = Tv + U $$
>
> for each $ v \in V $.
