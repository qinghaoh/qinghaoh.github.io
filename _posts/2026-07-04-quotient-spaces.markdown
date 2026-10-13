---
title:  "Quotient Spaces"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Notation

{% include notation-table.md keys="V LV LVW MT quotient_space quotient_op" %}

## Translate

{: .prompt-info }
> *Translate (coset)*
>
> Suppose $ U $ is a subspace of $ V $ and $ v \in V $. The **translate** of $ U $ by $ v $ is
>
> $$ v + U = \{ v + u : u \in U \}. $$
>
> In the language of groups, $ v + U $ is the **coset** of the subgroup $ U $ in the additive group $ (V, +) $.

For example, let $ V = \mathbb{R}^2 $, with vectors written $ (a, b) $, and $ U = \operatorname{span}((1,1)) $. The translates of $ U $ are the lines of slope 1. Each one is the set of vectors with a fixed value $ c $ of $ a - b $, and the figure labels five of them by $ c $.

![Translates of a subspace](/assets/img/math/translates_of_a_subspace.png){: w="400" h="270" }

There are three ways to see the same object.

### Geometric View

A translate is **a parallel copy of $ U $ through $ v $**, which is exactly an [affine subspace]({% post_url 2026-06-28-affine-geometry %}#affine-subspace).

The subspace $ U $ is the *direction* of $ v + U $, and affine subspaces with the same direction are *parallel*. So the translates of $ U $ are all the affine subspaces with direction $ U $. They tile $ V $: every vector lies on exactly one of them. Different subspaces give different tilings, because the tile through $ 0 $ is $ U $ itself.

{: .prompt-tip }
> Every subspace is an affine subspace, namely the translate through the origin: $ v + U = U $ exactly when $ v \in U $. Every other translate of $ U $ is an affine subspace but not a subspace, because it does not contain $ 0 $.

### Algebraic View

A translate is **an [equivalence class](https://en.wikipedia.org/wiki/Equivalence_class)**. Define $ v \sim w \Leftrightarrow v - w \in U $, read "$ v $ and $ w $ are equal up to $ U $". The equivalence class of $ v $ is $ v + U $, and the equivalence classes of any equivalence relation partition the set. This is why the translates tile $ V $.

In the example, $ (a, b) \sim (a', b') \Leftrightarrow a - b = a' - b' $. The points $ v = (2, 1) $ and $ x = (0.5, -0.5) $ lie on the same line, so either one names it: $ v + U = x + U $.

{: .prompt-tip }
> *Why a subspace?*
>
> $ \sim $ is an equivalence relation because of three properties of $ U $:
>
> - reflexive, because $ 0 \in U $;
> - symmetric, because $ U $ is closed under negation;
> - transitive, because $ U $ is closed under addition.
>
> Without them, translates can overlap without coinciding: in $ \mathbb{R} $, the intervals $ [0, 1] $ and $ 0.5 + [0, 1] $ do.
>
> Scalar multiplication is not used. The translates of $ \mathbb{Z}^2 $ tile $ \mathbb{R}^2 $ for the same reason, although $ \mathbb{Z}^2 $ is not a subspace. Closure under scalar multiplication is needed only later, to make the set of translates a vector space.

### [Fiber](https://en.wikipedia.org/wiki/Fiber_(mathematics)) View

A translate is **a [level set](https://en.wikipedia.org/wiki/Level_set) of a linear map**. Suppose $ T \in \mathcal{L}(V, W) $ and $ \operatorname{null} T = U $. Then $ Tv = Tw \Leftrightarrow v - w \in U $, so the level set of $ T $ through $ v $ is

$$ \{ x \in V : Tx = Tv \} = v + U. $$

The translates of $ U $ are exactly the nonempty level sets of $ T $. Read as a statement about the equation $ Tx = c $, this is the [structure of the solution set]({% post_url 2026-06-30-systems-of-linear-equation %}#the-solution-set) of a linear system: one particular solution plus the solution set of the homogeneous system.

In the example, each line is a level set of $ T(a, b) = a - b $, whose null space is $ U $, and the labels in the figure are the values of $ T $ on each line. This $ T $ is a single nonzero linear functional, so its level sets are [hyperplanes]({% post_url 2026-06-28-affine-geometry %}#hyperplanes). In general, each coordinate of $ T $ is a linear functional, so a level set of $ T $ is an intersection of hyperplanes.

### Properties

Each property is stated once as a formula and then read in the three views.

{: .prompt-info }
> *Equal or disjoint*
>
> Suppose $ U $ is a subspace of $ V $ and $ v, w \in V $. Then the following are equivalent:
>
> - $ w \in v + U $;
> - $ v - w \in U $;
> - $ v + U = w + U $;
> - $ (v + U) \cap (w + U) \ne \emptyset $.
>
> In particular, $ v \in U \Leftrightarrow v + U = U $.

A translate has no preferred base point: any of its points can name it. So two translates of the same subspace cannot cross: if they share a point, re-basing both at that point makes them the same set.

![Translates are equal or disjoint](/assets/img/math/translates_equal_or_disjoint.png){: w="400" h="270" }

- **Geometric.** A line is determined by its direction and any one point on it, so parallel lines either coincide or never meet. They coincide exactly when the displacement $ v - w $ points along $ U $ (left); otherwise it carries $ w $ off its line onto a different parallel copy (right). The copy through the origin is $ U $ itself.
- **Algebraic.** Any member of an equivalence class represents the class, two classes are equal or disjoint, and $ U $ is the class of $ 0 $.
- **Fiber.** $ w $ lies in the level set through $ v $ exactly when $ Tw = Tv $, that is, when $ T(v - w) = 0 $. A point has only one value under $ T $, so level sets for different values share no point. The level set for the value $ 0 $ is $ \operatorname{null} T = U $.

{: .prompt-info }
> *Intersection*
>
> Suppose $ U_1 $ and $ U_2 $ are subspaces of $ V $ and $ v, w \in V $. Then $ (v + U_1) \cap (w + U_2) $ is empty or a translate of $ U_1 \cap U_2 $: for every $ x $ in the intersection,
>
> $$ (v + U_1) \cap (w + U_2) = (x + U_1) \cap (x + U_2) = x + (U_1 \cap U_2). $$

Translates of different subspaces can cross, and where they do, the crossing is again a translate.

![Intersection of two translates](/assets/img/math/intersection_of_translates.png){: w="400" h="270" }

- **Geometric.** Re-base both at a common point $ x $. A step away from $ x $ stays in both exactly when it lies in both directions. In the figure, two non-parallel planes in $ \mathbb{R}^3 $ meet in a line whose direction is the intersection of their directions. Two distinct parallel planes do not meet at all.
- **Algebraic.** Being equal up to $ U_1 $ and also equal up to $ U_2 $ is the same as being equal up to $ U_1 \cap U_2 $.
- **Fiber.** If $ U_1 = \operatorname{null} T_1 $ and $ U_2 = \operatorname{null} T_2 $, the intersection is the solution set of the two equations $ T_1 x = T_1 v $ and $ T_2 x = T_2 w $ taken together. Stacked into one map $ x \mapsto (T_1 x, T_2 x) $ with null space $ U_1 \cap U_2 $, they have a solution set that is empty or a translate of that null space.

## Quotient Space

{: .prompt-info }
> *Quotient space*
>
> Suppose $ U $ is a subspace of $ V $. The **quotient space** $ V/U $ is the set of all translates of $ U $,
>
> $$ V/U = \{ v + U : v \in V \}, $$
>
> with addition and scalar multiplication defined by
>
> $$ (v + U) + (w + U) = (v + w) + U, \qquad \lambda (v + U) = \lambda v + U. $$
>
> With these operations $ V/U $ is a vector space, and its zero vector is $ 0 + U = U $.

Each translate, a whole subset of $ V $, is a single vector of $ V/U $. Both operations are defined through representatives, so the results must not depend on which representatives are used.

{: .prompt-proof }
> Suppose $ v + U = v' + U $ and $ w + U = w' + U $. Then $ v - v' \in U $ and $ w - w' \in U $, so
>
> $$ (v + w) - (v' + w') = (v - v') + (w - w') \in U, \qquad \lambda v - \lambda v' = \lambda (v - v') \in U. $$
>
> By [Equal or disjoint](#properties), $ (v + w) + U = (v' + w') + U $ and $ \lambda v + U = \lambda v' + U $. The vector space axioms hold in $ V/U $ because they hold for the representatives in $ V $. $\blacksquare$

In the example, each translate is the line of vectors $ (a, b) $ with $ a - b = c $, labelled $ c $ in the first figure. Adding the lines labelled $ c $ and $ c' $ gives the line labelled $ c + c' $, and $ \lambda $ times the line labelled $ c $ is the line labelled $ \lambda c $. So $ V/U $ behaves like the number line of the labels.

{: .prompt-info }
> *Quotient map*
>
> Suppose $ U $ is a subspace of $ V $. The **quotient map** is
>
> $$ \pi : V \to V/U, \qquad \pi(v) = v + U. $$
>
> It is linear and surjective, and $ \operatorname{null} \pi = U $. If $ V $ is finite-dimensional, then
>
> $$ \dim V/U = \dim V - \dim U. $$

{: .prompt-proof }
> Linearity is the definition of the operations on $ V/U $, and $ \pi $ is surjective because every element of $ V/U $ has the form $ v + U = \pi(v) $. Also $ \pi(v) = 0 $ means $ v + U = U $, that is, $ v \in U $. The dimension formula is the fundamental theorem of linear maps, $ \dim V = \dim \operatorname{null} \pi + \dim \operatorname{range} \pi $. $\blacksquare$

So every subspace $ U $ is the null space of a linear map, and the [fiber view](#fiber-view) always applies with $ T = \pi $: the translates of $ U $ are the level sets of $ \pi $.

### Complements and Bases

{: .prompt-info }
> *Complement $\Leftrightarrow$ isomorphism*
>
> Suppose $ U $ and $ W $ are subspaces of $ V $. Then
>
> $$ V = U \oplus W \Leftrightarrow \pi|_W : W \to V/U \text{ is an isomorphism.} $$

{: .prompt-proof }
> - Restricting $ \pi $ to $ W $ gives $ \operatorname{null}(\pi\|_W) = W \cap \operatorname{null} \pi = U \cap W $, so $ \pi\|_W $ is injective if and only if $ U \cap W = \\{0\\} $.
> - $ v + U \in \operatorname{range}(\pi\|_W) $ if and only if $ v = w + u $ for some $ w \in W $, $ u \in U $, so $ \pi\|_W $ is surjective if and only if $ U + W = V $. $\blacksquare$

Geometrically, $ W $ is a complement of $ U $ exactly when it crosses every translate of $ U $ exactly once. The crossing point is the one vector of $ W $ that $ \pi $ sends to that translate. In the example, $ W = \operatorname{span}((1,0)) $ crosses the line $ a - b = c $ at $ (c, 0) $.

![Two complements crossing each translate once](/assets/img/math/complements_cross_each_translate_once.png){: w="400" h="270" }

The isomorphism carries bases in both directions: a basis of a complement projects to a basis of $ V/U $, and a basis of $ V/U $ lifts to a basis of a complement.

{: .prompt-tip }
> *Project*
>
> Suppose $ V = U \oplus W $ and $ w_1, \dots, w_m $ is a basis of $ W $. Then $ w_1 + U, \dots, w_m + U $ is a basis of $ V/U $, because $ \pi\|_W $ is an isomorphism.

{: .prompt-tip }
> *Lift*
>
> Suppose $ U $ is a subspace of $ V $ and $ v_1 + U, \dots, v_m + U $ is a basis of $ V/U $. Let $ W = \operatorname{span}(v_1, \dots, v_m) $. Then
>
> - $ V = U \oplus W $ and $ \dim W = \dim V/U $;
> - $ v_1, \dots, v_m $ is a basis of $ W $.
>
> In particular, if $ V/U $ is finite-dimensional, then $ U $ has a finite-dimensional complement, even when $ V $ is not finite-dimensional.

{: .prompt-proof }
> Every vector of $ W $ has the form $ w = \sum_k a_k v_k $, and because the quotient map $ \pi $ is linear,
>
> $$ \pi(w) = \sum_k a_k (v_k + U). $$
>
> - **$ \pi\|_W $ is injective.** If $ \pi(w) = 0 $, then all $ a_k = 0 $, because $ v_1 + U, \dots, v_m + U $ is linearly independent. So $ w = 0 $.
> - **$ \pi\|_W $ is surjective.** Every element of $ V/U $ is a linear combination $ \sum_k a_k (v_k + U) $, because $ v_1 + U, \dots, v_m + U $ spans $ V/U $. That is $ \pi(w) $ for $ w = \sum_k a_k v_k \in W $.
>
> So $ \pi\|_W $ is an isomorphism, and both claims follow.
>
> - By [Complement $\Leftrightarrow$ isomorphism](#complements-and-bases), $ V = U \oplus W $. Isomorphic spaces have the same dimension, so $ \dim W = \dim V/U $.
> - The list $ v_1, \dots, v_m $ spans $ W $ by definition. It is linearly independent: if $ \sum_k a_k v_k = 0 $, then applying $ \pi $ gives $ \sum_k a_k (v_k + U) = 0 $, so all $ a_k = 0 $. $\blacksquare$

Projecting involves no choice, because $ \pi $ is determined by $ U $ alone. Lifting does, because each $ v_k $ can be any point of its translate. In the figure, $ V/U $ is 1-dimensional with basis $ v + U $. Lifting it to $ w $ gives the complement $ W = \operatorname{span}(w) $, and lifting it to $ w' $ gives a different complement $ W' = \operatorname{span}(w') $.

## Quotient Operator

{: .prompt-info }
> *Quotient operator*
>
> Suppose $ T \in \mathcal{L}(V) $ and $ U $ is a subspace of $ V $ invariant under $ T $. The **quotient operator** $ T/U \in \mathcal{L}(V/U) $ is defined by
>
> $$ (T/U)(v + U) = Tv + U $$
>
> for each $ v \in V $.

The formula uses a representative $ v $ of the translate, so the result must not depend on which representative is used. This is where invariance is needed.

{: .prompt-proof }
> Suppose $ v + U = w + U $. Then $ v - w \in U $, and because $ U $ is invariant under $ T $,
>
> $$ Tv - Tw = T(v - w) \in U, $$
>
> so $ Tv + U = Tw + U $. Linearity of $ T/U $ follows from the linearity of $ T $. $\blacksquare$

Geometrically, invariance means that $ T $ sends all the vectors of one translate into a single translate:

$$ x \in v + U \implies Tx \in Tv + U. $$

![The quotient operator moves translates to translates](/assets/img/math/quotient_operator.png){: w="400" h="270" }

Invariance is a statement about $ U $ itself, the translate through the origin: in the figure, $ u \in U $ is sent to $ Tu \in U $. The other translates need not be mapped to themselves. The blue translate $ v + U $ is moved to a different one, the orange translate $ Tv + U $, but it is moved as a whole: $ v $ and $ x $ start on the same translate, and $ Tv $ and $ Tx $ land on the same translate. The operator $ T/U $ records only which translate goes to which, shown on the bottom line, where each translate is a single point. In terms of the quotient map, $ (T/U) \circ \pi = \pi \circ T $: applying $ T $ and then reading off the translate gives the same result as reading off the translate and then applying $ T/U $.

### Block Matrix

A basis of $ U $ followed by a lifted basis of $ V/U $ splits the matrix of $ T $ into the matrices of $ T\vert_U $ and $ T/U $.

{: .prompt-info }
> *Block upper-triangular form*
>
> Suppose $ V $ is finite-dimensional, $ T \in \mathcal{L}(V) $, and $ U $ is a subspace of $ V $ invariant under $ T $. Let $ u_1, \dots, u_n $ be a basis of $ U $ and $ v_1 + U, \dots, v_m + U $ a basis of $ V/U $. Then $ u_1, \dots, u_n, v_1, \dots, v_m $ is a basis of $ V $, and with respect to it
>
> $$ \mathcal{M}(T) = \begin{pmatrix} \mathcal{M}(T\vert_U) & \ast \\ 0 & \mathcal{M}(T/U) \end{pmatrix}, $$
>
> where the diagonal blocks are taken with respect to the two given bases.

{: .prompt-proof }
> By [Lift](#complements-and-bases), $ V = U \oplus \operatorname{span}(v_1, \dots, v_m) $ and $ v_1, \dots, v_m $ is a basis of the second summand, so the combined list is a basis of $ V $.
>
> - **Left columns.** Each $ Tu_j $ lies in $ U $, so it is a combination of $ u_1, \dots, u_n $ only. Its coefficients form column $ j $ of $ \mathcal{M}(T\vert_U) $, and the entries below are $ 0 $.
> - **Right columns.** Let $ B = \mathcal{M}(T/U) $, so $ Tv_k + U = \sum_i B_{ik} (v_i + U) $. Then $ Tv_k - \sum_i B_{ik} v_i \in U $, so
>
>   $$ Tv_k = \sum_i B_{ik} v_i + (\text{a vector of } U). $$
>
>   The coefficients of the $ v_i $ form column $ k $ of $ B $, and the vector of $ U $ gives the entries above it. $\blacksquare$

The diagonal blocks depend only on the bases of $ U $ and $ V/U $. The block $ \ast $ also depends on the representatives $ v_k $, that is, on which complement the basis of $ V/U $ is lifted to.

### Example

Let $ V = \mathbb{R}^2 $ and $ U = \operatorname{span}(u) $ with $ u = (1,1) $, as before, and let

$$ T(a, b) = (2a + b, \; a + 2b). $$

![The quotient operator in the example](/assets/img/math/quotient_operator_example.png){: w="400" h="270" }

**Invariance.** $ Tu = (3,3) = 3u $. Every vector of $ U $ is a multiple $ tu $, and $ T(tu) = 3tu \in U $, so $ U $ is invariant under $ T $. The operator $ T\vert_U $ is multiplication by $ 3 $, so with respect to the basis $ u $ of $ U $,

$$ \mathcal{M}(T\vert_U) = \begin{pmatrix} 3 \end{pmatrix}. $$

**Quotient operator.** The translates of $ U $ are the lines $ a - b = c $, so the translate containing $ (a, b) $ is identified by the number $ c = a - b $. For the vector $ T(a, b) $ this number is

$$ (2a + b) - (a + 2b) = a - b, $$

the same as for $ (a, b) $. So $ T(a, b) $ lies on the same translate as $ (a, b) $, which means

$$ (T/U)\big((a, b) + U\big) = T(a, b) + U = (a, b) + U. $$

So $ T/U $ is the identity operator on $ V/U $. In the figure, $ T $ moves vectors along their own line and never off it.

The space $ V/U $ has dimension $ \dim V - \dim U = 1 $. Let $ w = (1,0) $. Then $ w \notin U $, so $ w + U $ is a nonzero element of $ V/U $ and therefore a basis of it. With respect to this basis,

$$ \mathcal{M}(T/U) = \begin{pmatrix} 1 \end{pmatrix}. $$

**Block matrix.** Lift the basis $ w + U $ to the representative $ w $. Then $ u, w $ is a basis of $ V $, and each column of $ \mathcal{M}(T) $ holds the coefficients of $ Tu $ or $ Tw $ in this basis:

$$ Tu = 3u + 0w, \qquad Tw = (2,1) = (1,1) + (1,0) = 1u + 1w, \qquad \mathcal{M}(T) = \begin{pmatrix} 3 & 1 \\ 0 & 1 \end{pmatrix}. $$

Now lift the same basis element to a different representative, $ x = (0.5, -0.5) $. It lies on the same translate, because $ x - w = (-0.5, -0.5) \in U $, so $ x + U = w + U $. This time $ Tx = (1 - 0.5, \; 0.5 - 1) = x $, so in the basis $ u, x $ of $ V $,

$$ Tu = 3u + 0x, \qquad Tx = 0u + 1x, \qquad \mathcal{M}(T) = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}. $$

In both matrices the diagonal entries are $ \mathcal{M}(T\vert_U) = 3 $ and $ \mathcal{M}(T/U) = 1 $. Only the top right entry changes with the lift: it records how far $ T $ moves the chosen representative along $ U $, which is $ u $ for $ w $ and $ 0 $ for $ x $.

### Eigenvalues

{: .prompt-info }
> *Eigenvalues of $ T $*
>
> Suppose $ V $ is finite-dimensional, $ T \in \mathcal{L}(V) $, and $ U $ is a subspace of $ V $ invariant under $ T $. Then $ \lambda $ is an eigenvalue of $ T $ if and only if it is an eigenvalue of $ T\vert_U $ or of $ T/U $.

{: .prompt-proof }
> Let $ S = T - \lambda I $. Then $ U $ is invariant under $ S $, with $ S\vert_U = T\vert_U - \lambda I $ and $ S/U = T/U - \lambda I $. An operator on a finite-dimensional space has eigenvalue $ \lambda $ exactly when subtracting $ \lambda I $ makes it not injective, so the claim is:
>
> $$ S \text{ is injective} \iff S\vert_U \text{ and } S/U \text{ are both injective.} $$
>
> ($\Leftarrow$) Suppose $ Sv = 0 $. Then $ (S/U)(v + U) = Sv + U = U $, the zero vector of $ V/U $, so $ v + U = U $ because $ S/U $ is injective. That is, $ v \in U $. Then $ S\vert_U v = 0 $, so $ v = 0 $ because $ S\vert_U $ is injective.
>
> ($\Rightarrow$) Suppose $ S $ is injective. Then its restriction $ S\vert_U $ is injective, and since $ U $ is finite-dimensional, $ S\vert_U $ is also surjective onto $ U $. Now suppose $ (S/U)(v + U) = U $, that is, $ Sv \in U $. By surjectivity of $ S\vert_U $, $ Sv = Su $ for some $ u \in U $. Then $ S(v - u) = 0 $, so $ v = u \in U $ and $ v + U = U $. So $ S/U $ is injective. $\blacksquare$

In matrix terms, $ \mathcal{M}(T - \lambda I) $ is block upper-triangular with diagonal blocks $ \mathcal{M}(T\vert_U) - \lambda I $ and $ \mathcal{M}(T/U) - \lambda I $, and the proof shows that such a matrix is invertible exactly when both diagonal blocks are. In the example, the eigenvalues of $ T $ are $ 3 $, from $ T\vert_U $, and $ 1 $, from $ T/U $.

{: .prompt-tip }
> *Polynomials*
>
> $ q(T/U) = q(T)/U $ for every polynomial $ q $ (see [Polynomials]({% post_url 2026-07-22-polynomials %}#homomorphism-gives-divisibility-injectivity-gives-equality)). So the minimal polynomial of $ T/U $ divides the minimal polynomial of $ T $.

### Induction on Dimension

When $ U \ne \\{0\\} $ and $ V $ is finite-dimensional, $ V/U $ has smaller dimension than $ V $. This gives a standard way to prove a statement about all operators by induction on $ \dim V $:

1. find a nonzero invariant subspace $ U $ on which $ T $ is understood;
2. apply the induction hypothesis to $ T/U $, an operator on a smaller space;
3. reassemble the two with the block matrix.

The existence of an upper-triangular matrix is the model case.

{: .prompt-info }
> *Upper-triangular matrix*
>
> Suppose $ V $ is a finite-dimensional complex vector space and $ T \in \mathcal{L}(V) $. Then $ T $ has an upper-triangular matrix with respect to some basis of $ V $.

{: .prompt-proof }
> Induction on $ n = \dim V $. If $ n = 1 $, every $ 1 \times 1 $ matrix is upper-triangular. Suppose $ n > 1 $ and the result holds for all complex vector spaces of dimension less than $ n $.
>
> 1. Over $ \mathbb{C} $, $ T $ has an eigenvalue $ \lambda $. Let $ u $ be a corresponding eigenvector and $ U = \operatorname{span}(u) $. Then $ U $ is invariant under $ T $, and $ \mathcal{M}(T\vert_U) = \begin{pmatrix} \lambda \end{pmatrix} $ with respect to the basis $ u $.
> 2. $ \dim V/U = n - 1 $, so by the induction hypothesis there is a basis $ v_1 + U, \dots, v_{n-1} + U $ of $ V/U $ with respect to which $ \mathcal{M}(T/U) $ is upper-triangular.
> 3. By [Block upper-triangular form](#block-matrix), $ u, v_1, \dots, v_{n-1} $ is a basis of $ V $, and with respect to it
>
>    $$ \mathcal{M}(T) = \begin{pmatrix} \lambda & \ast \\ 0 & \mathcal{M}(T/U) \end{pmatrix}. $$
>
>    The entries below the diagonal are those of the $ 0 $ block and those below the diagonal of $ \mathcal{M}(T/U) $, and all of them are $ 0 $. So $ \mathcal{M}(T) $ is upper-triangular. $\blacksquare$

The proof never needs to know the block $ \ast $, so it does not matter which representatives $ v_k $ are chosen. The quotient is what makes step 2 possible: a complement $ W $ of $ U $ need not be invariant under $ T $, so $ T $ does not restrict to an operator on $ W $, but it always induces an operator on $ V/U $.
