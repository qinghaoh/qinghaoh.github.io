---
title:  "Affine Geometry"
category: [math, "linear algebra"]
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Notation

{% include notation-table.md keys="F V LVW" %}

## Forgetting the origin

{: .prompt-tip }
> *Affine* = linear structure with the origin forgotten
>
> The objects of affine geometry are still the elements of $ V $, that is, vectors. What changes is which constructions are allowed: only those whose result does not depend on which vector is called $ 0 $.

There is a simple test for that. Moving the origin by $ -t $ has the same effect as leaving the origin alone and translating every vector by $ t $. So translate the whole picture and watch what the result does.

### Translating the picture

Take a combination $ \sum_i \lambda_i v_i $ and translate every $ v_i $ by the same $ t $:

$$ \sum_i \lambda_i (v_i + t) = \sum_i \lambda_i v_i + \Big(\sum_i \lambda_i\Big)\, t. $$

So the result moves by $ \big(\sum_i \lambda_i\big)\, t $. There are three cases.

- **Sum 1.** The result moves by $ t $, exactly as the $ v_i $ do, so it keeps its position relative to them. For example, the midpoint of the translated vectors is the translated midpoint. Translating and then combining gives the same answer as combining and then translating, which is what "combining *commutes* with translation" means.
- **Sum 0.** The result does not move at all. For example $ (w + t) - (v + t) = w - v $.
- **Any other sum.** The result moves by some other multiple of $ t $, so it neither follows the picture nor stays fixed. For example $ v + w $ moves by $ 2t $.

### Points and displacements

The first two cases are the two ways an element of $ V $ can be used.

{: .prompt-info }
> *Point and displacement*
>
> - A **point** is a vector used as a location. When the picture is translated by $ t $, it moves by $ t $. The $ v_i $ above are points, and so is every combination of points whose coefficients sum to **1**. Such a combination is called an **affine combination**.
> - A **displacement** is a vector used as a difference between two locations. When the picture is translated, it does not change. Every combination of points whose coefficients sum to **0** is a displacement, and so is $ t $ itself.
>
> A combination of points with any other coefficient sum is neither. It has no meaning once the origin is forgotten.

{: .prompt-warning }
> *Point* and *displacement* are roles, not different kinds of object. Both are elements of $ V $.
>
> Fixing an origin identifies each point $ p $ with the displacement $ p - 0 $, which is why a vector space does not distinguish the two. Forgetting the origin separates them again.

The coefficient sum also says how the two roles combine:

- point − point = displacement ($ 1 - 1 = 0 $);
- point + displacement = point ($ 1 + 0 = 1 $), from any base point: $ u + (w - v) $ is a point for all points $ u, v, w $, not only when $ u = v $;
- displacement + displacement = displacement ($ 0 + 0 = 0 $);
- point + point is neither ($ 1 + 1 = 2 $).

Clock times are an everyday affine line. "3 pm + 5 pm" is meaningless, because it depends on what is called time zero. But "5 pm − 3 pm = 2 hours" and "the midpoint of 3 pm and 5 pm is 4 pm" are fine. Times are points; durations are displacements.

### Moving the origin instead

The same classification appears when the points stay fixed and the origin moves. Measured from an origin $ o $, each point $ v_i $ is the displacement $ v_i - o $, and the combination with coefficients $ \lambda_i $ is the displacement

$$ \sum_i \lambda_i (v_i - o) = \sum_i \lambda_i v_i - \Big(\sum_i \lambda_i\Big)\, o. $$

- **Sum 0.** The result does not involve $ o $: it is the same displacement from every origin.
- **Sum 1.** The displacement still involves $ o $, but the point it leads to from $ o $ does not:

  $$ o + \sum_i \lambda_i (v_i - o) = \sum_i \lambda_i v_i + \Big(1 - \sum_i \lambda_i\Big)\, o = \sum_i \lambda_i v_i. $$

- **Any other sum.** $ o $ remains in both readings, so the result depends on the origin.

**The sum of two points depends on the origin.** Measured from $ o $, the points $ v $ and $ w $ are the displacements $ v - o $ and $ w - o $. Adding them gives another displacement, $ (v - o) + (w - o) $, and the "sum" of the two points is the point that displacement leads to from $ o $:

$$ o + (v - o) + (w - o) = v + w - o. $$

The leading $ o $ is what turns the displacement back into a point. Without it the expression is only a displacement, not a location. In the picture, this point is the fourth corner of the parallelogram on $ o, v, w $. Move the origin and the answer moves, even though $ v $ and $ w $ stay put. The same happens to a scalar multiple: $ 2v $ becomes $ o + 2(v - o) = 2v - o $.

![The sum of two points depends on the origin](/assets/img/math/sum_depends_on_origin.png){: w="400" h="260" }

**The midpoint does not.** Build $ \tfrac12 v + \tfrac12 w $ the same way: from each origin, halve the displacements to $ v $ and $ w $ (bold), then complete the parallelogram (dashed). The two constructions are different, but they land on the same point.

![The midpoint does not depend on the origin](/assets/img/math/midpoint_ignores_origin.png){: w="400" h="238" }

**Neither does $ 2w - v $.** From each origin, double the displacement to $ w $ (bold), then subtract the displacement to $ v $ (dashed). Again the two paths differ and the endpoint does not.

![The point 2w - v does not depend on the origin](/assets/img/math/extrapolation_ignores_origin.png){: w="400" h="252" }

Both views say the same thing: an affine combination keeps its position *relative to the points*. In these figures the points stay still, so it stays still. Under a translation the points move by $ t $, so it moves by $ t $.

## Affine subspace

{: .prompt-tip }
> *Affine subspace* (closed under lines):
>
> A nonempty subset $ A $ of $ V $ such that for any two points of $ A $, the entire line through them lies in $ A $. That is, $ A $ contains every affine combination of two of its points:
>
> $$ \lambda v + (1 - \lambda)w \in A \quad \text{for all } v, w \in A \text{ and all } \lambda \in \mathbf{F}. $$

{: .prompt-info }
> *Affine subspaces are exactly translates of linear subspaces*
>
> A subset $ A $ of $ V $ is an affine subspace if and only if $ A = x + U $ for some $ x \in V $ and some subspace $ U $ of $ V $.

{: .prompt-proof }
> ($\Leftarrow$) Suppose $A = x + U$ for a subspace $U$. Then $A$ is nonempty, since $x \in A$. For $v, w \in A$, write $v = x + u_1$, $w = x + u_2$ with $u_1, u_2 \in U$. Then for any $\lambda \in \mathbf{F}$,
>
> $$\lambda v + (1-\lambda)w = x + \big(\lambda u_1 + (1-\lambda)u_2\big) \in x + U = A,$$
>
> since $\lambda u_1 + (1-\lambda)u_2 \in U$.
>
> ($\Rightarrow$) Suppose $A$ is an affine subspace: $A \neq \emptyset$ and $\lambda v + (1-\lambda)w \in A$ for all $v, w \in A$, $\lambda \in \mathbf{F}$. Fix $p \in A$ and let $U$ be the set of displacements from $p$ to points of $A$:
>
> $$U := \{ u \in V : p + u \in A \}.$$
>
> Then $A = p + U$ by definition, so it remains to show that $U$ is a subspace.
>
> *Zero:* $p + 0 = p \in A$, so $0 \in U$.
>
> *Scalar multiplication:* Let $u \in U$ and $\lambda \in \mathbf{F}$. The points $p + u$ and $p$ are in $A$, so every point of the line through them is in $A$:
>
> $$p + \lambda u = \lambda(p + u) + (1-\lambda)p \in A.$$
>
> Hence $\lambda u \in U$.
>
> *Addition:* Let $u_1, u_2 \in U$. The points $p + u_1$ and $p + u_2$ are in $A$, so their midpoint is in $A$:
>
> $$p + \tfrac12(u_1 + u_2) = \tfrac12(p + u_1) + \tfrac12(p + u_2) \in A.$$
>
> Hence $\tfrac12(u_1 + u_2) \in U$, and doubling it with the scalar step gives $u_1 + u_2 \in U$. $\blacksquare$

{: .prompt-warning }
> *Nonempty* is doing real work here: the empty set is closed under lines but is not a translate of any subspace.

{: .prompt-info }
> *An affine subspace is closed under every finite affine combination* ("Closed under lines" is the $ k = 2 $ case).
>
> If $ a_1, \dots, a_k \in A $ and $ \sum \lambda_i = 1 $, then $ \sum \lambda_i a_i \in A $.

{: .prompt-proof }
> *Proof by induction on $k$.*
>
> For $ k = 1 $, $ \lambda_1 = 1 $ gives $ a_1 \in A $. For the step, given $ \sum_{i=1}^{k}\lambda_i = 1 $ with $ k \ge 2 $, the coefficients cannot all equal 1, because then they would sum to $ k \neq 1 $. Reorder the points so that the last coefficient satisfies $ \lambda_k \neq 1 $. Set $ s = \lambda_1 + \dots + \lambda_{k-1} = 1 - \lambda_k \neq 0 $. Then
>
> $$\sum_{i=1}^k \lambda_i a_i = s\underbrace{\left(\sum_{i=1}^{k-1}\tfrac{\lambda_i}{s} a_i\right)}_{=:b} + \lambda_k a_k.$$
>
> The inner combination $b$ has coefficients $ \tfrac{\lambda_i}{s} $ summing to $ \tfrac{s}{s} = 1 $, so by induction $ b \in A $. Then $ s\,b + \lambda_k a_k $ is an affine combination of the two points $ b, a_k \in A $ (coefficients $ s + \lambda_k = 1 $), so it lies in $ A $ by the line condition. $\blacksquare$

{: .prompt-tip }
> *Geometric picture: a sweep, one dimension at a time*
>
> **Step 1.** Two points $ a_1, a_2 $ give a line.
>
> ![The line through two points](/assets/img/math/sweep_1_line.png){: w="400" h="125" }
>
> **Step 2.** Add a third point $ a_3 $ off that line. The lines from $ a_3 $ to the points of the first line (blue) sweep out a plane. The dashed line is explained in the next box.
>
> ![Lines from a third point to a line sweep out a plane](/assets/img/math/sweep_2_plane.png){: w="400" h="250" }
>
> **Step 3.** Add a fourth point $ a_4 $ off that plane. The lines from $ a_4 $ to the points of the plane sweep out a 3-dimensional affine subspace.
>
> ![Lines from a fourth point to a plane sweep out space](/assets/img/math/sweep_3_space.png){: w="400" h="270" }
>
> So a set that contains the line through any two of its points must also contain the whole plane through any three of them, the whole 3-dimensional affine subspace through any four, and so on.

{: .prompt-tip }
> *What a single sweep misses*
>
> **The plane in step 2 is complete.** The set contains every point of it, including the dashed line. What the dashed line shows is only that the blue lines alone, the ones drawn from $ a_3 $, do not cover the whole plane: a line from $ a_3 $ parallel to the first line never meets the first line, so it is not one of the blue lines.
>
> The points of the dashed line are in the set for a different reason: they lie on lines drawn from $ a_1 $ or $ a_2 $.
>
> ![A point of the dashed line is reached by a line from another corner](/assets/img/math/sweep_2_plane_missed_line.png){: w="400" h="265" }
>
> Here $ p $ is a point of the dashed line. The line from $ a_1 $ through $ p $ crosses the line through $ a_2 $ and $ a_3 $ at a point $ b' $. Both $ a_1 $ and $ b' $ are in the set, so the whole line through them is, and $ p $ is on it.
>
> **In coordinates.** A point of the plane is $ p = \lambda_1 a_1 + \lambda_2 a_2 + \lambda_3 a_3 $ with $ \lambda_1 + \lambda_2 + \lambda_3 = 1 $.
>
> - **Sweeping from $ a_3 $ reaches $ p $ exactly when $ \lambda_3 \neq 1 $.** Then $ p = s\,b + \lambda_3 a_3 $ with $ b $ on the first line and $ s = \lambda_1 + \lambda_2 \neq 0 $.
> - **The dashed line is exactly where $ \lambda_3 = 1 $.** Then $ \lambda_2 = -\lambda_1 $ and $ p = a_3 + \lambda_1 (a_1 - a_2) $: start at $ a_3 $ and move parallel to the first line.
> - **Such a point is reached from another corner.** The three coefficients cannot all equal 1, since they sum to 1. If $ \lambda_1 \neq 1 $, then $ p $ lies on a line from $ a_1 $ to a point $ b' $ of the line through $ a_2 $ and $ a_3 $, as in the figure.
>
> **For any number of points.** Let $ p = \sum_{i=1}^{k} \lambda_i a_i $ with $ \sum_i \lambda_i = 1 $ and suppose $ \lambda_k = 1 $. Then $ \lambda_1 + \dots + \lambda_{k-1} = 0 $, so subtracting $ \big(\sum_{i<k} \lambda_i\big) a_1 = 0 $ changes nothing:
>
> $$ p = a_k + \sum_{i=1}^{k-1} \lambda_i a_i = a_k + \sum_{i=2}^{k-1} \lambda_i (a_i - a_1). $$
>
> Let $ D = \operatorname{span}(a_2 - a_1, \dots, a_{k-1} - a_1) $, the subspace of all displacements built from the first $ k - 1 $ points.
>
> - The first $ k - 1 $ points sweep out $ a_1 + D $: start at $ a_1 $ and move by any displacement in $ D $. (This is proved under [Affine hull](#affine-hull) below.)
> - The formula says $ p \in a_k + D $: start at $ a_k $ and move by a displacement in $ D $. Conversely, every point of $ a_k + D $ has $ \lambda_k = 1 $.
>
> So what the sweep from $ a_k $ misses is a translate of the same subspace $ D $, through $ a_k $ instead of $ a_1 $. That is what *parallel* means: $ a_k + D $ is a copy of $ a_1 + D $ shifted to pass through $ a_k $. For $ k = 4 $ it is the plane through $ a_4 $ parallel to the plane of step 3.
>
> **Back to the proof.** The formula in the proof is a single sweep from $ a_k $: the point $ b $ lies in the affine subspace swept out by the other $ k - 1 $ points, and $ p $ is on the line through $ b $ and $ a_k $. This works exactly when $ \lambda_k \neq 1 $. The step "reorder the points so that $ \lambda_k \neq 1 $" chooses, for each target $ p $ separately, a point to sweep from that can actually reach $ p $. Filling in the dashed line is the same formula with a different point in the last position.

### Direction and dimension

An affine subspace can be written as $ A = x + U $ with many different base points $ x $. The subspace $ U $, however, is the same in every such expression. That is what allows it to be given a name.

{: .prompt-info }
> *The subspace is determined by the affine subspace*
>
> Suppose $ A = x + U $ for some $ x \in V $ and some subspace $ U $ of $ V $. Then
>
> $$ U = \{ a - b : a, b \in A \}. $$
>
> So $ U $ is determined by $ A $ alone, and $ A = a + U $ for every $ a \in A $.

{: .prompt-proof }
> If $ a = x + u_1 $ and $ b = x + u_2 $ with $ u_1, u_2 \in U $, then $ a - b = u_1 - u_2 \in U $. Conversely, each $ u \in U $ equals $ (x + u) - x $, a difference of two points of $ A $.
>
> For the last claim, let $ a = x + u_0 $ with $ u_0 \in U $. Because $ U $ is a subspace containing $ u_0 $, adding $ u_0 $ maps $ U $ onto itself: $ u_0 + u \in U $ for every $ u \in U $, and every $ u \in U $ equals $ u_0 + (u - u_0) $. So $ u_0 + U = U $, and therefore
>
> $$ a + U = x + (u_0 + U) = x + U = A. \quad \blacksquare $$

{: .prompt-tip }
> *Direction, dimension, parallel*
>
> Suppose $ A = x + U $ is an affine subspace of $ V $.
>
> - The **direction** of $ A $ is the subspace $ U $: the set of all displacements between points of $ A $.
> - The **dimension** of $ A $ is $ \dim U $.
> - Two affine subspaces are **parallel** if they have the same direction.

### Example

Let $ V = \mathbb{R}^2 $, $ U = \operatorname{span}((2,1)) $ and $ A = (0,1) + U $, a line that misses the origin. Take the points $ v = (-1, 0.5) $ and $ w = (1, 1.5) $ of $ A $.

![An affine subspace and its direction](/assets/img/math/affine_subspace_and_direction.png){: w="400" h="260" }

- **Direction.** $ w - v = (2, 1) \in U $. The line $ A $ is parallel to $ U $ and has dimension 1.
- **Closed under affine combinations.** $ 2w - v = (3, 2.5) $ stays on $ A $.
- **Not a subspace.** $ v + w = (0, 2) $ leaves $ A $, and $ 0 \notin A $.

### Correspondence with subspaces

The affine subspaces of $ V $ are exactly the translates $ x + U $ of its subspaces, but the correspondence between affine subspaces and subspaces is not one-to-one:

- **$ A $ determines $ U $.** Every affine subspace has exactly one direction.
- **$ U $ does not determine $ A $.** All the translates of $ U $ share the direction $ U $. To recover $ A $ you need $ U $ together with one point of $ A $.

In the language of [Quotient Spaces]({% post_url 2026-07-04-quotient-spaces %}), the affine subspaces with direction $ U $ are the cosets of $ U $, that is, the elements of $ V/U $.

Separately, each single $ A $ is a copy of its direction: for any base point $ x \in A $, the map $ u \mapsto x + u $ is a bijection from $ U $ onto $ A $. It depends on the choice of $ x $, which is the "origin forgotten" again.

### Hyperplanes

{: .prompt-info }
> *Hyperplane*
>
> Suppose $ V $ is finite-dimensional. A **hyperplane** is an affine subspace of dimension $ \dim V - 1 $. A subset $ H $ of $ V $ is a hyperplane if and only if
>
> $$ H = \{ v \in V : \varphi(v) = c \} $$
>
> for some nonzero linear functional $ \varphi $ on $ V $ and some $ c \in \mathbf{F} $.

{: .prompt-proof }
> ($\Leftarrow$) A nonzero linear functional is surjective: pick $ v $ with $ \varphi(v) = a \neq 0 $; then $ \varphi\big(\tfrac{c}{a} v\big) = c $ for every $ c \in \mathbf{F} $. So $ \varphi(x_0) = c $ for some $ x_0 \in V $, and then $ H = x_0 + \operatorname{null} \varphi $. By the fundamental theorem of linear maps, $ \dim \operatorname{null} \varphi = \dim V - 1 $.
>
> ($\Rightarrow$) Suppose $ H = x + U $ with $ \dim U = \dim V - 1 $. Extend a basis of $ U $ by one vector $ w $ to a basis of $ V $. Then every vector of $ V $ can be written in exactly one way as $ u + \lambda w $ with $ u \in U $ and $ \lambda \in \mathbf{F} $, so
>
> $$ \varphi(u + \lambda w) := \lambda $$
>
> is a well-defined linear functional on $ V $. It is nonzero because $ \varphi(w) = 1 $, and $ \operatorname{null} \varphi = U $ because $ \varphi(u + \lambda w) = 0 $ exactly when $ \lambda = 0 $. Therefore, for $ v \in V $,
>
> $$ v \in H \iff v - x \in U \iff \varphi(v - x) = 0 \iff \varphi(v) = \varphi(x), $$
>
> which says $ H = \\{ v \in V : \varphi(v) = c \\} $ with $ c = \varphi(x) $. $\blacksquare$

"Hyper" does not mean "high-dimensional": a hyperplane is one dimension short of the whole space.

- In $ \mathbb{R}^1 $ a hyperplane is a *point* (dimension 0).
- In $ \mathbb{R}^2 $ a hyperplane is a *line* (dimension 1).
- In $ \mathbb{R}^3 $ a hyperplane is a *plane* (dimension 2).

In coordinates, $ \varphi(v) = c $ is a single linear equation $ a_1 x_1 + \dots + a_n x_n = c $. Some coefficient is nonzero, say $ a_n \neq 0 $. Then $ x_1, \dots, x_{n-1} $ can be chosen freely, and the equation determines the last one:

$$ x_n = \frac{c - a_1 x_1 - \dots - a_{n-1} x_{n-1}}{a_n}. $$

So of the $ n $ free directions in $ V $, one equation removes exactly one: $ n - 1 $ coordinates stay free and one is forced. That is why a hyperplane has dimension $ n - 1 $.

Each nonzero equation of a [system of linear equations]({% post_url 2026-06-30-systems-of-linear-equation %}) therefore cuts out a hyperplane, and the solution set is their intersection. Each equation that is independent of the others removes one more direction, so the solution set is either empty or an affine subspace of dimension $ n - r $, where $ r $ is the number of independent equations.

## Affine hull

{: .prompt-tip }
> *Affine hull* (all *affine combinations*):
>
> The affine hull of $ v_1, \dots, v_m \in V $ is the set of all their affine combinations:
>
> $$ A = \{ \lambda_1 v_1 + \dots + \lambda_m v_m : \lambda_1, \dots, \lambda_m \in \mathbf{F} \ \text{and} \ \lambda_1 + \dots + \lambda_m = 1 \}. $$

{: .prompt-tip }
> *Geometric picture*
>
> The affine hull of some points is the smallest affine subspace that passes through all of them. It extends infinitely, beyond the points themselves.
>
> - 2 distinct points: the whole line through them.
> - 3 points not on a line: the whole plane through them.
> - 4 points not in a plane: the whole 3-dimensional affine subspace through them.
>
> It is what the [sweep](#affine-subspace) produces. Start with the points, draw the line through every pair, then the lines through every pair of points you now have, and keep going until nothing new appears. What you end up with is the affine hull.
>
> "Smallest" is relative to the given points. Larger affine subspaces contain them too: for 2 points in $ \mathbb{R}^3 $, every plane through their line does, and so does $ \mathbb{R}^3 $ itself. The affine hull is the one that contains only what the points force, so it lies inside all the others. To get a larger affine subspace, sweep from more points.
>
> Both claims, "affine subspace" and "smallest", are proved below.

{: .prompt-info }
> *An affine hull is a translate of a span*
>
> Suppose $ A $ is the affine hull of $ v_1, \dots, v_m \in V $. Then
>
> $$ A = v_1 + \operatorname{span}(v_2 - v_1,\ v_3 - v_1,\ \dots,\ v_m - v_1). $$

{: .prompt-proof }
> Let $ U := \operatorname{span}(v_2 - v_1,\ \dots,\ v_m - v_1) $.
>
> **$A \subseteq v_1 + U$.** Take $\lambda_1 v_1 + \dots + \lambda_m v_m \in A$ with $\sum_i \lambda_i = 1$. Use $\lambda_1 = 1 - (\lambda_2 + \dots + \lambda_m)$ to eliminate $\lambda_1$:
>
> $$\sum_{i=1}^m \lambda_i v_i = v_1 + \sum_{i=2}^m \lambda_i (v_i - v_1).$$
>
> The sum $ \sum_{i=2}^m \lambda_i (v_i - v_1)$ lies in $U$, so the point is in $v_1 + U$.
>
> **$v_1 + U \subseteq A$.** A general element is $v_1 + \sum_{i=2}^m c_i(v_i - v_1)$. Reversing the computation, this equals $\sum_i \lambda_i v_i$ with $\lambda_i = c_i$ for $i \ge 2$ and $\lambda_1 = 1 - \sum_{i\ge2} c_i$, whose coefficients sum to $1$. So it's in $A$. $\blacksquare$

{: .prompt-tip }
> Corollary: The affine hull of $ v_1, \dots, v_m $ is an affine subspace of dimension at most $ m - 1 $.

{: .prompt-tip }
> *Degenerate cases*
>
> The dimension equals $ m - 1 $ exactly when the displacements $ v_2 - v_1, \dots, v_m - v_1 $ are linearly independent. The points are then called **affinely independent**. Otherwise the affine hull is smaller:
>
> - Three points on one line: the affine hull is that line (dimension 1, not 2).
> - Four points in one plane: the affine hull is that plane (dimension 2, not 3).
> - All points equal: the affine hull is a single point (dimension 0).

{: .prompt-info }
> The affine hull of $ v_1, \dots, v_m $ is the smallest affine subspace containing $ v_1, \dots, v_m $.

{: .prompt-proof }
> The affine hull contains each $ v_i $ (take $ \lambda_i = 1 $ and the other coefficients $ 0 $), and it is an affine subspace by the corollary. Conversely, an affine subspace containing $ v_1, \dots, v_m $ contains all their affine combinations, because affine subspaces are closed under finite affine combinations. $\blacksquare$

### Hulls in general

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

- *extensive* ($S \subseteq \overline{S}$: it encloses),
- *monotone* ($S \subseteq T \Rightarrow \overline{S} \subseteq \overline{T}$), and
- *idempotent* ($\overline{\overline{S}} = \overline{S}$: re-enclosing an already-enclosed set adds nothing),

is called a **closure operator**. Every hull in the table is one.

## Affine maps

{: .prompt-tip }
> *Affine map* = linear map plus a translation
>
> A map $ f : V \to W $ is **affine** if there exist $ T \in \mathcal{L}(V, W) $ and $ b \in W $ such that $ f(x) = Tx + b $ for all $ x \in V $.

{: .prompt-info }
> *Affine maps are exactly the maps that preserve affine combinations*
>
> A map $ f : V \to W $ is affine if and only if
>
> $$ f\Big(\sum_i \lambda_i v_i\Big) = \sum_i \lambda_i f(v_i) \quad \text{whenever } \sum_i \lambda_i = 1. $$

{: .prompt-proof }
> ($\Rightarrow$) If $ f(x) = Tx + b $ and $ \sum_i \lambda_i = 1 $, then
>
> $$ f\Big(\sum_i \lambda_i v_i\Big) = \sum_i \lambda_i T v_i + \Big(\sum_i \lambda_i\Big) b = \sum_i \lambda_i (T v_i + b). $$
>
> ($\Leftarrow$) Suppose $ f $ preserves affine combinations. Let $ b := f(0) $ and define $ T : V \to W $ by $ Tv := f(v) - b $. Then $ f(v) = Tv + b $ by construction, so it remains to show that $ T $ is linear.
>
> *Scalar multiplication:* $ \lambda v $ is an affine combination of the two points $ v $ and $ 0 $, with coefficients $ \lambda $ and $ 1 - \lambda $:
>
> $$ \lambda v = \lambda \cdot v + (1 - \lambda) \cdot 0. $$
>
> Since $ f $ preserves it, $ f(\lambda v) = \lambda f(v) + (1 - \lambda) f(0) = \lambda f(v) + (1 - \lambda) b $. Therefore
>
> $$ T(\lambda v) = f(\lambda v) - b = \lambda f(v) - \lambda b = \lambda (f(v) - b) = \lambda\, Tv. $$
>
> *Addition:* $ v + w $ is an affine combination of the three points $ v $, $ w $ and $ 0 $, with coefficients $ 1, 1, -1 $:
>
> $$ v + w = 1 \cdot v + 1 \cdot w + (-1) \cdot 0. $$
>
> Since $ f $ preserves it, $ f(v + w) = f(v) + f(w) - f(0) = f(v) + f(w) - b $. Therefore
>
> $$ T(v + w) = f(v + w) - b = (f(v) - b) + (f(w) - b) = Tv + Tw. \quad \blacksquare $$

{: .prompt-tip }
> *What $ T $ and $ b $ do*
>
> - **$ T $ acts on displacements, $ f $ acts on points.** For all $ x, y \in V $, the $ b $ cancels: $ f(x) - f(y) = T(x - y) $.
> - **Images.** Apply $ f $ to each point $ x + u $ of an affine subspace $ x + U $: the result is $ f(x + u) = f(x) + Tu $. So the image of $ x + U $ is $ f(x) + T(U) $, where $ T(U) = \\{ Tu : u \in U \\} $ is a subspace of $ W $. That is again an affine subspace, with base point $ f(x) $ and direction $ T(U) $. Parallel affine subspaces share the same $ U $, so their images share the same $ T(U) $ and are parallel again.
> - **Preimages.** The set of points that $ f $ sends to a given point $ y $ is empty or an affine subspace with direction $ \operatorname{null} T $. Indeed, if $ f(x_0) = y $ for some $ x_0 $, then by the first item
>
>   $$ f(x) = y \iff f(x) - f(x_0) = 0 \iff T(x - x_0) = 0 \iff x \in x_0 + \operatorname{null} T. $$

{: .prompt-tip }
> *Systems of linear equations*
>
> A [system of linear equations]({% post_url 2026-06-30-systems-of-linear-equation %}) $ Tx = c $ asks which points the affine map $ f(x) = Tx - c $ sends to $ 0 $. By the last item above, the solution set is empty or an affine subspace with direction $ \operatorname{null} T $: one particular solution plus all solutions of the homogeneous system $ Tx = 0 $. This is the same set that [Hyperplanes](#hyperplanes) describes as an intersection of hyperplanes, one per equation.

## Summary

Everything in this post is built from one notion, the affine combination.

```mermaid
flowchart TD
	nComb["<b>Affine combination</b><br/>Σ λᵢvᵢ with Σ λᵢ = 1<br/>the same from every origin"]
	nSub["<b>Affine subspace</b><br/>x + U"]
	nHull["<b>Affine hull</b><br/>of v₁, …, vₘ"]
	nMap["<b>Affine map</b><br/>f(x) = Tx + b"]
	nComb -->|a set closed under them| nSub
	nComb -->|all of them,<br/>for given points| nHull
	nComb -->|a map that<br/>preserves them| nMap
	nHull -->|is the smallest one<br/>containing the points| nSub
	nMap -->|images and preimages<br/>are again| nSub
```

- **Affine subspace:** a nonempty set closed under affine combinations. Equivalently, a translate $ x + U $ of a subspace.
- **Affine hull:** all affine combinations of given points. It is the smallest affine subspace containing them.
- **Affine map:** a map that preserves affine combinations. Equivalently, $ f(x) = Tx + b $. It sends affine subspaces to affine subspaces, and the set of points it sends to a given point is empty or an affine subspace.
