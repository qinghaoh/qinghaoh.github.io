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
> Keep everything about a vector space except the knowledge of which point is $ 0 $. A combination $ \sum_i \lambda_i v_i $ of points still gives the same result from every origin in exactly two cases:
>
> - the coefficients sum to **1**, and the result is a **point**. This is called an **affine combination**;
> - the coefficients sum to **0**, and the result is a **displacement** (a vector).

{: .prompt-warning }
> *Point* and *displacement* are roles, not different kinds of object. Both are elements of $ V $.
>
> - A **point** is a location. If the whole picture is translated by $ t $, it moves by $ t $.
> - A **displacement** is a difference between two locations. If the whole picture is translated, it does not change.
>
> Fixing an origin identifies each point $ p $ with the displacement $ p - 0 $, which is why a vector space does not distinguish the two. Forgetting the origin separates them again.

### What is lost

Sums and scalar multiples of points. Measured from an origin $ o $, the points $ v $ and $ w $ are the arrows $ v - o $ and $ w - o $. Adding them gives another arrow, $ (v - o) + (w - o) $, and the "sum" of the two points is where that arrow lands when it starts at $ o $:

$$ o + (v - o) + (w - o) = v + w - o. $$

The leading $ o $ is what turns the arrow back into a point. Without it the expression is a displacement, not a location. In the picture, this point is the fourth corner of the parallelogram on $ o, v, w $.

Move the origin and the answer moves, even though $ v $ and $ w $ stay put. So "the sum of two points" is not a property of the points alone.

![The sum of two points depends on the origin](/assets/img/math/sum_depends_on_origin.png){: w="400" h="260" }

### What survives

Measured from an origin $ o $, each point $ v_i $ is the vector $ v_i - o $, and the combination with coefficients $ \lambda_i $ is the vector

$$ \sum_i \lambda_i (v_i - o) = \sum_i \lambda_i v_i - \Big(\sum_i \lambda_i\Big)\, o. $$

There are two ways for $ o $ to drop out.

- **Coefficients sum to 0.** The vector itself does not involve $ o $. It is a displacement, the same from every origin. The basic example is the difference $ w - v $.
- **Coefficients sum to 1.** The vector still involves $ o $, but the point it leads to from $ o $ does not:

  $$ o + \sum_i \lambda_i (v_i - o) = \sum_i \lambda_i v_i + \Big(1 - \sum_i \lambda_i\Big)\, o = \sum_i \lambda_i v_i. $$

  Examples are the midpoint $ \tfrac12 v + \tfrac12 w $ and the point $ 2w - v $.

For any other sum, $ o $ remains in both readings, so the result depends on the origin. That is the whole list.

The midpoint, built the same way as the sum above: from each origin, halve the arrows to $ v $ and $ w $ (bold), then complete the parallelogram (dashed). The two constructions are different, but they land on the same point.

![The midpoint does not depend on the origin](/assets/img/math/midpoint_ignores_origin.png){: w="400" h="238" }

The point $ 2w - v $: from each origin, double the arrow to $ w $ (bold), then subtract the arrow to $ v $ (dashed). Again the two paths differ and the endpoint does not.

![The point 2w - v does not depend on the origin](/assets/img/math/extrapolation_ignores_origin.png){: w="400" h="252" }

The coefficient sum also says how the two kinds of result combine:

- point − point = displacement ($ 1 - 1 = 0 $);
- point + displacement = point ($ 1 + 0 = 1 $), from any base point: $ u + (w - v) $ is a point for all points $ u, v, w $, not only when $ u = v $;
- displacement + displacement = displacement ($ 0 + 0 = 0 $);
- point + point is neither ($ 1 + 1 = 2 $).

| Expression                  | Coefficient sum | Result                |
| --------------------------- | --------------- | --------------------- |
| $ w - v $                   | 0               | displacement          |
| $ \tfrac12 v + \tfrac12 w $ | 1               | point                 |
| $ 2w - v $                  | 1               | point                 |
| $ u + w - v $               | 1               | point                 |
| $ v + w $                   | 2               | depends on the origin |
| $ 2v $                      | 2               | depends on the origin |

{: .prompt-tip }
> *In terms of translations*
>
> The figures keep the points fixed and move the origin. The opposite view keeps the origin fixed and translates every point by $ t $:
>
> $$ \sum_i \lambda_i (v_i + t) = \sum_i \lambda_i v_i + \Big(\sum_i \lambda_i\Big)\, t. $$
>
> So the result moves by $ \big(\sum_i \lambda_i\big)\, t $.
>
> - Sum 1: the result moves by $ t $, as a point does. For example, the midpoint of the translated points is the translated midpoint. Translating and then combining gives the same answer as combining and then translating, which is what "combining *commutes* with translation" means.
> - Sum 0: the result does not move, as a displacement does not.
> - Any other sum: the result moves by some other multiple of $ t $, so it is neither. For example $ v + w $ moves by $ 2t $.
>
> Both views say the same thing: an affine combination keeps its position *relative to the points*. In the figures the points stay still, so it stays still. Here the points move by $ t $, so it moves by $ t $.

Clock times are an everyday affine line. "3 pm + 5 pm" is meaningless, because it depends on what is called time zero. But "5 pm − 3 pm = 2 hours" and "the midpoint of 3 pm and 5 pm is 4 pm" are fine. Times are points; durations are the vectors underneath.

## Affine subspace

{: .prompt-tip }
> *Affine subspace* (closed under lines):
>
> A nonempty subset $ A $ of $ V $ such that for any two points of $ A $, the entire line through them lies in $ A $:
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
> ($\Rightarrow$) Suppose $A$ is an affine subspace: $A \neq \emptyset$ and $\lambda v + (1-\lambda)w \in A$ for all $v, w \in A$, $\lambda \in \mathbf{F}$. Fix $p \in A$ and define $U := A - p$. We show $U$ is a subspace; then $A = p + U$.
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

{: .prompt-warning }
> Two hypotheses are doing real work here.
>
> - **Nonempty.** The empty set is closed under lines but is not a translate of any subspace.
> - **$ 2 \neq 0 $ in $ \mathbf{F} $.** The addition step uses $ \lambda = \tfrac12 $. This holds for $ \mathbf{R} $ and $ \mathbf{C} $.

In the language of [Quotient Spaces]({% post_url 2026-07-04-quotient-spaces %}), the affine subspaces of $ V $ are exactly the cosets $ x + U $ of its subspaces.

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
> The inner combination $b$ has coefficients $ \tfrac{\lambda_i}{s} $ summing to $ \tfrac{s}{s} = 1 $, so by induction $ b \in A $. Then $ s\,b + \lambda_k a_k $ is an affine combination of the two points $ b, a_k \in A $ (coefficients $ s + \lambda_k = 1 $), so it lies in $ A $ by the line condition. $\blacksquare$

### Direction and dimension

{: .prompt-info }
> *The direction does not depend on the base point*
>
> Suppose $ A = x + U $ for some $ x \in V $ and some subspace $ U $ of $ V $. Then
>
> $$ U = \{ a - b : a, b \in A \}. $$
>
> So $ U $ is determined by $ A $ alone, and $ A = a + U $ for every $ a \in A $.

{: .prompt-proof }
> If $ a = x + u_1 $ and $ b = x + u_2 $ with $ u_1, u_2 \in U $, then $ a - b = u_1 - u_2 \in U $. Conversely, each $ u \in U $ equals $ (x + u) - x $, a difference of two points of $ A $.
>
> For the last claim, let $ a = x + u_0 $ with $ u_0 \in U $. Then $ a + U = x + (u_0 + U) = x + U $. $\blacksquare$

{: .prompt-tip }
> Suppose $ A = x + U $ is an affine subspace of $ V $.
>
> - The **direction** of $ A $ is the subspace $ U $.
> - The **dimension** of $ A $ is $ \dim U $.
> - Two affine subspaces are **parallel** if they have the same direction.
>
> Affine subspaces of dimension 0, 1 and 2 are *points*, *lines* and *planes*.

{: .prompt-info }
> *Hyperplane*
>
> Suppose $ V $ is finite-dimensional. A **hyperplane** is an affine subspace of dimension $ \dim V - 1 $. A subset $ H $ of $ V $ is a hyperplane if and only if
>
> $$ H = \{ v \in V : \varphi(v) = c \} $$
>
> for some nonzero linear functional $ \varphi $ on $ V $ and some $ c \in \mathbf{F} $.

{: .prompt-proof }
> ($\Leftarrow$) A nonzero linear functional is surjective, so $ \varphi(x_0) = c $ for some $ x_0 \in V $, and then $ H = x_0 + \operatorname{null} \varphi $. By the fundamental theorem of linear maps, $ \dim \operatorname{null} \varphi = \dim V - 1 $.
>
> ($\Rightarrow$) Suppose $ H = x + U $ with $ \dim U = \dim V - 1 $. Extend a basis of $ U $ by one vector $ w $ to a basis of $ V $, and define $ \varphi(u + \lambda w) = \lambda $ for $ u \in U $, $ \lambda \in \mathbf{F} $. Then $ \varphi \neq 0 $ and $ \operatorname{null} \varphi = U $, so $ H = \\{ v \in V : \varphi(v) = \varphi(x) \\} $. $\blacksquare$

"Hyper" does not mean "high-dimensional": a hyperplane is one dimension short of the whole space. It is a line in $ \mathbf{R}^2 $ and a plane in $ \mathbf{R}^3 $. Each nonzero equation of a [system of linear equations]({% post_url 2026-06-30-systems-of-linear-equation %}) cuts out a hyperplane, and the solution set is their intersection.

### Example

Let $ V = \mathbf{R}^2 $, $ U = \operatorname{span}((2,1)) $ and $ A = (0,1) + U $, a line that misses the origin. Take the points $ v = (-1, 0.5) $ and $ w = (1, 1.5) $ of $ A $.

![An affine subspace and its direction](/assets/img/math/affine_subspace_and_direction.png){: w="400" h="260" }

- **Direction.** $ w - v = (2, 1) \in U $. The line $ A $ is parallel to $ U $ and has dimension 1.
- **Coefficients sum to 1.** $ 2w - v = (3, 2.5) $ stays on $ A $.
- **Coefficients do not sum to 1.** $ v + w = (0, 2) $ leaves $ A $. Adding two points of $ A $ depends on where the origin is; an affine combination does not.

### Affine maps

{: .prompt-tip }
> *affine map* = linear map plus a translation
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
> ($\Leftarrow$) Let $ b := f(0) $ and $ Tv := f(v) - b $. We show $ T $ is linear.
>
> *Scalar multiplication:* $ \lambda v = \lambda v + (1 - \lambda) 0 $ is an affine combination, so
>
> $$ T(\lambda v) = \lambda f(v) + (1 - \lambda) b - b = \lambda (f(v) - b) = \lambda\, Tv. $$
>
> *Addition:* $ v + w = v + w - 0 $ is an affine combination with coefficients $ 1, 1, -1 $, so
>
> $$ T(v + w) = f(v) + f(w) - b - b = Tv + Tw. \quad \blacksquare $$

{: .prompt-tip }
> Affine maps send affine subspaces to affine subspaces, and parallel ones to parallel ones:
>
> $$ f(x + U) = f(x) + T(U). $$

## Affine hull

{: .prompt-tip }
> *Affine hull* (closed under *affine combinations*):
>
> The affine hull of $ v_1, \dots, v_m \in V $ is the set of all their affine combinations:
>
> $$ A = \{ \lambda_1 v_1 + \dots + \lambda_m v_m : \lambda_1, \dots, \lambda_m \in \mathbf{F} \ \text{and} \ \lambda_1 + \dots + \lambda_m = 1 \}. $$

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

{: .prompt-info }
> The affine hull of $ v_1, \dots, v_m $ is the smallest affine subspace containing $ v_1, \dots, v_m $.

{: .prompt-proof }
> The affine hull contains each $ v_i $ (take $ \lambda_i = 1 $ and the other coefficients $ 0 $), and it is an affine subspace by the corollary. Conversely, an affine subspace containing $ v_1, \dots, v_m $ contains all their affine combinations, because affine subspaces are closed under finite affine combinations. $\blacksquare$

## Hull

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

is called a **closure operator**. Every hull in the table is one.
