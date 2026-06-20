---
title:  "Linear Algebra"
category: math
tags: math
mathjax_font: mathjax-pagella
mermaid: true
---

## Vector Space

### Definition

Motivation: properties of addition and scalar multiplication in $ \mathbf{F}^n $

```mermaid
flowchart LR
	subgraph addGroup["$$u + v$$"]
		direction TB
		addSet["abelian group under addition"]
	end
	subgraph scalarGroup["$$\lambda v$$"]
		direction TB
		scalarSet["&bull; associativity<br/>&bull; 1"]
	end
	addSet <-->|distributive<br/>properties| scalarSet
	style addGroup fill:transparent,stroke:transparent
	style scalarGroup fill:transparent,stroke:transparent
```

{: .prompt-tip }
> Examples

* $ \mathbf{F}^S $
* $ \mathcal{P}(\mathbf{F}) $
* $ \mathcal{P}_m(\mathbf{F}) $
* $ \mathcal{L}(V, W) $
* $ \mathbf{F}^{m,n} $ ($\dim = mn $)

### Subspace

{: .prompt-tip }
> Examples

* $ V_1 \cap \ldots \cap V_m $
* $ V_1 \cup V_2 $ ($ \Leftrightarrow V1 \subseteq V2 $ or $ V1 \supseteq V2 $)
* $ V_1 + \ldots + V_m $ (smallest containing subspace)

Finite-dimensional $V$, $ U \subset V $:

* $ \dim U \leq \dim V $
* $ \dim U = \dim V \Leftrightarrow U = V $

### Sum

Analogy between sets and vector spaces:

* Cardinality: Dimension
* Union: Sum
* Union of disjoint sets: Direct sum

### Number of Vectors

The number of vectors equals the number of choices of the coefficient tuple $(a_1, \dots, a_n)$:

$$\#(\text{vectors}) = \lvert \mathbf{F} \rvert ^n$$

Finite fields exist precisely for prime-power sizes $q = p^k$.

|                      | over $\mathbb{R}$ or $\mathbb{C}$ | over $\mathbf{F}_q$ |
| -------------------- | --------------------------------- | ------------------- |
| trivial space        | $1$                               | $1$                 |
| dimension $n \geq 1$ | $\infty$                          | $q^n$               |

## Bases

![Basis](/assets/img/math/basis.png)

A list is a basis if it satisfies any two of the following three conditions:

* It is linearly independent
* It spans $V$
* Its length equals $ \dim(V) $

A direct-sum decomposition of $ V $ is the same thing as a partition of a basis of $ V $.

![basis partition](/assets/img/math/basis_partition_equals_direct_sum.png)

## Linear Maps

$ \forall \ \text{basis} \ v_1, \ldots, v_n \in V $ and $ \forall w_1, \ldots, w_n \in W $, $ \exists! T \in \mathcal{L}(V, W) \ \text{s.t.} $

$$ Tv_k = w_k $$

for each $k = 1, \ldots, n$

{: .prompt-info }
> Extension

Finite-dimensional $V$:

$ U \subseteq V $, $ S \in \mathcal{L}(U, W) \Rightarrow \exists T \in \mathcal{L}(V, W) \ \text{s.t.} \ T \vert _U = S $

Extend a basis $ u_1,\dots,u_m$ of $U $ to a basis $ u_1,\dots,u_m,\,v_1,\dots,v_n $ of $ V $. Define $ T u_i = S u_i $ and $ T v_j = $ anything in $ W $; extend linearly.

### Algebraic Operations

* $ \mathcal{L}(V, W) $ is a vector space
* Product of linear maps is a [bilinear map](https://en.wikipedia.org/wiki/Bilinear_map)

### Null Spaces and Ranges

![linear map](https://upload.wikimedia.org/wikipedia/commons/8/89/Kernel_and_image_of_linear_map.svg){: w="400" h="200" }

{: .prompt-info }
> Fundamental theorem of linear maps

Finite-dimensional $V$, $ T \in \mathcal{L}(V, W) $:

$$ \dim V = \dim \text{null} \, T + \dim \text{range} \, T $$

{: .prompt-tip }
> Generalization

Finite-dimensional $V$, $ T \in \mathcal{L}(V, W) $, $ U \subseteq W $:

$$ \dim \{ v \in V : Tv \in U \} = \dim \text{null} \, T + \dim (U \cap \text{range} \, T) $$

{: .prompt-info }
> Injectivity, Surjectivity and Invertibility

Finite-dimensional $V$, $ T \in \mathcal{L}(V, W) $:

* $ T $ is injective $ \Leftrightarrow \text{null} \, T = {0} $
* $ T $ is surjective $ \Leftrightarrow \text{range} \, T = W $
* $ T $ is invertible $ \Leftrightarrow $ $ T $ is injective and $ T $ is surjective

Finite-dimensional $V$, $W$, $ T \in \mathcal{L}(V, W) $:

* $ T $ is injective $ \Rightarrow \dim V \leq \dim W $
* $ T $ is surjective $ \Rightarrow \dim V \geq \dim W $
* $ T $ is invertible $ \Leftrightarrow $ $ T $ is injective $ \Leftrightarrow $ $ T $ is surjective $ \Leftrightarrow \dim V = \dim W $

Finite-dimensional $W$, $ T \in \mathcal{L}(V, W) $:

* $ T $ is injective $ \Rightarrow \exists S \in \mathcal{L}(W, V) \ \text{s.t.} \ ST = E $
* $ T $ is surjective $ \Rightarrow \exists S \in \mathcal{L}(W, V) \ \text{s.t.} \ TS = E $

{: .prompt-info }
> Product

{: .prompt-tip }
> $ ST $ is a linear map

Finite-dimensional $U$, $V$, $ S \in \mathcal{L}(V, W) $ and $ T \in \mathcal{L}(U, V) $:

* $ \text{null} \, T \subseteq \text{null} \, ST $
* $ \text{range} \, ST \subseteq \text{range} \, S $

Finite-dimensional $U$, $V$, $ S \in \mathcal{L}(V, W) $ and $ T \in \mathcal{L}(U, V) $:

* $ \dim \text{null} \, ST \leq \dim \text{null} \, S + \dim \text{null} \, T $
* $ \dim \text{range} \, ST \leq \min(\dim \text{range} \, S + \dim \text{range} \, T) $

## Matrices

$ \mathcal{M}(T,(v_1,\ldots,v_n),(w_1,\ldots,w_m)) \in \mathbf{F}^{m,n} $ is the matrix *representation* of $ T $ with respect to the chosen bases.

$$ Tv_k = \sum_{j=1}^m {A_{j,k}w_j} $$

{: .prompt-tip }
> $ T \in \mathcal{L}(\mathbf{F}^n, \mathbf{F}^m) $, $ \mathcal{M}(T)_{\cdot,k} = Te_k $

{: .prompt-info }
> Column of matrix product equals matrix times column

![column of matrix product](/assets/img/math/column_of_matrix_product_equals_A_times_column.png)

{: .prompt-info }
> Linear combination of columns

![linear combination of columns](../assets/img/math/Ab_as_linear_combination_of_columns.png)

{: .prompt-info }
> Column–row factorization

![column–row factorization](../assets/img/math/column_row_factorization_A_equals_CR.png)

$ C $ is a *basis* of the column space.

{: .prompt-tip }
> $ A \mapsto A^T $ is a linear map
