---
title:  "System of linear equations"
category: math
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## System of linear equations

{: .prompt-info }
> For $A \in \mathbf{F}^{m\times n}$ and one particular $b \in \mathbf{F}^n$, then
>
> $Ax = b$ has exactly one solution $\iff$ ($ A $ injective) and ($ b \in \operatorname{range} A $).

{: .prompt-info }
> Suppose $ T \in \mathcal{L}(V,W) $ and $ c \in W $, then
>
> $ {x \in V : Tx = c} $ is either the empty set or is a translate of $ \operatorname{null} T $.

{: .prompt-tip }
> Special case: system of linear equations
>
> general solution = particular solution + homogeneous solution
>
> $ V/\operatorname{null} T \cong_\tilde{T} \operatorname{range} T $

{: .prompt-info }
> If a linear system with all coefficients and constants in $\mathbb{Q}$ has a solution in $\mathbb{F} \supseteq \mathbb{Q}$, it has a solution in $\mathbb{Q}$.

{: .prompt-proof }
> Row-reduce the augmented matrix. Gaussian elimination uses only addition, subtraction, multiplication, and division by nonzero pivots — so starting from rational entries, **every intermediate entry stays rational**, and the resulting echelon form $R$ is rational.
>
> The very same row operations are legitimate over $\mathbb{F}$ and produce the very same $R$. Now, if the system had no solution over $\mathbb{Q}$, the reduction would exhibit a row of the form $(0\;0\;\cdots\;0 \mid c)$ with $c \neq 0$ — but that row is equally an obstruction over $\mathbb{F}$, contradicting the existence of a solution there. So the reduction is consistent, and back-substitution (again only rational arithmetic) produces a solution with all $x_k \in \mathbb{Q}$. $\blacksquare$

{: .prompt-tip }
> **Field-independent (determined over the small field $\mathbb{K}$):** the minimal polynomial. Whether you compute it over $\mathbb{K}$ or over a bigger $\mathbb{F}$, you get the identical polynomial.
>
> **Field-dependent (can change when you enlarge):** eigenvalues, eigenvectors, diagonalizability. These are about *roots* of polynomials, and roots can require a bigger field to exist.
