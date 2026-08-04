---
title:  "Complexification"
category: math
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Complexification

{: .prompt-info }
> _Complexification_
>
> $V_{\mathbb{C}} = V \oplus iV$ with elements $u + iw$ ($u, w \in V$), and $T_{\mathbb{C}}(u + iw) = Tu + iTw$.

{: .prompt-proof }
> Write $p = p_T$ and $q = p_{T_{\mathbb{C}}}$.
>
> **Direction 1: $q \mid p$**
>
> $p$ has real coefficients and $p(T) = 0$, so $p(T_{\mathbb{C}}) = 0$. Thus $p$ annihilates $T_{\mathbb{C}}$, giving $q \mid p$. In particular $\deg q \le \deg p$.
>
> **Direction 2: $p \mid q$**
>
>$q$ a priori has *complex* coefficients, so **split $q$ into real and imaginary parts.**
>
> Write $q(z) = g(z) + i\,h(z)$, where $g, h \in \mathcal{P}(\mathbb{R})$ are obtained by taking the real and imaginary parts of each coefficient of $q$. Since $g, h$ have real coefficients:
>
> $$0 = q(T_{\mathbb{C}}) = g(T_{\mathbb{C}}) + i\,h(T_{\mathbb{C}}) = (g(T))_{\mathbb{C}} + i\,(h(T))_{\mathbb{C}}.$$
>
> Evaluate at $u + i\cdot 0 = u, \ \forall u \in V$: $$(g(T))_{\mathbb{C}}u = g(T)u \in V$$ and $i(h(T))_{\mathbb{C}}u = i\,h(T)u \in iV$. These lie in the complementary summands $V$ and $iV$, so both must vanish:
>
> $$g(T)u = 0 \quad\text{and}\quad h(T)u = 0 \quad\text{for all } u,$$
>
> i.e. $g(T) = 0$ and $h(T) = 0$.
>
> Now $q$ is **monic**, so its leading coefficient is $1 = 1 + i\cdot 0$; hence $g$ is monic of degree $\deg q$, while $\deg h < \deg q$. Since $g(T) = 0$ and $g$ is a monic real annihilator of $T$, minimality gives $p \mid g. \blacksquare$

### Complexification

{: .prompt-info }
> Suppose $ \mathbf{F} = \mathbb{R} $ and $ \lambda \in \mathbb{R} $. Then
>
> $\lambda$ is an eigenvalue of $T \iff \lambda$ is an eigenvalue of the complexification $ T_{\mathbb{C}}. $

{: .prompt-info }
> Suppose $ \mathbf{F} = \mathbb{R} $ and $ \lambda \in \mathbb{C} $. Then
>
> $\lambda$ is an eigenvalue of $T_{\mathbb{C}} \iff \bar{\lambda} $ is an eigenvalue of $ T_{\mathbb{C}} $.

{: .prompt-proof }
> Recall $V_{\mathbb{C}} = \\{ u + iv : u, v \in V \\}$ with $T_{\mathbb{C}}(u+iv) = Tu + iTv$. Define **conjugation** $C : V_{\mathbb{C}} \to V_{\mathbb{C}}$ by
>
> $$C(u + iv) = u - iv.$$
>
> Two properties, both routine to check:
>
> - $C$ is a **bijection**, in fact an involution: $C\big(C(u+iv)\big) = C(u - iv) = u + iv$, so $C^{-1} = C$. In particular $C$ sends nonzero vectors to nonzero vectors.
> - $C$ is **conjugate-linear**: $C(\alpha w) = \bar\alpha\, C(w)$ for $\alpha \in \mathbb{C}$. (Direct from the scalar-multiplication rule on $V_{\mathbb{C}}$.)
>
> **Claim:** $T_{\mathbb{C}} \circ C = C \circ T_{\mathbb{C}}$, i.e. $T_{\mathbb{C}}$ commutes with conjugation.
>
> $$T_{\mathbb{C}}\big(C(u+iv)\big) = T_{\mathbb{C}}(u - iv) = Tu - iTv = C(Tu + iTv) = C\big(T_{\mathbb{C}}(u+iv)\big).\ \checkmark$$
>
> Suppose $\lambda$ is an eigenvalue of $T_{\mathbb{C}}$: there is $w \neq 0$ in $V_{\mathbb{C}}$ with
>
> $$T_{\mathbb{C}}\,w = \lambda w.$$
>
> Apply $C$ to both sides. On the right, conjugate-linearity gives $C(\lambda w) = \bar\lambda\, Cw$. On the left, the commuting relation gives $C(T_{\mathbb{C}} w) = T_{\mathbb{C}}(Cw)$. Hence
>
> $$T_{\mathbb{C}}(Cw) = \bar\lambda\,(Cw).$$
>
> Since $C$ is a bijection and $w \neq 0$, we have $Cw \neq 0$. So $Cw$ is an eigenvector of $T_{\mathbb{C}}$ with eigenvalue $\bar\lambda$ — meaning **$\bar\lambda$ is an eigenvalue of $T_{\mathbb{C}}$.**
>
> That proves the forward direction. The converse needs no new work: the statement is symmetric under $\lambda \leftrightarrow \bar\lambda$, since $\overline{\bar\lambda} = \lambda$. Concretely, apply what we just proved to $\bar\lambda$ in place of $\lambda$: if $\bar\lambda$ is an eigenvalue, so is $\overline{\bar\lambda} = \lambda$. $\blacksquare$

{: .prompt-tip }
> The proof gives more than the statement: conjugation *matches up* the two eigenspaces,
>
> $$C\big(E(\lambda, T_{\mathbb{C}})\big) = E(\bar\lambda, T_{\mathbb{C}}),$$
>
> so in particular $\dim E(\lambda, T_{\mathbb{C}}) = \dim E(\bar\lambda, T_{\mathbb{C}})$: complex eigenvalues of a real operator come in conjugate pairs *with equal multiplicities*.
