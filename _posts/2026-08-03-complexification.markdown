---
title:  "Complexification"
category: math
tags: [math, linear algebra]
mathjax_font: mathjax-pagella
mermaid: true
---

## Vector Space

{: .prompt-info }
> Suppose $V$ is a real vector space.
>
> The _complexification_ of $V$, denoted by $V_{\mathbb{C}}$, equals $V \times V$. An element of $V_{\mathbb{C}}$ is an ordered pair $(u,v)$, where $u,v \in V$, but we write this as $u + iv$.
>
> * Addition on $V_{\mathbb{C}}$ is defined by
>
> $$ (u_1 + iv_1) + (u_2 + iv_2) = (u_1 + u_2) + i(v_1 + v_2) $$
>
> for all $u_1,v_1,u_2,v_2 \in V$.
> * Complex scalar multiplication on $V_{\mathbb{C}}$ is defined by
>
> $$ (a + bi)(u + iv) = (au - bv) + i(av + bu)$$
>
> for all $a,b \in \mathbb{R}$ and all $u,v \in V$.
>
> With the definitions of addition and scalar multiplication as above, $V_{\mathbb{C}}$ is a _complex_ vector space.

{: .prompt-warning }
> The notation $u + iv$ is suggestive but the two slots are just the two components of an ordered pair. It pays to keep both readings available:
>
> * Addition acts **slot by slot**: $(u_1,v_1) + (u_2,v_2) = (u_1+u_2,\, v_1+v_2)$.
> * Multiplication by a **real** $a$ also acts slot by slot: $a(u,v) = (au, av)$.
> * Multiplication by $i$ **mixes** the slots: $i(u,v) = (-v, u)$.
>
> Almost every argument below turns on which of these two behaviours is in play.

{: .prompt-tip }
> If $v_1, \dots, v_n$ is a basis of $V$, then $v_1, \dots, v_n$ is also a basis of the complexification $V_{\mathbb{C}}$.

{: .prompt-proof }
> Here $v_k$ is read as $v_k + i0$, and "basis" means basis over $\mathbb{C}$.
>
> **Spanning.** Take any $u + iv \in V_{\mathbb{C}}$. Since $v_1,\dots,v_n$ is a basis of $V$, write $u = \sum a_k v_k$ and $v = \sum b_k v_k$ with $a_k, b_k \in \mathbb{R}$. Then
>
> $$u + iv = \sum_{k=1}^n a_k v_k + i\sum_{k=1}^n b_k v_k = \sum_{k=1}^n (a_k + b_k i)\, v_k$$
>
> The real and imaginary parts get absorbed into the complex coefficients.
>
> **Independence.** Suppose $\sum (a_k + b_k i) v_k = 0$ in $V_{\mathbb{C}}$. Unpacking, that's the pair $\left(\sum a_k v_k, \sum b_k v_k\right) = (0,0)$. Both components vanish in $V$, so real independence of $v_1,\dots,v_n$ forces every $a_k = 0$ and every $b_k = 0$, hence each complex coefficient is $0$. $\blacksquare$

{: .prompt-warning }
> Nothing is "missing" from the second slot: the scalar $i$ supplies it. Concretely $i(v_k + i0) = 0 + iv_k$, so $(0, v_k)$ already lies in the **complex** span of $v_k$. This is exactly the step that fails over $\mathbb{R}$ — see the next box.

{: .prompt-tip }
> * $\dim_{\mathbb{C}} V_{\mathbb{C}} = \dim_{\mathbb{R}} V$
> * $\dim_{\mathbb{R}} V_{\mathbb{C}} = 2\dim_{\mathbb{R}} V$, with real basis $v_1, \dots, v_n, iv_1, \dots, iv_n$.
>
> The same set $V \times V$ has dimension $n$ or $2n$ depending on which field one counts over. Always say which.

{: .prompt-tip }
> $V_{\mathbb{C}} = V \oplus iV$, where $V$ is identified with $\\{u + i0 : u \in V\\}$ and $iV = \\{0 + iv : v \in V\\}$.
>
> This is a direct sum of **real** subspaces only: neither summand is closed under multiplication by $i$, which swaps them. Its use below is always the same and always elementary — an element of $V \times V$ is $0$ exactly when both components are.

## Linear Map

{: .prompt-info }
> Suppose that $V$ and $W$ are real vector spaces and $T \in \mathcal{L}(V,W)$. Define $T_{\mathbb{C}}: V_{\mathbb{C}} \to W_{\mathbb{C}}$ by
>
> $$T_{\mathbb{C}}(u + iv) = Tu + iTv$$
>
> for all $u,v \in V$. In slot form: $T_{\mathbb{C}}(u,v) = (Tu, Tv)$ — apply $T$ to each component, with no cross-talk.

{: .prompt-tip }
> $T_{\mathbb{C}} \in \mathcal{L}(V_{\mathbb{C}}, W_{\mathbb{C}})$; that is, $T_{\mathbb{C}}$ is linear over $\mathbb{C}$, not merely over $\mathbb{R}$.

{: .prompt-proof }
> **Additivity** is immediate from slot-by-slot application of the additivity of $T$.
>
> **Homogeneity** needs the mixing scalar $i$ checked, so take $\alpha = a + bi$ with $a,b \in \mathbb{R}$:
>
> $$T_{\mathbb{C}}\big(\alpha(u + iv)\big) = T_{\mathbb{C}}\big((au - bv) + i(av + bu)\big) = (aTu - bTv) + i(aTv + bTu),$$
>
> $$\alpha\, T_{\mathbb{C}}(u+iv) = (a + bi)(Tu + iTv) = (aTu - bTv) + i(aTv + bTu).$$
>
> These agree, using only that $T$ is real-linear. $\blacksquare$

{: .prompt-tip }
> If $v_1, \dots, v_n$ is a basis of $V$ and $w_1, \dots, w_m$ a basis of $W$, then
>
> $$\mathcal{M}(T_{\mathbb{C}}) = \mathcal{M}(T),$$
>
> the matrices being taken with respect to these same lists viewed as complex bases. In particular the matrix of $T_{\mathbb{C}}$ can always be taken to have **real entries**.

{: .prompt-proof }
> The $k$-th column of $\mathcal{M}(T_{\mathbb{C}})$ records the expansion of $T_{\mathbb{C}}v_k = Tv_k + i0$. Expanding $Tv_k$ over $\mathbb{R}$ in $w_1, \dots, w_m$ gives an expansion over $\mathbb{C}$ that happens to use real scalars, and expansions in a basis are unique. So the columns agree with those of $\mathcal{M}(T)$. $\blacksquare$

{: .prompt-warning }
> The real entries are a feature of *this* basis. An arbitrary complex basis of $V_{\mathbb{C}}$ (say $v_1 + iv_2, v_2, \dots$) will generally give complex entries. Basis-independent conclusions still hold, but the real-ness stops being visible — so when real-ness is the point, choose a basis of $V$.

## Polynomial

{: .prompt-tip }
> **Lemma.** Suppose $V$ is a real vector space, $T \in \mathcal{L}(V)$, and $p \in \mathcal{P}(\mathbb{R})$. Then
>
> $$p(T_{\mathbb{C}}) = (p(T))_{\mathbb{C}}.$$

{: .prompt-proof }
> Complexification $S \mapsto S_{\mathbb{C}}$ preserves sums, products, and multiplication by **real** scalars. A polynomial in $T$ is built from exactly those three operations, so the whole construction passes through.
>
> Throughout, recall the definition $T_{\mathbb{C}}(u + iv) = Tu + i\,Tv$, and let $u, v \in V$ be arbitrary.
>
> **Sums.**
>
> $$(S+T)_{\mathbb{C}}(u+iv) = (S+T)u + i(S+T)v = (Su + i\,Sv) + (Tu + i\,Tv) = S_{\mathbb{C}}(u+iv) + T_{\mathbb{C}}(u+iv).$$
>
> **Products.**
>
> $$(ST)_{\mathbb{C}}(u+iv) = (ST)u + i\,(ST)v = S(Tu) + i\,S(Tv) = S_{\mathbb{C}}(Tu + i\,Tv) = S_{\mathbb{C}}T_{\mathbb{C}}(u+iv).$$
>
> Induction on $k$ then gives $$(T^k)_{\mathbb{C}} = (T_{\mathbb{C}})^k$$, the base case being $$(T^0)_{\mathbb{C}} = I_{\mathbb{C}} = I$$ on $V_{\mathbb{C}}$.
>
> **Real scalars.** For $a \in \mathbb{R}$:
>
> $$(aT)_{\mathbb{C}}(u+iv) = a\,Tu + i\,a\,Tv = a(Tu + i\,Tv) = a\,T_{\mathbb{C}}(u+iv).$$
>
> **Assembly.** Write $p(z) = \sum_{k=0}^m a_k z^k$ with every $a_k \in \mathbb{R}$. Then
>
> $$p(T_{\mathbb{C}}) = \sum_k a_k (T_{\mathbb{C}})^k = \sum_k a_k (T^k)_{\mathbb{C}} = \sum_k (a_k T^k)_{\mathbb{C}} = \Big(\sum_k a_k T^k\Big)_{\mathbb{C}} = (p(T))_{\mathbb{C}}. \qquad \blacksquare$$

{: .prompt-warning }
> The hypothesis $p \in \mathcal{P}(\mathbb{R})$ is not a technicality — without it the right-hand side is undefined. Evaluating $p(T)$ means forming $\sum a_k T^k$ inside $\mathcal{L}(V)$, which is a vector space over $\mathbb{R}$; there is no operator "$iT$" on a real space. In the proof, real-ness is used exactly once, at the step $a_k (T^k)_{\mathbb{C}} = (a_k T^k)_{\mathbb{C}}$.
>
> For a general $p \in \mathcal{P}(\mathbb{C})$, the most one can say is
>
> $$p(T_{\mathbb{C}}) = (g(T))_{\mathbb{C}} + i\,(h(T))_{\mathbb{C}},$$
>
> where $p = g + ih$ with $g,h$ real. This is *not* the complexification of any single real operator: $i(S)_{\mathbb{C}}$ sends $u + i0 \mapsto 0 + iSu$, pushing the slice $V$ into $iV$, whereas every complexification maps $V$ into $V$. That is exactly why Step 2 of the theorem below has to split $q$ instead of applying the Lemma directly.

{: .prompt-tip }
> **Theorem.** Suppose $\mathbf{F} = \mathbb{R}$, $V$ is finite-dimensional, and $T \in \mathcal{L}(V)$. Then the minimal polynomial of $T_{\mathbb{C}}$ equals the minimal polynomial of $T$.

{: .prompt-proof }
> Write $p = p_T$ and $q = p_{T_{\mathbb{C}}}$, and set $n = \deg p$, $m = \deg q$.
>
> Note the asymmetry in what we may assume. Since $V$ is a vector space over $\mathbb{R}$, the definition of the minimal polynomial forces $p \in \mathcal{P}(\mathbb{R})$. But $V_{\mathbb{C}}$ is a vector space over $\mathbb{C}$, so $q$ is drawn from the strictly larger pool $\mathcal{P}(\mathbb{C})$ and may *a priori* have complex coefficients. The content of the theorem is that this larger pool buys nothing.
>
> **Step 1 ($q \mid p$, hence $m \le n$).**
>
> $p$ has real coefficients, so the Lemma applies: $$p(T_{\mathbb{C}}) = (p(T))_{\mathbb{C}} = 0_{\mathbb{C}} = 0.$$ Thus $p$ annihilates $T_{\mathbb{C}}$, and since the minimal polynomial divides every annihilating polynomial, $q \mid p$. In particular $m \le n$.
>
> **Step 2 (split $q$ into real and imaginary parts).**
>
> The Lemma cannot be applied to $q$ directly, since $q$ need not be real. So write each coefficient $c_k = a_k + i b_k$ with $a_k, b_k \in \mathbb{R}$ and regroup:
>
> $$q(z) = \sum_{k=0}^m c_k z^k = g(z) + i\,h(z), \qquad g(z) = \sum_k a_k z^k, \quad h(z) = \sum_k b_k z^k,$$
>
> so that $g, h \in \mathcal{P}(\mathbb{R})$. Now the Lemma *does* apply, to each piece separately:
>
> $$0 = q(T_{\mathbb{C}}) = g(T_{\mathbb{C}}) + i\,h(T_{\mathbb{C}}) = (g(T))_{\mathbb{C}} + i\,(h(T))_{\mathbb{C}}.$$
>
> **Step 3 ($g(T) = 0$ and $h(T) = 0$).**
>
> Apply the operator identity above to an arbitrary $u + i0 \in V_{\mathbb{C}}$. Since $$(g(T))_{\mathbb{C}}(u + i0) = g(T)u + i0$$ and $$i\,(h(T))_{\mathbb{C}}(u + i0) = 0 + i\,h(T)u,$$ we get
>
> $$0 = g(T)u + i\,h(T)u, \qquad \text{i.e. the ordered pair} \quad \big(g(T)u,\; h(T)u\big) = (0,0).$$
>
> An element of $V_{\mathbb{C}} = V \times V$ is zero exactly when both components are, so $g(T)u = 0$ and $h(T)u = 0$. As $u \in V$ was arbitrary, $g(T) = 0$ and $h(T) = 0$.
>
> **Step 4 (locate the degrees of $g$ and $h$).**
>
> By construction $\deg g \le m$ and $\deg h \le m$, since both were assembled from the coefficients $c_0, \dots, c_m$ of $q$. Everything sharper follows from the single top coefficient. Since $q$ is monic of degree $m$,
>
> $$c_m = 1 = 1 + i \cdot 0,$$
>
> and the decomposition of a complex number into real and imaginary parts is unique, so $a_m = 1$ and $b_m = 0$. Hence:
>
> * the degree-$m$ coefficient of $g$ is $a_m = 1 \neq 0$, so $\deg g = m$ **and** $g$ is monic;
> * the degree-$m$ coefficient of $h$ is $b_m = 0$, so $\deg h \le m - 1 < m$ (this includes $h = 0$, under the convention $\deg 0 = -\infty$).
>
> The asymmetry is the point: $a_m \ne 0$ pins $g$ *at* degree $m$, while $b_m = 0$ pushes $h$ *below* it. The strictness is used in Step 5.
>
> **Step 5 (conclusion).**
>
> By Steps 3 and 4, $g$ is a monic polynomial in $\mathcal{P}(\mathbb{R})$ with $g(T) = 0$ — that is, $g$ is a competitor to $p$ in the very contest that defines $p$. Minimality of $p$ therefore gives $n \le \deg g = m$. Combined with Step 1,
>
> $$n = m = \deg g.$$
>
> So $g$ is a monic real annihilator of $T$ of minimal degree, and the minimal polynomial is the *unique* such polynomial, whence $g = p$.
>
> Finally $h = 0$: otherwise, dividing $h$ by its leading coefficient would give a monic real annihilator of $T$ of degree $\deg h < m = n$, contradicting minimality of $p$.
>
> Therefore $q = g + i\,h = g = p. \qquad \blacksquare$

{: .prompt-tip }
> **Corollary.** The minimal polynomial of $T_{\mathbb{C}}$ has real coefficients.

## Eigenvalue

Both results below rest on the standard characterization: for $\lambda \in \mathbf{F}$, $\lambda$ is an eigenvalue of an operator iff $\lambda$ is a zero of its minimal polynomial.

{: .prompt-info }
> Suppose $ \mathbf{F} = \mathbb{R} $ and $ \lambda \in \mathbb{R} $. Then
>
> $\lambda$ is an eigenvalue of $T \iff \lambda$ is an eigenvalue of the complexification $ T_{\mathbb{C}}. $

{: .prompt-proof }
> Let $p$ be the minimal polynomial of $T$. By the theorem above, $p$ is also the minimal polynomial of $T_\mathbb{C}$. Then for $\lambda \in \mathbb{R}$:
>
> $$\lambda \text{ is an eigenvalue of } T \iff p(\lambda) = 0 \iff \lambda \text{ is an eigenvalue of } T_\mathbb{C},$$
>
> the first equivalence by the characterization applied to $T$ over $\mathbb{R}$, the second by the same characterization applied to $T_\mathbb{C}$ over $\mathbb{C}$ — legitimate because the two operators have the *same* polynomial $p$. $\blacksquare$

{: .prompt-proof }
> **Directly.** ($\Rightarrow$) If $Tv = \lambda v$ with $v \neq 0$, then
>
> $$T_\mathbb{C}(v + i0) = Tv + i0 = \lambda v + i0 = \lambda(v + i0),$$
>
> the last step because $\lambda$ is real and so scales slots; and $v + i0 \neq 0$.
>
> ($\Leftarrow$) If $T_\mathbb{C}(u + iv) = \lambda(u+iv)$ with $u + iv \neq 0$, comparing slots gives $Tu = \lambda u$ and $Tv = \lambda v$. At least one of $u,v$ is nonzero, so $\lambda$ is an eigenvalue of $T$. $\blacksquare$

{: .prompt-warning }
> The characterization requires $\lambda \in \mathbf{F}$: the scalar must belong to the field of the space. Since $\mathbb{R} \subseteq \mathbb{C}$, a real $\lambda$ is eligible on both sides, which is why the hypothesis $\lambda \in \mathbb{R}$ appears. For non-real $\lambda$ the right-hand side still makes sense but the left does not — there is no equation $Tv = \lambda v$ to write down, since $\lambda v$ is undefined for $v \in V$. So the non-real case needs a different statement entirely.

{: .prompt-info }
> Suppose $ \mathbf{F} = \mathbb{R} $ and $ \lambda \in \mathbb{C} $. Then
>
> $\lambda$ is an eigenvalue of $T_{\mathbb{C}} \iff \bar{\lambda} $ is an eigenvalue of $ T_{\mathbb{C}} $.
>
> Note this is a statement *internal* to $T_{\mathbb{C}}$ — a symmetry of its spectrum — not a transfer between two operators.

{: .prompt-proof }
> **Via the minimal polynomial.** Let $p$ be the minimal polynomial of $T$, which is also that of $T_\mathbb{C}$; crucially $p \in \mathcal{P}(\mathbb{R})$. Then
>
> $$\lambda \text{ eigenvalue of } T_\mathbb{C} \iff p(\lambda) = 0 \iff \overline{p(\lambda)} = 0 \iff p(\bar\lambda) = 0 \iff \bar\lambda \text{ eigenvalue of } T_\mathbb{C}.$$
>
> The only nontrivial step is $\overline{p(\lambda)} = p(\bar\lambda)$, and that is exactly where real coefficients enter. Conjugation preserves sums and products, so with $p(z) = \sum a_k z^k$,
>
> $$\overline{p(\lambda)} = \sum \overline{a_k}\,\overline{\lambda}^k = \sum a_k \bar\lambda^{\,k} = p(\bar\lambda),$$
>
> using $\overline{a_k} = a_k$. Had $p$ carried a genuinely complex coefficient this would fail, and the spectrum would have no reason to be conjugate-symmetric. $\blacksquare$

{: .prompt-proof }
> **Via conjugation.** Define $C : V_{\mathbb{C}} \to V_{\mathbb{C}}$ by
>
> $$C(u + iv) = u - iv.$$
>
> Three properties:
>
> * $C$ is an **involution**, hence a bijection: $C\big(C(u+iv)\big) = C(u-iv) = u+iv$, so $C^{-1} = C$. In particular $C$ sends nonzero vectors to nonzero vectors.
> * $C$ is **conjugate-linear**: $C(w_1 + w_2) = Cw_1 + Cw_2$ and $C(\alpha w) = \bar\alpha\, Cw$ for $\alpha \in \mathbb{C}$. For the second, with $\alpha = a+bi$ and $w = u+iv$, both sides equal $(au - bv) + i\big(-(av+bu)\big)$ by the scalar-multiplication rule. (So $C$ is real-linear but *not* complex-linear.)
> * $C$ **commutes with $T_{\mathbb{C}}$**:
>
> $$T_{\mathbb{C}}\big(C(u+iv)\big) = T_{\mathbb{C}}(u - iv) = Tu - iTv = C(Tu + iTv) = C\big(T_{\mathbb{C}}(u+iv)\big).$$
>
> Now suppose $T_{\mathbb{C}}w = \lambda w$ with $w \neq 0$. Apply $C$ to both sides: on the left the commuting relation gives $C(T_{\mathbb{C}}w) = T_{\mathbb{C}}(Cw)$, and on the right conjugate-linearity gives $C(\lambda w) = \bar\lambda\,Cw$. Hence
>
> $$T_{\mathbb{C}}(Cw) = \bar\lambda\,(Cw),$$
>
> with $Cw \neq 0$ since $C$ is a bijection. So $\bar\lambda$ is an eigenvalue of $T_{\mathbb{C}}$.
>
> The converse needs no new work: apply this to $\bar\lambda$ in place of $\lambda$ and use $\overline{\bar\lambda} = \lambda$. $\blacksquare$

{: .prompt-tip }
> The second proof gives more than the statement: $C$ restricts to a bijection
>
> $$C\big(E(\lambda, T_{\mathbb{C}})\big) = E(\bar\lambda, T_{\mathbb{C}}),$$
>
> and a conjugate-linear bijection carries a complex basis to a complex basis (independence and spanning survive, with coefficients conjugated). Hence
>
> $$\dim E(\lambda, T_{\mathbb{C}}) = \dim E(\bar\lambda, T_{\mathbb{C}}).$$
>
> The same argument upgrades to **generalized** eigenspaces. From conjugate-linearity and commuting, $C(T_{\mathbb{C}} - \lambda I) = (T_{\mathbb{C}} - \bar\lambda I)C$, so by induction $$C(T_{\mathbb{C}} - \lambda I)^k = (T_{\mathbb{C}} - \bar\lambda I)^k C$$ for every $k$. Therefore
>
> $$C\big(G(\lambda, T_{\mathbb{C}})\big) = G(\bar\lambda, T_{\mathbb{C}}), \qquad \dim G(\lambda, T_{\mathbb{C}}) = \dim G(\bar\lambda, T_{\mathbb{C}}).$$
>
> Non-real eigenvalues of a real operator come in conjugate pairs with equal multiplicities, geometric **and** algebraic.

{: .prompt-tip }
> **How the two results divide the work.** For $\lambda \in \mathbb{R}$ the conjugate-pair statement is vacuous, since $\bar\lambda = \lambda$; all its content is in the non-real case, which is precisely the case the first result cannot address. Together: real eigenvalues of $T_{\mathbb{C}}$ are inherited from $T$ and are literally the same objects, while non-real ones have no counterpart in $T$ at all but are forced to occur in $\lambda, \bar\lambda$ pairs.

## Application: odd dimension forces a real eigenvalue

{: .prompt-tip }
> Suppose $V$ is a real vector space with $\dim V$ odd and $T \in \mathcal{L}(V)$. Then $T$ has an eigenvalue.

{: .prompt-proof }
> Let $n = \dim_{\mathbb{R}} V$, so $\dim_{\mathbb{C}} V_{\mathbb{C}} = n$ is odd, and let $\lambda_1, \dots, \lambda_r$ be the **distinct** eigenvalues of $T_{\mathbb{C}}$ (there is at least one, since $V_{\mathbb{C}}$ is a nonzero complex vector space).
>
> Decompose $V_{\mathbb{C}}$ into generalized eigenspaces:
>
> $$V_{\mathbb{C}} = G(\lambda_1, T_{\mathbb{C}}) \oplus \cdots \oplus G(\lambda_r, T_{\mathbb{C}}), \qquad \sum_{j=1}^r d_j = n, \quad d_j := \dim G(\lambda_j, T_{\mathbb{C}}).$$
>
> Split the eigenvalues into the real ones and the non-real ones. By the previous box, conjugation permutes the eigenvalues and preserves the $d_j$; and it has no fixed points among the non-real ones, since $\lambda = \bar\lambda$ forces $\lambda$ real. So the non-real eigenvalues fall into genuine two-element pairs $\{\lambda, \bar\lambda\}$, each contributing $d + d = 2d$ to the total. Hence
>
> $$n = \underbrace{\sum_{\lambda_j \in \mathbb{R}} d_j}_{=:N} \;+\; \underbrace{\sum_{\text{pairs}} 2d}_{\text{even}}.$$
>
> Since $n$ is odd, $N$ is odd, so $N \neq 0$ and $T_{\mathbb{C}}$ has at least one real eigenvalue $\lambda$. By the first eigenvalue result, that $\lambda$ is an eigenvalue of $T$. $\blacksquare$

{: .prompt-warning }
> The parity is used only to rule out $N = 0$: the pairing makes the non-real contribution even, so an odd total cannot be accounted for by non-real eigenvalues alone. For even $n$ the argument collapses, as it must — rotation by $90°$ on $\mathbb{R}^2$ has eigenvalues $i, -i$, one conjugate pair, and no real eigenvalue.
>
> **The count must be by multiplicity.** It is tempting to argue "an odd-dimensional space cannot have its eigenvalues occur solely in conjugate pairs", but the *number of distinct* eigenvalues has no bearing on parity of the dimension — an operator on a $3$-dimensional space may have exactly two distinct eigenvalues. Ordinary eigenspaces are also not enough, since they need not fill $V_{\mathbb{C}}$ when $T_{\mathbb{C}}$ is not diagonalizable. Generalized eigenspaces are what make the sum come out to exactly $n$.

{: .prompt-warning }
> This is not how Axler first proves the result. In Chapter 5 he obtains it without complexification, from the factorization of real polynomials into linear and irreducible quadratic factors (equivalently, from the fact that every operator on a nonzero finite-dimensional real vector space has an invariant subspace of dimension $1$ or $2$). The complexification material sits much later in the book, so the proof above is best filed as a second look rather than the canonical one.
>
> There is no circularity: the generalized eigenspace decomposition is proved for operators on **complex** vector spaces, using only that such an operator has an eigenvalue, and does not depend on the odd-dimensional real result.
