---
title:  "Cryptography"
category: cryptography
tags: cryptography
---

# Cryptography I

Study notes for *Cryptography I (Standord University)* on Coursera.

## Stream Ciphers

**Probability**
* Random variable: a *function* that maps universal set to a subset: \\(X: U \rightarrow V\\).
* Randomized algorithm: \\(y \leftarrow A(m; r) \enspace \text{where} \enspace r \xleftarrow{R} \\{0,1\\}^{n}\\) defines a uniform random variable \\(y \xleftarrow{R} A(m)\\)
  - e.g. Encryption with a key: \\(A(m; k) = E(k, m)\\)

An important property of *XOR*: \\(Y\\) is a random variable over \\(\\{0,1\\}^n\\), \\(X \xleftarrow{R} \\{0,1\\}^{n}\\), \\(X\\) and \\(Y\\) are independent. Then \\(Z := Y \oplus X\\) is a uniform variable over \\(\\{0,1\\}^{n}\\).

**Perfect Secrecy**
* Ciphertext should reveal no information about plaintext
* No ciphertext only attack
* One Time Pad (OTP) has perfect secrecy
* \\(\Rightarrow \lvert \mathcal{K} \rvert \ge \lvert \mathcal{M} \rvert\\)

**Pseudo-Random Generator (PRG)**
* \\(G: \\{0,1\\}^{s} \rightarrow \\{0,1\\}^{n}, \enspace s \ll n\\)
* Expands a seed to a much much larger random looking sequence
* *Effectively* compuatable by deterministic algorithms
* Weak PRGs: Do not use for crypto
  - [Linear Congruential Generator](https://en.wikipedia.org/wiki/Linear_congruential_generator)
  - `glibc random()`

**Negligibility**
* In practice:
  - Non-negligible: scalar \\(\epsilon \ge \frac{1}{2^{30}}\\)
  - Negligible: scalar \\(\epsilon \le \frac{1}{2^{80}}\\)
* In theory:
  - Non-negligible: function \\(\exists d: \epsilon(\lambda) \ge \frac{1}{\lambda^{d}}\\), infinitely often
  - Negligible: function \\(\forall d, \lambda \ge \lambda_{d}: \epsilon(\lambda) \le \frac{1}{\lambda^{d}}\\)

**Stream Cipher**
* \\(E(k, m) := m \oplus G(k)\\)
* \\(D(k, c) := c \oplus G(k)\\)
* Attacks
  - Two time pad (e.g. MS-PPTP)
  - Related keys (e.g. 802.11b WEP `PRG(IV || k)`)
    * Solution: `PRG(PRG(IV || k))`
  - OTP is malleable (no integrity)
* Real world examples
  - RC4
    * SW
    * Not recommended: bias in output; related key attacks
  - [CSS](https://en.wikipedia.org/wiki/Content_Scramble_System)
    * HW
    * Badly broken: [LFSR](https://en.wikipedia.org/wiki/Linear-feedback_shift_register)
  - eSTREAM (PRG with a *nonce* `R`: \\(PRG: \\{0,1\\}^{s} \times R \rightarrow \\{0,1\\}^{n}, \enspace s \ll n\\))
    * [Salsa20](https://en.wikipedia.org/wiki/Salsa20): SW + HW
    * Sosemanuk
    
**Advantage**

$$Adv_{PRG}[A,G] = \lvert \Pr_{k \xleftarrow{R} \mathcal{k}}[A(G(k)) = 1] - \Pr_{r \xleftarrow{R} \\{0,1\\}^{n}}[A(r) = 1]\rvert \in [0, 1]$$

**Secure PRG**
* \\(\forall\\) "eff" statistical tests \\(A\\), \\(Adv_{PRG}[A,G]\\) is "neg".
* Unprovable (\\(\Rightarrow P \ne NP\\))
* \\(\Leftrightarrow\\) *Unpredictable* (\\(\Leftarrow\\) Yao's Theorem)
  - \\(\forall i\\), no "eff" adv. can predict bit \\((i + 1)\\) for "non-neg" \\(\epsilon\\)

**Computationally indistinguishable**

(\\(P_1 \approxeq_{p} P_2\\)): \\(\forall\\) "eff" statistical tests \\(A\\),

$$\lvert \Pr_{x \leftarrow P_1}[A(x) = 1] - \Pr_{x \leftarrow P_2}[A(x) = 1]\rvert < neg$$

**Semantic Security**

$$Adv_{SS}[A,\mathbb{E}] := \lvert \Pr[W_0] - \Pr[W_1]\rvert \in [0, 1]$$

where \\(W_b\\) is the event that \\(EXP(b) = 1\\)

\\(\forall\\) "eff" \\(A\\), \\(Adv_{SS}[A, \mathbb{E}]\\) is "neg".

Secure PRG \\(\Rightarrow\\) Semantically secure stream cipher

## Block Ciphers

**Block Ciphers**

**Iteration**

Key expansion -> Round function (`R(k,m)`)

||Block size (bits)|Key size (bits)|Number of rounds|
|-|-|-|-|
|3DES|64|168|48|
|AES|128|128/192/256|10|

Considerably slower than stream ciphers.

||Pseudo Random Function (PRF)|Pseudo Random Permutation (PRP)|
|-|-|-|
|Function|`E(k,x)`|`E(k,x)` and `D(k,y)`|
|Invertible?|N/A|Yes, one-to-one|
|Deterministic?|N/A|Yes|

PRP \\(\subset\\) PRF


