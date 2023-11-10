---
title:  "Cryptography"
category: cryptography
tags: cryptography
---

* Random variable: a **function** that maps universal set to a subset: \\(X: U \rightarrow V\\).
* Randomized algorithm: \\(y \leftarrow A(m; r) \enspace \text{where} \enspace r \xleftarrow{R} \\{0,1\\}^{n}\\) defines a uniform random variable \\(y \xleftarrow{R} A(m)\\)
  - e.g. Encryption with a key: \\(A(m; k) = E(k, m)\\)

An important property of **XOR**: \\(Y\\) is a random variable over \\(\\{0,1\\}^n\\), \\(X \xleftarrow{R} \\{0,1\\}^{n}\\), \\(X\\) and \\(Y\\) are independent. Then \\(Z := Y \oplus X\\) is a uniform variable over \\(\\{0,1\\}^{n}\\).


Perfect Secrecy:
* Ciphertext should reveal no information about plaintext
* No ciphertext only attack
* One Time Pad (OTP) has perfect secrecy
* \\(\Rightarrow \lvert \mathcal{K} \rvert \ge \lvert \mathcal{M} \rvert\\)

Pseudo-Random Generator (PRG):
* \\(G: \\{0,1\\}^{s} \rightarrow \\{0,1\\}^{n}, \enspace s \ll n\\)
* Expands a seed to a much much larger random looking sequence
* **Effectively** compuatable by deterministic algorithms
* Must be **unpredictable**: \\(\forall i\\), no "eff" adv. can predict bit \\((i + 1)\\) for "non-neg" \\(\epsilon\\)

Stream Cipher:
* \\(E(k, m) := m \oplus G(k)\\)
* \\(D(k, c) := c \oplus G(k)\\)

