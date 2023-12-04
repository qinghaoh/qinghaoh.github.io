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
|DES|64|56|16|
|3DES|64|168|48|
|AES|128|128/192/256|10|

Considerably slower than stream ciphers.

||Pseudo Random Function (PRF)|Pseudo Random Permutation (PRP)|
|-|-|-|
|Function|`E(k,x)`|`E(k,x)` and `D(k,y)`|
|Invertible?|N/A|Yes, one-to-one|
|Deterministic?|N/A|Yes|

PRP \\(\subset\\) PRF

**Secure PRFs**
* A random function in \\(Funs[X,Y]\\) (size = \\(\lvert Y \rvert ^ {\lvert X \rvert}\\)) is indistinguishable from a random function in \\(S_F = \{ F(k,\cdot)) \enspace \text{s.t.} \enspace k \in K\}\\) (size = \\(\lvert K \rvert\\))
  - \\(S_F \subseteq Funs[X,Y]\\)
* Secure PRF \\(\Rightarrow\\) Secure PRG
  - \\(F:K \times \{0,1\}^{n} \rightarrow \{0,1\}^{n} \enspace G:k \rightarrow \{0,1\}^{nt}\\), \\(G(k) = F(k,0) \parallel F(k,1) \parallel \ldots \parallel F(k,t)\\), 
  - Parallelizable

**Feistel Network**
* Build *invertible* function from arbitrary functions
* ![Construction](https://upload.wikimedia.org/wikipedia/commons/f/fa/Feistel_cipher_diagram_en.svg)
* Used in many block ciphers, but not AES
* Luby-Rackoff Theorem: Secure PRF \\(\xrightarrow{\text{3-round Feistel}}\\) Secure PRP

**DES (Data Encryption Standard)**
* Overall Feistel structure
  - ![DES Main Networking](https://upload.wikimedia.org/wikipedia/commons/6/6a/DES-main-network.png){: width="250" }
* Key sechedule
  - ![DES Key Schedule](https://upload.wikimedia.org/wikipedia/commons/0/06/DES-key-schedule.png)
* F function
  - ![DES F Function](https://upload.wikimedia.org/wikipedia/commons/a/a3/DES-f-function.png){: width="400" }
* [S-box](https://en.wikipedia.org/wiki/S-box)
  - 6 bits -> 4 bits
  - 4-to-1 maps: 1 output has 4 preimages
  - Nonlinear, otherwise DES is linear (_insecure_)

**Exhausive Search Attacks**
* Suppose DES is an ideal cipher:
  - \\(2^56\\) (= number of keys) random invertible functions
  - \\(\forall m,c\\), \\(\exists\\) at most one key \\(k \enspace \text{s.t.} \enspace \Pr[c=DES(k,m)] >= 1 - 2^{56}\frac{1}{2^{64}} = 99.5%\\)
* 3DES
  - \\(E(k1, D(k2, E(k3, m)))\\): not 3 E's because when \\(k1=k2=k3\\) we get hardware implementation of normal DES
  - Meet-in-the-middle Attack \\(\approx 2^{118} > 2^{90}\\).
* 2DES
  - \\(E(k1, E(k2, m))\\)
  - Meet-in-the-middle Attack
    * \\(2^{56}\log(2^{56}) + 2^{56}\log(2^{56}) \lt 2^{63} \lll 2^{112}\\): build and sort in one way + binary search in the other way
* DESX
  - \\(k1 \oplus E(k2, m \oplus k3)\\)
  - Key size = 64 + 56 + 64 = 184 bits
  - Meet-in-the-middle Attack \\(2^{120}\\)
  - Vulnerable to more subtle attacks
  - \\(k1 \oplus E(k2, m)\\) and \\(E(k2, m \oplus k1)\\) are both wrong constructions

**More Attacks**
* Attacks on the implementation
  - Side channel attacks: time, power, ...
  - Fault attacks: computing errors in the last round exposes the secret key
* Linear and differential attacks
  - There's a dependence between message, ciphertext and the key bits
  - 5th S-box of DES it too close to a linear function
  - Success probability >= 97.7% given \\(1/\epsilon^2\\) random \\((m,c)\\) pairs. For DES:
    * \\(\epsilon = 1/2^{21}\\). Can find 14 key bits this way in time \\(2^{42}\\)
    * The remaining 42 key bits can be found by brute force in \\(2^{42}\\)
    * In total \\(2^{43}\\)
* Quantum Attacks
  - Could solve generic search problem in \\(O(\lvert X \rvert^{1/2})\\).
   
