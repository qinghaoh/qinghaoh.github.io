---
title:  "Cryptography"
category: cryptography
tags: cryptography
mermaid: true
---

# Cryptography I

Study notes for *Cryptography I (Standord University)* on Coursera.

## Stream Ciphers

**Probability**
* Random variable: a *function* that maps universal set to a subset: $$ X: U \rightarrow V $$.
* Randomized algorithm: $$ y \leftarrow A(m; r) \enspace \text{where} \enspace r \xleftarrow{R} \\{0,1\\}^{n} $$ defines a uniform random variable $$ y \xleftarrow{R} A(m) $$
  - e.g. Encryption with a key: $$ A(m; k) = E(k, m) $$

An important property of *XOR*: $$ Y $$ is a random variable over $$ \\{0,1\\}^n $$, $$ X \xleftarrow{R} \\{0,1\\}^{n} $$, $$ X $$ and $$ Y $$ are independent. Then $$ Z := Y \oplus X $$ is a uniform variable over $$ \\{0,1\\}^{n} $$.

**Perfect Secrecy**
* Ciphertext should reveal no information about plaintext
* No ciphertext only attack
* One Time Pad (OTP) has perfect secrecy
* \$$ \Rightarrow \lvert \mathcal{K} \rvert \ge \lvert \mathcal{M} \rvert $$

**Pseudo-Random Generator (PRG)**
* \$$ G: \\{0,1\\}^{s} \rightarrow \\{0,1\\}^{n}, \enspace s \ll n $$
* Expands a seed to a much much larger random looking sequence
* *Effectively* compuatable by deterministic algorithms
* Weak PRGs: Do not use for crypto
  - [Linear Congruential Generator](https://en.wikipedia.org/wiki/Linear_congruential_generator)
  - `glibc random()`

**Negligibility**
* In practice:
  - Non-negligible: scalar $$ \epsilon \ge \frac{1}{2^{30}} $$
  - Negligible: scalar $$ \epsilon \le \frac{1}{2^{80}} $$
* In theory:
  - Non-negligible: function $$ \exists d: \epsilon(\lambda) \ge \frac{1}{\lambda^{d}} $$, infinitely often
  - Negligible: function $$ \forall d, \lambda \ge \lambda_{d}: \epsilon(\lambda) \le \frac{1}{\lambda^{d}} $$

**Stream Cipher**
* \$$ E(k, m) := m \oplus G(k) $$
* \$$ D(k, c) := c \oplus G(k) $$
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
  - eSTREAM (PRG with a *nonce* `R`: $$ PRG: \\{0,1\\}^{s} \times R \rightarrow \\{0,1\\}^{n}, \enspace s \ll n $$)
    * [Salsa20](https://en.wikipedia.org/wiki/Salsa20): SW + HW
    * Sosemanuk

**Advantage**

$$ Adv_{PRG}[A,G] = \lvert \Pr_{k \xleftarrow{R} \mathcal{k}}[A(G(k)) = 1] - \Pr_{r \xleftarrow{R} \{0,1\}^{n}}[A(r) = 1]\rvert \in [0, 1] $$

**Secure PRG**
* $$ \forall $$ "eff" statistical tests $$ A $$, $$ Adv_{PRG}[A,G] $$ is "neg".
* Unprovable ($$ \Rightarrow P \ne NP $$)
* $$ \Leftrightarrow $$ *Unpredictable* ($$ \Leftarrow $$ Yao's Theorem)
  - $$ \forall i $$, no "eff" adv. can predict bit $$ (i + 1) $$ for "non-neg" $$ \epsilon $$

**Computationally indistinguishable**

($$ P_1 \approxeq_{p} P_2 $$): $$ \forall $$ "eff" statistical tests $$ A $$,

$$ \lvert \Pr_{x \leftarrow P_1}[A(x) = 1] - \Pr_{x \leftarrow P_2}[A(x) = 1]\rvert < neg $$

**Semantic Security**

$$ Adv_{SS}[A,\mathbb{E}] := \lvert \Pr[W_0] - \Pr[W_1]\rvert \in [0, 1] $$

where $$ W_b $$ is the event that $$ EXP(b) = 1 $$

$$ \forall $$ "eff" $$ A $$, $$ Adv_{SS}[A, \mathbb{E}] $$ is "neg".

Secure PRG $$ \Rightarrow $$ Semantically secure stream cipher

## Block Ciphers

**Block Ciphers**

**Iteration**

Key expansion -> Round function (`R(k,m)`)

|      | Block size (bits) | Key size (bits) | Number of rounds | Network   | Secure?         |
| ---- | ----------------- | --------------- | ---------------- | --------- | --------------- |
| DES  | 64                | 56              | 16               | Feistel   | No              |
| 3DES | 64                | 168             | 48               | Feistel   | Yes (Heuristic) |
| AES  | 128               | 128/192/256     | 10/12/14         | Subs-Perm | Yes (Heuristic) |

Considerably slower than stream ciphers.

|                | Pseudo Random Function (PRF) | Pseudo Random Permutation (PRP) |
| -------------- | ---------------------------- | ------------------------------- |
| Function       | `E(k,x)`                     | `E(k,x)` and `D(k,y)`           |
| Invertible?    | N/A                          | Yes, one-to-one                 |
| Deterministic? | N/A                          | Yes                             |

PRP $$ \subset $$ PRF

**Secure PRFs**
* A random function in $$ Funs[X,Y] $$ (size = $$ \lvert Y \rvert ^ {\lvert X \rvert} $$) is indistinguishable from a random function in $$ S_F = \\{ F(k,\cdot) \enspace \text{s.t.} \enspace k \in K\\} $$ (size = $$ \lvert K \rvert $$)
  - $$ S_F \subseteq Funs[X,Y] $$
* Secure PRF $$ \Rightarrow $$ Secure PRG
  - $$ F:K \times \\{0,1\\}^{n} \rightarrow \\{0,1\\}^{n} \enspace G:k \rightarrow \\{0,1\\}^{nt} $$, $$ G(k) = F(k,0) \parallel F(k,1) \parallel \ldots \parallel F(k,t) $$
  - e.g. Deterministic CTR mode
  - Parallelizable

**Secure PRPs**
* *PRF Switching Lemma*
  - $$ \lvert Adv_{PRF}[A,E] - Adv_{PRP}[A,E] \rvert \lt q^2/2\lvert X \rvert $$, where $$ q $$ is the number of queries
  - If $$ \lvert X \rvert $$ is sufficiently large, then Secure PRP $$ \Rightarrow $$ Secure PRF

**Feistel Network**
* Build *invertible* function from arbitrary functions
* ![Construction](https://upload.wikimedia.org/wikipedia/commons/f/fa/Feistel_cipher_diagram_en.svg){: width="300" }
* Used in many block ciphers, but not AES
* Luby-Rackoff Theorem: Secure PRF $$ \xrightarrow{\text{3-round Feistel}} $$ Secure PRP

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
  - $$ 2^{56} $$ (= number of keys) random invertible functions
  - $$ \forall m,c $$, $$ \exists $$ at most one key $$ k \enspace \text{s.t.} \enspace \Pr[c=DES(k,m)] \ge 1 - 2^{56}\frac{1}{2^{64}} = 99.5\% $$
* 3DES
  - $$ E(k1, D(k2, E(k3, m))) $$: not 3 E's because when $$ k1=k2=k3 $$ we get hardware implementation of normal DES
  - Meet-in-the-middle Attack $$ \approx 2^{118} > 2^{90} $$.
* 2DES
  - $$ E(k1, E(k2, m)) $$
  - Meet-in-the-middle Attack
    * $$ 2^{56}\log(2^{56}) + 2^{56}\log(2^{56}) \lt 2^{63} \lll 2^{112} $$: build and sort in one way + binary search in the other way
* DESX
  - $$ k1 \oplus E(k2, m \oplus k3) $$
  - Key size = 64 + 56 + 64 = 184 bits
  - Meet-in-the-middle Attack $$ 2^{120} $$
  - Vulnerable to more subtle attacks
  - $$ k1 \oplus E(k2, m) $$ and $$ E(k2, m \oplus k1) $$ are both wrong constructions

**More Attacks**
* Attacks on the implementation
  - Side channel attacks: time, power, ...
  - Fault attacks: computing errors in the last round exposes the secret key
* Linear and differential attacks
  - There's a dependence between message, ciphertext and the key bits
  - 5th S-box of DES it too close to a linear function
  - Success probability >= 97.7% given $$ 1/\epsilon^2 $$ random $$ (m,c) $$ pairs. For DES:
    * $$ \epsilon = 1/2^{21} $$. Can find 14 key bits this way in time $$ 2^{42} $$
    * The remaining 42 key bits can be found by brute force in $$ 2^{42} $$
    * In total $$ 2^{43} $$
* Quantum Attacks
  - Could solve generic search problem in $$ O(\lvert X \rvert^{1/2}) $$.

**AES**
* [Algorithm](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard#High-level_description_of_the_algorithm)
  - [Key schedule](https://en.wikipedia.org/wiki/AES_key_schedule): 11 round keys
  - SubBytes
    * ![SubBytes](https://upload.wikimedia.org/wikipedia/commons/a/a4/AES-SubBytes.svg){: width="400" }
  - ShiftRows
    * ![ShiftRows](https://upload.wikimedia.org/wikipedia/commons/6/66/AES-ShiftRows.svg){: width="450" }
  - MixColumns
    * ![MixColumns](https://upload.wikimedia.org/wikipedia/commons/7/76/AES-MixColumns.svg){: width="400" }
* Hardware
  - Intel Westmere
    * `aesenc`, `aesenclast`: one round of AES; 128 bit registers
    * `aeskeygenassist`: key expansion
  - AMD Bulldozer
* Attacks
  - Key recovery attack: 4x better than exhaustive search (e.g. 128 bit key -> $$ 2^{126} $$)
  - Related key attack: given $$ 2^{99} $$ in/out pairs from 4 related keys AES-256; recovery time: $$ \approx 2^{99} $$

[GGM (Goldreich-Goldwasser-Micali) PRF](https://crypto.stanford.edu/pbc/notes/crypto/ggm.html)
* Secure PRG $$ \Rightarrow $$ Secure PRF
* Not used in practice due to slow practice

**CPA Security**
* $$ m,c $$ pairs; $$ q $$ queries
* Suppose $$ E(k,m) $$ always outputs the same ciphertext for msg $$ m $$, it's CPA *insecure*. Solutions:
  - Randomized encryption
    * CT size = PT size + "# random bits"
  - Nonce-based encryption: `(k,n)` pair never used more than once. A nonce can be:
    * A counter (Stateful)
    * A random nonce (Stateless; Nonce space is sufficiently large)

**Modes of Operation**
* ECB
  - Not semantically secure if #blocks > 1
* CBC
  - Random IV
    * ![CBC Encryption](https://upload.wikimedia.org/wikipedia/commons/8/80/CBC_encryption.svg)
    * ![CBC Decryption](https://upload.wikimedia.org/wikipedia/commons/2/2a/CBC_decryption.svg)
    * $$ Adv_{CPA}[A,E_{CBC}] \le 2 \cdot Adv_{PRP}[B,E] + 2q^2L^2/\lvert X \rvert $$, where $$ q $$ = # messages encrypted with $$ k $$, $$ L $$ = message length (# blocks).
      - $$ qL $$ = message lengt in bits
    * CBC is only secure if $$ q^2L^2 \ll \lvert X \rvert $$
    * IV must be *unpredictable*
  - Nonce-based
    * `key = (k,k1)`
    * `E(k1,nonce) -> IV`
    * `(key,nonce)` pair must be *unique*
    * `k1 != k` (see [CBC1](https://web.cs.ucdavis.edu/~rogaway/papers/nonce.pdf#page=6))
  - Padding: [PKCS#7](https://en.wikipedia.org/wiki/Padding_(cryptography)#PKCS#5_and_PKCS#7)
    * Dummy block if multiple of block size
    * [Ciphertext stealing](https://en.wikipedia.org/wiki/Ciphertext_stealing) can avoid padding
* CTR (Counter Mode)
  - Turns a block cipher into stream cipher
  - Parallelizable
  - Deterministic: *One-time Key*
    * Stream cipher: $$ c[i] = m[i] \oplus F(k, i) $$, where $$ F $$ is a PRF
    * \$$ Adv_{SS}[A,E_{DETCTR}] = 2 \cdot Adv_{PRF}[B,F] $$
    * Secure PRF $$ \Rightarrow $$ $$ E_{DETCTR} $$ is sem. sec.
  - Randomized
    * \$$ c[i] = m[i] \oplus F(k, IV + i) $$
    * IV is chosen at random for every message
    * $$ Adv_{CPA}[A,E_{CTR}] \le 2 \cdot Adv_{PRF}[B,F] + 2q^2L/\lvert X \rvert $$, where $$ q $$ = # messages encrypted with $$ k $$, $$ L $$ = message length (# blocks).
    * CTR is only secure if $$ q^2L \ll \lvert X \rvert $$
      - Better than CBC
  - Nonce-based
    * IV = 64 bit nonce + 64 bit counter

```mermaid
graph LR
    PRP -- \subseteq --> PRF
    PRF -- Feistel --> PRP
    sPRP[Secure PRP] -- PRF Switching Lemma --> sPRF[Secure PRF]
    sPRF -- Luby-Rackoff Theorem --> sPRP
    sPRF -- DETCTR --> sPRG[Secure PRG]
    sPRG -- GGM --> sPRF
    sPRG --> ssStreamCipher[Sem. sec. Stream Cipher]
```

## Message Integrity

**MAC**
* Integrity, no confidentiality
* Signing: `S(k,m) -> t`
* Verifification: `V(k,m,t) -> 0,1`

**Secure MACs**
* *Chosen message attack*: given `q` `(m,t)` pairs, the attacker:
  - Cannot produce a valid tag for a new message
    * e.g. prevent CCA against Encrypt-then-MAC
  - Cannot produce `(m,t')` given `(m,t)`
* Secure PRF $$ \Rightarrow $$ Secure MAC
  - `S(k,m) := F(k,m)`
  - `V(k,m,t) := 1 if t = F(k, m), 0 otherwise`
  - \$$ Adv_{MAC}[A,I_F] \le Adv_{PRF}[B,F] + 1/\lvert Y \rvert $$
    * $$ I_F $$ is secure as long as $$ \lvert Y \rvert $$ is sufficiently large
  - Lemma: A MAC is secure if truncated to `w` bits and $$ 1/2^w $$ is still negligible

**Small-MAC -> Big-MAC**
* CBC-MAC (banking)
  - Commonly used as an AES-based MAC
    * CCM encryption mode
    * CMAC
* NMAC (Internet protocols)
  - Not usually used with AES or 3DES: need to change AES key on every block (re-computing AES key expansion)
  - HMAC

**Encrypted CBC-MAC (ECBC-MAC)**
* \$$ F: K \times X \rightarrow X $$
* ![ECBC-MAC](https://upload.wikimedia.org/wikipedia/commons/a/ae/CBC-MAC_%28encrypt_last_block%29_structure.svg)
* [Raw CBC-MAC](https://en.wikipedia.org/wiki/CBC-MAC) is not secure: Chosen message attack
  - Choose an arbitrary one-block message $$ m \in X $$
  - Request tag $$ t = F(k,m) $$
  - Output $$ t $$ as MAC forgery for the 2-block message $$ (m, t \oplus m) $$
* \$$ Adv_{PRF}[A,F_{ECBC}] \le Adv_{PRP}[B,F] + 2q^2/\lvert X \rvert $$
  - Secure as long as $$ q \ll \lvert X \rvert ^{1/2} $$

**NMAC (Nested MAC)**
* \$$ F: K \times X \rightarrow K $$
* [NMAC](https://cseweb.ucsd.edu/~mihir/papers/kmd5.pdf): $$ \text{NMAC}_K(x) = F_{k_1}(F_{k_2}(x)) $$
* Cascade function is not secure: Chosen message attack
  - \$$ cascade(k,m \parallel w) = F(cascade(m),w) $$
* \$$ Adv_{PRF}[A,F_{NMAC}] \le q \cdot L \cdot Adv_{PRP}[B,F] + q^2/2\lvert K \rvert $$
  - Secure as long as $$ q \ll \lvert K \rvert ^{1/2} $$

**Extension Property**
* For both ECBC-MAC and NMAC, $$ \forall x,y,w: F_{BIG}(k,x) = F_{BIG}(k,y) \Rightarrow F_{BIG}(k,x \parallel w) = F_{BIG}(k,y \parallel w) $$
* Attack: Issue $$ \lvert Y \rvert^{1/2} $$ to find a collision; *b-day paradox*
* The security bounds are *tight*

**MAC Padding**
* Must be invertible
* CBC-MAC: [Bit padding](https://en.wikipedia.org/wiki/ISO/IEC_9797-1#Padding_method_2)
  - Dummy block if multiple of block size

**CMAC**
* NIST SP 800-38B
* 3-key construction
* ![CMAC](https://i.stack.imgur.com/ISalk.png)
* No final encryption step (extension attack thwarted by last keyed xor)
* No dummy block
* \$$ Adv_{PRF}[A,F_{PMAC}] \le Adv_{PRF}[B,F] + 2q^2L^2/\lvert X \rvert $$
  - Secure as long as $$ qL \ll \lvert X \rvert ^{1/2} $$

**PMAC (Parallelizable MAC)**
* [PMAC](https://web.cs.ucdavis.edu/~rogaway/ocb/pmac.pdf)
* Gray codes $$ \gamma_i $$ are used to enforce order on message blocks
* Padding similar to CBC-MAC: no need for dummy block
* Incremental (i.e. we can quickly update the tag if one block changes) if PRF is also a PRP

**One-time MAC**
* Fast
* Example: $$ S(key,m) = P_m(k) + a (\mod q) $$, where $$ P_m(x) = \sum_{i=1}^{L}{m[i]x^i} $$, $$ key = (k,a) \in \\{1,2,\ldots, q\\}^2 $$

**Carter-Wegman MAC**
* One-time MAC $$ \Rightarrow $$ Many-time MAC
* Randomized MAC
* \$$ CW((k1,k2),m) = (r, F(k1,r) \oplus S(k2,m)) $$
  - CW is a secure MAC if $$ (S,V) $$ is a secure one-time MAC and $$ F $$ is a secure PRF.

|               | ECBC-MAC | CMAC | NMAC | HMAC | PMAC | Carter-Wegman MAC |
| ------------- | -------- | ---- | ---- | ---- | ---- | ----------------- |
| Property      | PRF      | PRF  | PRF  | PRF  | PRF  | Randomized MAC    |
| Parallizable? | No       | No   | No   | No   | Yes  | No                |

**MACs from Collision Resistance**
* \$$ S^{big}(k,m) = S(k,H(m)) $$
* \$$ V^{big}(k,m,t) = V(k,H(m),t) $$
* $$ I^{big} $$ is a secure MAC if $$ I $$ is a secure MAC and $$ H $$ is collision resistant.

**Birthday Paradox**
* \$$ n \approx 1.2\sqrt{B} \Rightarrow \Pr \le 1/2 $$
* Generic attack on collision resistant functions: time and space: $$ O(2^{n/2}) $$

**Merkle–Damgård Construction**
* Collision resistant: short message -> long message
* ![Merkle-Damgard](https://upload.wikimedia.org/wikipedia/commons/e/ed/Merkle-Damgard_hash_big.svg)
* Length padding: `10...0 || 64-bit message length`; possible dummy block
* `f` is compression function
  - If `f` is collision resistant, then so is `H`
  - Block Cipher
    * ![Davies-Meyer](https://upload.wikimedia.org/wikipedia/commons/5/53/Davies-Meyer_hash.svg){: width="150" }
      - Suppose `E` is an ideal cipher, then it takes $$ O(2^{n/2}) $$ evaluations to find a collision - best possible
      - $$ h(H,m) = E(m,H) $$ is not collision resistant -> $$ H'=D(m',E(m,H)) $$
      - Used by all SHA functions, e.g. SHA-256 with SHACAL-2 as the block cipher, and key size (block size) is 512-bit
    * 12 variants, e.g. Miyaguchi–Preneel (Whirlpool)
      - $$ h(H,m) = E(m,H) \oplus m $$ is insecure
  - Provable
    * Deiscrete log
    * Slow

**HMAC**
* `S(k,m) = H(k || m)` is insecure due to extension attack
* [Definition](https://en.wikipedia.org/wiki/HMAC#Definition)
* Similar to NMAC PRF; main difference: k1 and k2 are dependent
* Secure PRF if
  - Compression function is a PRF when dependent keys are used
  - \$$ q \ll \lvert T \rvert^{1/2} $$
* TLS: HMAC-SHA1-96 (HMAC doesn't require compression function to be collision resistant)
* Attacks:
  - Verification timing attacks: `==` byte-by-byte comparison and returns false when first inequality found
    * Defense #1: `res |= ord(x) ^ ord(y); return res == 0`; difficult to ensure due to compiler optimization
    * Defense #2: `mac = HMAC(k,m); return HMAC(k,mac) == HMAC(k,sig_bytes)`

## Authenticated Encryption

**Security**
* Sem. sec. under a CPA attack, and
* Ciphertext integrity

**Chosen Ciphertext Security**
* Sem. sec. under Both CPA and CCA
* CBC with random IV does not provide AE, because $$ D(k,\cdot) $$ never outputs $$ \perp $$.
* AE $$ \Rightarrow $$ CCA security
  - \$$ Adv_{CCA}[A,E] \le 2q \cdot Adv_{CI}[B_1,E] + Adv_{CPA}[B_2,E] $$
* Does not prevent replay attacks and side channels

|              | MAC-then-Encrypt                        | Encrypt-then-MAC | Encrypt-and-MAC                               |
| ------------ | --------------------------------------- | ---------------- | --------------------------------------------- |
| Application  | SSL                                     | IPSec            | SSH                                           |
| Secure?      | No (CCA)                                | Yes (AE)         | No (CPA; MAC doesn't provide confidentiality) |
| Construction | Rand-CTR or Rand-CBC                    | Always           | N/A                                           |
| Note         | One-time MAC is sufficient for Rand-CTR | N/A              | N/A                                           |

**AEAD**

```
                  |<---     encrypted      --->|
 ----------------------------------------------
| associated data |       encrypted data       |
 ----------------------------------------------
|<---            authenticated             --->|
```

|              | GCM               | CCM                        | EAX                        |
| ------------ | ----------------- | -------------------------- | -------------------------- |
| Type         | Encrypt-then-MAC  | MAC-then-Encrypt           | Encrypt-then-MAC           |
| Construction | CTR then CW-MAC   | CBC-MAC then CTR           | CTR then CMAC              |
| NIST?        | Yes               | Yes                        | No                         |
| Nonce-based? | Yes               | Yes                        | Yes                        |
| AEAD?        | Yes               | Yes                        | Yes                        |
| Code size    | Large (Non-Intel) | Smaller                    | Smaller                    |
| Speed        | Fast              | Slower                     | Slower                     |
| Note         | Intel `PCLMULQDQ` | Block cipher for MAC & Enc | Block cipher for MAC & Enc |

**OCB**
![OCB](https://web.cs.ucdavis.edu/~rogaway/ocb/faq1.gif)

**TLS 1.2**
* MAC-then-Encrypt
* Unidirectional keys
* Stateful encryption
* CBC AES-128, HMAC-SHA1
* 4 keys, e.g. $$ k_{b->s}=(k_{mac},k_{enc}) $$
* Attacks (Prior to TLS 1.1)
  - Predictable IV for CBC (chained IV): Not CPA secure; BEAST attack
  - Padding oracle: CBC only; CTR doesn't have padding
    * IMAP over TLS: query every 5 min

**802.11b WEP**
* Attack: CRC is linear

**SSH**
* Binary Packet Protocol
* Non-atomic decrypt
* Len field decrypted and used before it is authenticated

**Key Derivation**
* Extract pseudo-random key `k` from source key `SK`
  - Salt: a fixed non-secret string chosen at random
* Expand uniform `k`
  - \$$ KDF(k,CTX,L) = \parallel_{i = 0}^{L}F(k,(CTX \parallel i)) $$

**HKDF**
* Extract: $$ k \leftarrow HMAC(k=salt,data=SK) $$

**Password-Based KDF (PBKDF)**
* Deriving keys from passwords:
  - Do not use HKDF: passwords have insufficient entropy
  - Derived keys will be vulnerable to dictionary attacks
* Slow hash function: $$ H^{(c)}(pwd \parallel salt) $$

**Deterministic Encryption**
* Cannot be CPA secure
* Never encrypts same message twice
  - Choose message at random from a large message space
  - Message structure ensures uniqueness
* Deterministic CPA security
  - CBC with fixed IV is not det. CPA secure
* Synthetic IV (SIV)
  - $$ E_{det}((k1,k2),m) = E(k2,m;r \leftarrow F(k1,m)) $$, where $$ (E,D) $$ is CPA-secure and $$ F $$ is a secure PRF.
  - $$ E_{det} $$ is sem. sec. under det. CPA
  - Well suited for messages longer than ana AES block
  - Automatically ensures Deterministic Authenciated Encryption (DAE): det. CPA + ciphertext integrity
    * In decryption, apply the PRF to the decrypted message and verify it's identical to the IV
    * Secure PRF + CPA-secure CTR -> SIV-CTR provides DAE
* PRP
  - sem. sec. under det. CPA
  - Good for short messages (< 16 bytes); just use AES
  - Wide PRP
    * For long messages
    * [EME](https://www.cs.ucdavis.edu/~rogaway/papers/eme.pdf): a PRP on $$ \\{0,1\\}^N $$ for $$ N \gg n $$, where $$ n $$ is the size of PRP block
      - Secure
      - Parallelizable
      - 2x slower than SIV
  - PRP-based DAE
    * Append 0's to the LSB of the message
    * DAE if $$ 1/2^n $$ is negligible, where $$ n $$ is the count of appended 0's

**Disk encryption**
* Sectors on disk are fixed size
* No expansion ($$ \lvert M \rvert = \lvert C \rvert $$)
* Must use deterministic encryption; no integrity
* Det. CPA secure cipher with ($$ \lvert M \rvert = \lvert C \rvert $$) $$ \Rightarrow $$ PRP
  - \$$ PRP(k_t, sector_t) $$
  - \$$ k_t = PRF(k,t) $$
* Tweakable block ciphers
  - Construct many PRPs from a master key
  - $$ E(k,t,\cdot) $$ is invertable; indist. from random
  - Construction
    * $$ E_{tweak}(k,t,x) = E(E(k,t),x) $$, where $$ (E,D) $$ is a secure PRP, $$ E:K \times X \rightarrow X, K = X $$
      - `2n` evaluations of `E`
    * [XTS (XEX Tweakable Block Cipher with Ciphertext Stealing)](https://luca-giuzzi.unibs.it/corsi/Support/papers-cryptography/1619-2007-NIST-Submission.pdf)
      - `n + 1` evaluations of `E`
      - It is necessary to encrypt the tweak before using it
      - Block-level PRP, not sector-level
      - Mac OS X-Lion, TrueCrypt, BestCrypt

**Format Preserving Encryption (FPE)**
* Build a PRP on $$ \\{0,\ldots,s-1\\} $$ from a secure PRF $$ F:K \times \\{0,1\\}^n \rightarrow \\{0,1\\}^n $$, where $$ 0 \lt s \le 2^n $$
  - From $$ \\{0,1\\}^n $$ to $$ \\{0,1\\}^t $$, s.t. $$ 2^{t-1} \lt s \le 2^t $$
    * PRP on $$ \\{0,\ldots,s-1\\} $$
    * Truncate $$ F $$, $$ F': K \times \\{0,1\\}^{t/2} \rightarrow \\{0,1\\}^{t/2} $$
    * Patarin (7 rounds) is better than Luby-Rackoff
    * Security is the same as Patarin
  - From $$ \\{0,1\\}^t $$ to $$ \\{0,\ldots,s-1\\} $$
    * Given PRP $$ (E,D):K \times \\{0,1\\}^t \rightarrow \\{0,1\\}^t $$
    * Build $$ (E',D'): K \times \\{0,\ldots,s-1\\} \rightarrow \\{0,\ldots,s-1\\} $$: $$ x \in \\{0,\ldots,s-1\\} $$. $$ y \leftarrow x $$, do {$$ y \leftarrow E(k,y) $$} until $$ y \in \\{0,ldots,s-1\\} $$
    * Expected 2 iterations
    * Security is tight: $$ Adv_{PRP}[A,E] = Adv_{PRP}[B,E'] $$
  - No integrity

## Basic Key Exchange

Trusted 3rd Party: simple protocol; replay attack

**[Merkle's Puzzles](https://en.wikipedia.org/wiki/Merkle%27s_Puzzles)**
* Quadratic gap - best possible if ciphers are black box oracle

**Diffie-Hellman Protocol**
* [Overview](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange#General_overview)
* Best known algorithm (GNFS): $$ \exp(\tilde{O}(\sqrt[3]{n})) $$ (sub-exponential)
* Multi Party Key Agreement
  - n = 2: Diffie-Hellman
  - n = 3: Joux
  - n > 3: open question

**Arithmetic Algorithms**

For $$ n $$-bit integers:
* Addition and subtraction: $$ O(n) $$
* Multiplication
  - Naive: $$ o(n^2) $$
  - Karatsuba: $$ O(n^{1.585}) $$; 3 mults
  - Best (asymptotic) algorithm: $$ \tilde{O}(n\log(n)) $$; not practical
* Division with remainder: $$ O(n^2) $$
* Modualr exponentiation: successive square $$ O(n^2\log(n)) \le O(n^3) $$

## Public-Key Encrytion

**Security**
* `(G,E,D)`
* One-time security $$ \Rightarrow $$ Many-time security (CPA): attacker can encrypt any message with the public key
* IND-CCA

**Trapdoor Functions (TDF)**
* \$$ G,F,F^{-1} $$
  - \$$ G() \rightarrow (pk, sk) $$
  - \$$ F(pk,\cdot) $$
  - \$$ F^{-1}(sk,\cdot) $$
* Secure if $$ F $$ is a one-way function
* Public-key encryption
  - $$ (G,E,D) $$: ISO standard
  - $$ E(pk,m) $$: $$ x \xleftarrow{R} X $$, $$ y \leftarrow F(pk,x) $$, $$ k \leftarrow H(x) $$, $$ c \leftarrow E_s(k,m) $$; output $$ (y,c) $$
  - $$ D(sk,(y,c) $$: $$ x \leftarrow F^{-1}(sk,y) $$, $$ k \leftarrow H(x) $$, $$ m \leftarrow D_s(k,c) $$; output $$ m$$
  - Secure TDF + $$ (E_s,D_s) $$ auth. enc. + $$ H $$ is random oracle $$ \Rightarrow (G,E,D) $$ is $$ CCA^{ro} $$ secure
  - *Never* encrypt by applying $$ F $$ directly to plaintext! (e.g. Textbook RSA)
    * Deterministic
    * Many attacks exist

```
|<-- header -->|<---     body      --->|
 ---------------------------------------
|   F(pk,x)    |       Es(H(x),m)      |
 ---------------------------------------
```

**RSA Trapdoor Permutation**
* $$ G() $$: $$ p,q \approx 1024 $$ bits, $$ N = pq $$, $$ e \cdot d \equiv 1 \pmod{\varphi(N)}) $$; output $$ pk = (N,e) $$, $$ sk = (N,d) $$
* \$$ F(pk,x) = RSA(x) \equiv x^e \pmod{N} $$
* \$$ F^{-1}(sk,x) \equiv y^d \pmod{N} $$
* Attacks on textbook RSA
  - Exhausive search: if $$ k = k_1 \cdot k_2 $$ (prob. $$ \approx $$ 20%), $$ c/k_1^e \equiv k_2^e \pmod{N} $$

**PKCS #1**
* ISO standard is not often used
* E.g. preprocess a symmetric key $$ k $$ to 2048 bit then use RSA() to encrypt it
* [PKCS1 v1.5](https://datatracker.ietf.org/doc/html/rfc2313#section-8.1)
  - Bleichenbacher Attack
    * Test if the 16 MSBs of plaintext = `02`
    * \$$ c' \leftarrow r^e \cdot c = (r \cdot PKCS1(m))^e $$
    * HTTPS Defense (RFC 5246): return a random string `R` of 46 bytes if decryption fails
* PKCS1 v2.0: OAEP
  - ![OAEP RFC 8017](https://upload.wikimedia.org/wikipedia/commons/8/8f/OAEP_encoding_schema.svg){: width="400"}
  - Check pad on decryption
  - *RSA* is trapdoor permutation + MGFs are random oracles $$ \Rightarrow $$ RSA-OAEP is CCA secure
    * The theorem is false if you use general trapdoor permutation
  - OAEP+
    * General trapdoor permutation
    * During decryption validate $$ W(m,r) $$ field
  - SAEP+
    * RSA $$ e = 3 $$
    * One MGF
    * During decryption validate $$ W(m,r) $$ field

**RSA One-Way Function**
* Best known algorithm to compute e'th roots modulo $$ N $$
  - Step 1: factor $$ N $$ (hard)
  - Step 2: compute e'th roots modulo $$ p $$ and $$ q $$ (easy)
* Reduction: efficient algorithm for e'th roots mod $$ N $$ $$ \Rightarrow $$ efficient algorithm for factoring $$ N $$
  - Unknown
  - $$ e = 2 \Rightarrow $$ factoring $$ N $$, however, it can't be used in RSA
* Caveats
  - Wiener: if $$ d < N^{0.25} $$, then RSA is insecure
    * $$ \lvert e/N - k/d \rvert \le 1/2d^2 $$: difference is so small
    * Continued fraction algorithm to find $$ k/d $$; $$ e \cdot d \equiv 1 \pmod{k} \Rightarrow \gcd(k,d) = 1 $$
  - BD: if $$ d < N^{0.292} $$, then RSA is insecure. (Conjecture: $$ d < N^{0.5} $$)

**RSA in Practice**
* Use a small $$ e $$ to speed up RSA encryption
  - Minimum value: $$ e = 3 $$
  - Recommened: $$ e = 65537 = 2^{16} + 1 $$
* Asymmetry of RSA
  - Fast enc./slow dec.: 10~30:1
  - RSA-CRT: 4x dec., but still much slower than enc.
* Attacks
  - Timing attack (Kocher 97)
  - Power attack (Kocher 99)
  - Faults attack (BDL 97)
    * Defence: always check output (10% slowdown)
  - Low entropy at RSA key generation

**ElGamal Public-key System**
* `(Gen,E,D)`
* KeyGen: $$ g \xleftarrow{R} G $$, $$ a \xleftarrow{R} [0,n) $$; output $$ sk=a $$, $$ pk=(g,h=g^a) $$
* $$ E(pk=(g,h),m) $$: $$ b \xleftarrow{R} [0,n) $$, $$ u \leftarrow g^b $$, $$ v \leftarrow h^b $$, $$ k \leftarrow H(u,v) $$, $$ c \leftarrow E_s(k,m) $$; output $$ (u,c) $$
  - 2 exp. (fixed basis)
  - Can pre-compute (3x speed-up)
* $$ D(sk=a,(u,c)) $$: $$ v \leftarrow u^a $$, $$ k \leftarrow H(u,v) $$, $$ m \leftarrow D_s(k,c) $$; output $$ m $$
  - 1 exp. (variable basis)

```
|<- header ->|<---     body      --->|
 -------------------------------------
|     u      |       Es(H(x),m)      |
 -------------------------------------
```

**ElGamal Security**
* Computational Diffie-Hellman (CDH) Assumption
  - \$$ \Pr[A(g,g^a,g^b) = g^{ab}] < negligible $$
* Hash Diffie-Hellman (HDH) Assumption
  - \$$ (g,g^a,g^b,H(g^b,g^{ab})) \approx_{p} (g,g^a,g^b,R) $$
  - Slightly stronger: CDH is easy in $$ G \Rightarrow $$ HDH is easy in $$ (G,H) \enspace \forall H, \lvert Im(H) \rvert \ge 2 $$
  - ElGamal is sem. sec. under HDH
* Interactive Diffie-Hellman (IDH)
  - Stronger; needed to prove CCA security
  - Adv. can query $$ u_1,v_1 $$ and Chal. returns 1 if $$ (u_1)^a=v_1 $$
  - IDH + $$ (E_s,D_s) $$ auth. enc. + $$ H $$ random oracle $$ \Rightarrow $$ ElGamal is $$ CCA^{ro} $$ secure
* Prove CCA security based on CDH?
  - Option 1: use group $$ G $$ where CDH = IDH (e.g. bilinear group)
  - Option 2: twin ElGamal
    * KeyGen: $$ g \xleftarrow{R} G $$, $$ a1,a2 \xleftarrow{R} [0,n) $$; output $$ sk=(a1,a2), pk=(g,h_1=g^{a1},h_2=g^{a2}) $$
    * $$ E(pk=(g,h_1,h_2),m) $$: $$ b \xleftarrow{R} [0,n) $$, $$ k \leftarrow H(g^b,h_1^b,h_2^b) $$, $$c \leftarrow E_s(k,m) $$; output $$ (u=g^b,c) $$
    * $$ D(sk=(a1,a2),(u,c)) $$: $$ k \leftarrow H(u,u^{a1},u^{a2}) $$, $$ m \leftarrow D_s(k,c) $$; output $$ m $$
    * CDH + $$ (E_s,D_s) $$ auth. enc. + $$ H $$ random oracle $$ \Rightarrow $$ twin ElGamal is $$ CCA^{ro} $$ secure
    * Cost: one more exp. during enc./dec.
    * No one knows if it is worth it...
* Prove CCA security without random oracles
  - Option 1: use HDH in bilinear groups
  - Option 2: use Decision-DH assumption in any group
    * [Cramer-Shoup cryptogsystem](https://en.wikipedia.org/wiki/Cramer%E2%80%93Shoup_cryptosystem)

