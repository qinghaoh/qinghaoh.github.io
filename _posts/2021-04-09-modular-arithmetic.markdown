---
title:  "Modular Arithmetic"
category: algorithm
tags: math
---
# Euler's Theorem

In number theory, [Euler's theorem](https://en.wikipedia.org/wiki/Euler's_theorem) (also known as the Fermat–Euler theorem or Euler's totient theorem) states that if $$ n $$ and $$ a $$ are coprime positive integers, then $$ a $$ raised to the power of the totient of $$ n $$ is congruent to $$ 1 $$ modulo $$ n $$, or:

$$
a^{\varphi (n)} \equiv 1 \pmod{n}
$$

where $$ \varphi (n) $$ is Euler's totient function.

**Euler's totient function** counts the positive integers up to a given integer $$ n $$ that are relatively prime to $$ n $$.

If $$ n $$ is a prime:
* \$$ \varphi (n) = n - 1 $$
* \$$ a^{-1} \equiv a^{n-2} \pmod{n} $$

Multiplicative: if $$ \gcd(m, n) = 1 $$, then $$ \varphi (m) \varphi (n) = \varphi (mn) $$.

[Super Pow][super-pow]

```java
public int superPow(int a, int[] b) {
    // 1337 = 7 * 191
    // phi(1337) = phi(7) * phi(191) = 6 * 190 = 1140
    //
    // a ^ b mod 1337 = a ^ (b mod 1140) mod 1337
    int p = 0;
    for (int i : b) {
        p = (p * 10 + i) % 1140;
    }

    if (p == 0) {
        p += 1440;
    }
    return pow(a, p, 1337);
}

// 50. Pow(x, n)
private int pow(int a, int n, int mod) {
    a %= mod;
    long res = 1, al = a % mod;
    while (n != 0) {
        if (n % 2 == 1) {
            res = res * al % mod;
        }
        al = al * al % mod;
        n /= 2;
    }
    return (int)res;
}
```

$$
a^b \equiv a^{b \pmod{\varphi} + \varphi} \pmod{c}
$$

where $$ b>\varphi = \varphi(c) $$

A more straightforward solution:

```java
public int superPow(int a, int[] b) {
    final int mod = 1337;
    int res = 1;
    for (int i : b) {
        res = pow(res, 10, mod) * pow(a, i, mod) % mod;
    }
    return res;
}

// 50. Pow(x, n)
private int pow(int a, int n, int mod) {
}
```

# Group

[Multiplicative group of integers modulo n](https://en.wikipedia.org/wiki/Multiplicative_group_of_integers_modulo_n): the integers coprime (relatively prime) to $$ n $$ from the set $$ \\{0,1,\dots ,n-1\\} $$ of $$ n $$ non-negative integers form a group under multiplication modulo $$ n $$.

$$
\lvert(\mathbb {Z} /n\mathbb {Z} )^{\times }\rvert = \varphi (n)
$$

For prime $$ n $$ the group is cyclic.

Generator: $$ \langle g \rangle = \\{g^k \| k \in \mathbb{Z}\\} $$

[Lagrange's theorem](https://en.wikipedia.org/wiki/Lagrange%27s_theorem_(group_theory)): If $$ H $$ is a subgroup of a group $$ G $$, then

$$
\left|G\right|=\left[G:H\right]\cdot \left|H\right|
$$

# Pigeonhole Principle

[Modular arithmetic](https://en.wikipedia.org/wiki/Modular_arithmetic)

[Smallest Integer Divisible by K][smallest-integer-divisible-by-k]

Evaluate these remainders:

$$
1 \bmod k, 11 \bmod k, \cdots, \underbrace{11\cdots1}_{k} \bmod k
$$

* If any remainder is 0, then the smallest number of them is the result
* If none is 0, there must be dupliated remainders as per Pigeonhole Principle, as the $$ k $$ remainders can only take at most $$ k - 1 $$ different values excluding 0

In the second case, if $$ a_{i} \bmod k $$ has a duplicate $$ a_{j} \bmod k $$, since $$ a_{i + 1} = 10a_{i} + 1 $$, $$ a_{i + 1} \bmod k = a_{j + 1} \bmod k $$. Therefore, we will never see remainder = 0.

```java
public int smallestRepunitDivByK(int k) {
    if (k % 2 == 0 || k % 5 == 0) {
        return -1;
    }

    int r = 0;
    for (int n = 1; n <= k; n++) {
        r = (r * 10 + 1) % k;
        if (r == 0) {
            return n;
        }
    }
    return -1;
}
```

# Modular Inverse

$$ a $$ has modular inverse modulo $$ m $$ iff $$ gcd(a,m) = 1 $$. Modular inverse can be computed by [Extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm) in time $$ O(\log^2(m)) $$.


[Fancy Sequence][fancy-sequence]

```java
class Fancy {
    private static final int MOD = (int)1e9 + 7;
    private long[] arr = new long[100001];
    // a[i] * m + inc
    private long inc = 0, m = 1;
    private int size = 0;

    public Fancy() {

    }

    public void append(int val) {
        // a[i] * m + inc = val
        // a[i] = (val - inc) / m
        arr[size++] = (((val - inc + MOD) % MOD) * pow(m, MOD - 2, MOD)) % MOD;
    }

    public void addAll(int inc) {
        this.inc = (this.inc + inc) % MOD;
    }

    public void multAll(int m) {
        // (a[i] * m + inc) * m'
        // = a[i] * m * m' + inc * m'
        this.inc = (this.inc * m) % MOD;
        this.m = (this.m * m) % MOD;
    }

    public int getIndex(int idx) {
        return idx < size ? (int)((arr[idx] * m) % MOD + inc) % MOD : -1;
    }

    // 50. Pow(x, n)
    private int pow(long a, int n, int mod) {
    }
}
```

# Modular Kth Roots

To compute $$ k^{th} $$ roots modulo $$ m $$:

## Easy Case

If $$ gcd(b,m) = 1 $$ and $$ gcd(k,\varphi(m)) = 1 $$, the following steps find the congruence $$ x^{k} \equiv b \pmod{ma} $$[^1]:
1. Compute $$ \varphi(m) $$
1. Find positive integers $$ u $$ and $$ v $$ that satisfy $$ ku - \varphi(m)v = 1 $$
1. Compute $$ x \equiv b^u \pmod{m} $$ by successive squaring

As a special case: if $$ m $$ is a prime, $$ \forall c \in \mathbb {Z} /m\mathbb {Z} $$

$$
c^{1/k} \equiv c^d \pmod{m}
$$

where $$ d \equiv k^{-1} \pmod{m-1} $$.

## Quadratic Residue

$$
k = 2
$$.

[Euler's criterion](https://en.wikipedia.org/wiki/Euler%27s_criterion): Let $$ p $$ be an odd prime and $$ a $$ be an integer coprime to $$ p $$. Then

$$
a^{\tfrac {p-1}{2}}\equiv {
\begin{cases}
  1{\pmod {p}} & {\text{ if there is an integer }}x{\text{ such that }}x^{2}\equiv a{\pmod {p}}, \\
  -1{\pmod {p}} & {\text{ otherwise.}}
\end{cases}}
$$

Legendre symbol:

$$
\left({\frac {a}{p}}\right)\equiv a^{\tfrac {p-1}{2}}{\pmod {p}}
$$

Modulo an odd prime $$ p $$, there are $$ (p+1)/2 $$ quadratic residues.

[Tonelli-Shanks algorithm](https://zerobone.net/blog/math/tonelli-shanks/): $$ O(\log^2(p)) $$

Special case: if $$ p \equiv 3 \pmod{4} $$, quadratic residue is $$ a^{\frac {p+1}{4}}{\pmod {p}} $$.

## General Case

If $$ m $$ is composite number and $$ k \gt 1 $$, computing the root efficiently requires factorization of $$ m $$.

[fancy-sequence]: https://leetcode.com/problems/fancy-sequence/
[smallest-integer-divisible-by-k]: https://leetcode.com/problems/smallest-integer-divisible-by-k/
[super-pow]: https://leetcode.com/problems/super-pow/

[^1]: Joseph H. Silverman. (2009). *A Friendly Introduction to Number Theory*
