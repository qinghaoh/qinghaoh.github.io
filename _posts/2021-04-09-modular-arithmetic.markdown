---
layout: post
title:  "Modular Arithmetic"
tags: math
usemathjax: true
---
# Theorem

[Euler's theorem](https://en.wikipedia.org/wiki/Euler's_theorem)

In number theory, Euler's theorem (also known as the Fermat–Euler theorem or Euler's totient theorem) states that if \\(n\\) and \\(a\\) are coprime positive integers, then \\(a\\) raised to the power of the totient of \\(n\\) is congruent to one, modulo \\(n\\), or:

\\[a^{\varphi (n)} \equiv 1 \pmod{n}\\]

where \\(\varphi (n)\\) is Euler's totient function.

**Euler's totient function** counts the positive integers up to a given integer \\(n\\) that are relatively prime to \\(n\\).

If \\(n\\) is a prime
* \\(\varphi (n) = n - 1\\)
* \\(a^{-1} \equiv a^{n-2} \pmod{n}\\)

Multiplicative: if \\(\gcd(m, n) = 1\\), then \\(\varphi (m) \varphi (n) = \varphi (mn)\\).

[Super Pow][super-pow]

{% highlight java %}
public int superPow(int a, int[] b) {
    if (a % 1337 == 0) {
        return 0;
    }

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
    return power(a, p, 1337);
}

// 50. Pow(x, n)
private int power(int a, int n, int mod) {
    a %= mod;
    int result = 1;
    while (n != 0) {
        if (n % 2 == 1) {
            result = result * a % mod;
        }
        a = a * a % mod;
        n /= 2;
    }
    return result;
}
{% endhighlight %}

***Case 1***

If \\(\gcd(a, 1337) = 1\\),

\\[
\begin{equation} \label{eq:1}
a^b \bmod 1337 = a^{b \bmod \varphi(1337)} \bmod 1337 = a^{b \bmod 1140} \bmod 1337
\end{equation}
\\]

***Case 2***

If \\(a \bmod 7 = 0\\), let \\(a = 7^nm\\), \\(b = \varphi(1337)p + q\\), where \\(0 < q \le \varphi(1337)\\)

$$
\begin{aligned}
a^b \bmod 1337 &= (7^nm)^b \bmod 1337 \\
&= (7^{nb}m^b) \bmod 1337 \\
&= ((7^{nb} \bmod 1337) \cdot (m^b \bmod 1337)) \bmod 1337 \\
&= ((7^{n(\varphi(1337)p + q)} \bmod 1337) \cdot (m^{\varphi(1337)p + q} \bmod 1337)) \bmod 1337
\end{aligned}
$$

Since \\(\gcd(m, 1337) = 1\\), we know \\(m^{\varphi(1337)} \bmod 1337 = 1\\)

$$
\begin{aligned}
a^b \bmod 1337 = ((7^{1140np + nq} \bmod 1337) \cdot (m^q \bmod 1337)) \bmod 1337
\end{aligned}
$$

Cancellation of common terms: If \\(k a ≡ k b (mod kn)\\), then \\(a ≡ b (mod n)\\)

$$
\begin{aligned}
((7^{1140np + nq} \bmod 1337) \cdot (m^q \bmod 1337)) \bmod 1337 = (7 \cdot (7^{1140np + nq - 1} \bmod 191) \cdot (m^q \bmod 1337)) \bmod 1337
\end{aligned}
$$

Note \\(\gcd(7, 191) = 1\\), and \\(\varphi(191) = 190\\), so \\(7^{1140np} \bmod 191 = 7^{6 \cdot 190np} \bmod 191 = (7^6np)^{\varphi(191)} \bmod 191 = 0\\)

$$
\begin{aligned} \label{eq:2}
a^b \bmod 1337 &= (7 \cdot (7^{nq - 1} \bmod 191) \cdot (m^q \bmod 1337)) \bmod 1337 \\
&= ((7^{nq} \bmod 1337) \cdot (m^q \bmod 1337)) \bmod 1337 \\
&= 7^{nq}m^q \bmod 1337 \\
&= (7^nm)^q \bmod 1337 \\
&= a^q \bmod 1337 \\
&= a^{b \bmod 1140} \bmod 1337
\end{aligned}
$$

We can see \eqref{eq:1} and \eqref{eq:2} have the same format.

***Case 3***

If \\(a \bmod 191 = 0\\), it's similar to the case above.

#  Pigeonhole Principle

[Modular arithmetic](https://en.wikipedia.org/wiki/Modular_arithmetic)

[Smallest Integer Divisible by K][smallest-integer-divisible-by-k]

Evaluate these remainders:

\\[1 \bmod k, 11 \bmod k, \cdots, \underbrace{11\cdots1}_{k} \bmod k\\]

* If any remainder is 0, then the smallest number of them is the result
* If none is 0, there must be dupliated remainders as per Pigeonhole Principle, as the \\(k\\) remainders can only take at most \\(k - 1\\) different values excluding 0

In the second case, if \\(a_{i} \bmod k\\) has a duplicate \\(a_{j} \bmod k\\), since \\(a_{i + 1} = 10a_{i} + 1\\), \\(a_{i + 1} \bmod k = a_{j + 1} \bmod k\\). Therefore, we will never see remainder = 0.

{% highlight java %}
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
{% endhighlight %}

# Modular Inverse

[Fancy Sequence][fancy-sequence]

{% highlight java %}
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
        arr[size++] = (((val - inc + MOD) % MOD) * modPow(m, MOD - 2)) % MOD;
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

    private long modPow(long x, long y) {
        long res = 1;
        while (y > 0) {
            if ((y & 1) == 1) {
                res = res * x % MOD;
            }
            x = x * x % MOD;
            y >>= 1;
        }
        return res;
    }
}
{% endhighlight %}

[fancy-sequence]: https://leetcode.com/problems/fancy-sequence/
[smallest-integer-divisible-by-k]: https://leetcode.com/problems/smallest-integer-divisible-by-k/
[super-pow]: https://leetcode.com/problems/super-pow/
