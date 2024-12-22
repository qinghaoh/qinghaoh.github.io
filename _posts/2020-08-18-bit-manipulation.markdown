---
title:  "Bit Manipulation"
category: algorithm
tags: bit
---
## Properties
```
n ^ 0 = n
n ^ n = 0
2k ^ (2k + 1) = 1

n & -n 		// clears all but rightmost set bit
n & (n - 1)		// zeros out rightmost set bit, Brian Kernighan's Algorithm
(n & (n - 1)) == 0  	// power of 2

// enumerates all submasks
for (int s = mask; s > 0; s = (s - 1) & mask)
```

### Brian Kernighan's Algorithm

[Counting Bits][counting-bits]

```java
public int[] countBits(int num) {
    int[] result = new int[num + 1];
    for (int i = 1; i <= num; i++) {
        result[i] = result[i & (i - 1)] + 1;
    }
    return result;
}
```

[Bitwise AND of Numbers Range][bitwise-and-of-numbers-range]

```c++
int rangeBitwiseAnd(int left, int right) {
    while (right > left) {
        right &= right - 1;
    }
    return right;
}
```

[Concatenation of Consecutive Binary Numbers][concatenation-of-consecutive-binary-numbers]

```java
private static final int MOD = (int)1e9 + 7;

public int concatenatedBinary(int n) {
    long sum = 0;
    int length = 0;
    for (int i = 1; i <= n; i++) {
        // power of 2
        if ((i & (i - 1)) == 0) {
            length++;
        }
        sum = ((sum << length) | i) % MOD;
    }
    return (int)sum;
}
```

[Binary Number with Alternating Bits][binary-number-with-alternating-bits]

```java
public boolean hasAlternatingBits(int n) {
    return (n & (n >> 1)) == 0 && (n | (n >> 2)) == n;
}
```

`(n & (n >> 1)) == 0` ensures no consecutive 1's.

[Minimum Operations to Reduce an Integer to 0][minimum-operations-to-reduce-an-integer-to-0]

```java
public int minOperations(int n) {
    // if number of consecutive 1's == 1, +1
    // if number of consecutive 1's > 1, +2
    return Integer.bitCount(n ^ (n * 3));
}
```

[XOR Operation in an Array][xor-operation-in-an-array]

```java
public int xorOperation(int n, int start) {
    // nums[i] = start + 2 * i
    //  right shift each element
    //  nums[i] = start / 2 + i
    //
    // Igore LBS for now
    int xor = 2 * xorRightShift(n, start / 2);

    // If start is odd, then all elements are odd
    //  and if hte number of odd elements is odd
    //  then LSB == 1
    if (n % 2 == 1 && start % 2 == 1) {
        xor++;
    }

    return xor;
}

// nums[i] = start + i
private int xorRightShift(int n, int start) {
    // Let a_0 = start, a_i = nums[i]
    //  if n is even
    //   xor == a_0 ^ a_1 ^ a_2 ^ ... ^ a_(n - 1)
    //       == ((a_0 - 1) ^ (a_0 - 1)) ^ a_0 ^ a_1 ^ a_2 ^ ... ^ a_(n - 1)
    return start % 2 == 0 ? xorEvenStart(n, start) : (start - 1) ^ xorEvenStart(n + 1, start - 1);
}

/**
 * nums[i] = start + i
 * start % 2 == 0
 * We take use of the property: start ^ (start + 1) == 1
 */
private int xorEvenStart(int n, int start) {
    // Let a_0 = start, a_i = nums[i]
    //  if n is even
    //   xor == a_0 ^ a_1 ^ a_2 ^ ... ^ a_(n - 1)
    //       == (a_0 ^ a_1) ^ (a_2 ^ a_3) ^ ... ^ (a_(n - 2), a_(n - 1))
    //       == 1 ^ 1 ^ ... ^ 1
    //   where the number of 1's is n / 2
    //     if n / 2 is even, xor == 0; else xor == 1
    //
    // if n is odd
    //   xor == a_0 ^ a_1 ^ a_2 ^ ... ^ a_(n - 1)
    //       == (a_0 ^ a_1) ^ (a_2 ^ a_3) ^ ... ^ (a_(n - 3), a_(n - 2)) ^ a_(n - 1)
    //       == 1 ^ 1 ^ ... ^ 1 ^ a_(n - 1)
    //   where the number of 1's is n / 2
    //     if n / 2 is even, xor == 0 ^ a_(n - 1); else xor == 1 ^ a_(n - 1)
    //      a_(n - 1) == nums[i] == start + n - 1
    return n % 2 == 0 ? (n / 2) & 1 : ((n / 2) & 1) ^ (start + n - 1);
}
```

[Total Hamming Distance][total-hamming-distance]

```java
public int totalHammingDistance(int[] nums) {
    int count = 0;
    for (int i = 0; i < 31; i++) {
        int ones = 0;
        for (int num : nums) {
            if ((num & (1 << i)) != 0) {
                ones++;
            }
        }
        count += ones * (nums.length - ones);
    }
    return count;
}
```

[K-th Symbol in Grammar][k-th-symbol-in-grammar]

```java
public int kthGrammar(int N, int K) {
    // K is in [1, 2 ^ (N - 1)], so we can ignore N
    //
    // if k is 0 indexed
    // if f(k) == 0
    //   then f(2 * k) == 0, f(2 * k + 1) == 1
    // else if f(k) == 1
    //   then f(2 * k) == 1, f(2 * k + 1) == 0
    //
    // so f(2 * k) == f(k) ^ 0, f(2 * k + 1) == f(k) ^ 1
    //
    // f(10110)
    //   = f(1011) ^ 0
    //   = f(101) ^ 1 ^ 0
    //   = f(10) ^ 1 ^ 1 ^ 0
    //   = f(1) ^ 0 ^ 1 ^ 1 ^ 0
    //   = f(0) ^ 1 ^ 0 ^ 1 ^ 1 ^ 0
    //   = 1 ^ 0 ^ 1 ^ 1 ^ 0
    return Integer.bitCount(K - 1) & 1;
}
```

[Single Number][single-number]

```java
public int singleNumber(int[] nums) {
    int a = 0;
    for (int num : nums) {
      a ^= num;
    }
    return a;
}
```

[Single Number II][single-number-ii]

Generalization: every element appears `k (k > 1)` times except for one.

Solution I:

```java
public int singleNumber(int[] nums) {
    return singleNumber(nums, 3);
}

private int singleNumber(int[] nums, int k) {
    int result = 0;
    for (int i = 0; i < 32; i++) {
        // counts 1's at each bit
        int sum = 0;
        for (int num : nums) {
            sum += (num >> i) & 1;
        }
        sum %= 3;
        result |= sum << i;
    }
    return result;
}
```

@fun4LeetCode has a great [article](https://leetcode.com/problems/single-number-ii/discuss/43295/Detailed-explanation-and-generalization-of-the-bitwise-operation-method-for-single-numbers) about a futhur generaliztion, where the excepted number appears `m` times (`m > 0` and `m % k > 0`). That's a bit over complicated, so I'm just archiving the link here.

Solution II: [Karnaugh map](https://en.wikipedia.org/wiki/Karnaugh_map)

To cover `k` counts, we require `2 ^ n >= k`, where `n` is the total number of bits. Therefore, `n >= log(k)`. In this problem, `k == 3`, so the complete transition loop of the counter is `00 -> 01 -> 10 -> 00 -> ...`.

![Karnaugh map](/assets/img/algorithm/karnaugh_map.png)

[Karnaugh map tool](https://www.charlie-coleman.com/experiments/kmap/)

```java
public int singleNumber(int[] nums) {
    int n0 = 0, n1 = 0;
    for (int num : nums) {
        int tmp = n0;
        n0 = (n0 & ~num) | (~n0 & ~n1 & num);
        n1 = (tmp & num) | (n1 & ~num);
    }
    return n0;
}
```

[Single Number III][single-number-iii]

```java
public int[] singleNumber(int[] nums) {
    int lsb = 0;
    for (int num : nums) {
        lsb ^= num;
    }

    // the two elements are distinct, so lsb != 0
    lsb &= -lsb;

    // partitions the numbers based on lsb
    int[] result = {0, 0};
    for (int num : nums) {
        if ((num & lsb) == 0) {
            result[0] ^= num;
        } else {
            result[1] ^= num;
        }
    }
    return result;
}
```

## Bit Count

[Minimum Operations to Make the Integer Zero][minimum-operations-to-make-the-integer-zero]

```java
public int makeTheIntegerZero(int num1, int num2) {
    long num = num1;
    // num = num1 - k * num2 = 2 ^ i_1 + 2 ^ i_2 + ... + 2 ^ i_k
    int k = 0;
    while (true) {
        // the operations to subtract powers of two from `num` is in the range
        // [num.bitCount, num]: binary representation -> all ones
        if (Long.bitCount(num) <= k && k <= num) {
            return k;
        }

        // num >= 0 otherwise the msb will always be 1 and can't be removed
        if (num2 > 0 && num < 0) {
            return -1;
        }

        num -= num2;
        k++;
    }
}
```

[Maximum Number That Sum of the Prices Is Less Than or Equal to K][maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k]

Credits to @votrubac.

```c++
// @param fixedBits: carry of number of fixed bits from the previous recursion level.
long long findMaximumNumber(long long k, int fixedBits, int x) {
    if (k < fixedBits) {
        return 0;
    }

    // Denote the sum of prices of [1, 2 ^ i) as f(i).
    // The goal in this recursion level is to find the largest i such that f(i) <= k
    // (we use the variable `p` to represent one-indexed i).
    //
    // If x = 1, then f(0) = 0, f(1) = 1, f(2) = 4, f(3) = 12, ...
    // f(i) = 2 * f(i - 1) + 2 ^ (i - 1) <-- addend
    //      = 2 ^ i * f(0) + 2 ^ (i - 1) * i
    //
    // e.g. f(3):
    //       ----- msb = 0, count 4 = f(2)
    //       0 0 1   
    //       0 1 0
    //       0 1 1
    //       ----- msb = 1, count = 8 = f(2) + 4 = f(2) + 2 ^ 2 <-- addend
    //       1 0 0
    //       1 0 1
    //       1 1 0
    //       1 1 1
    //
    // `addend` needs to be added as long as p % x == 0.
    //
    // `fixedBits` is the number of bits that are computed and fixed by previous recursions.
    // These bits are more significant than the targeted `i` in this recursion level.
    // e.g. k = 40, x = 1
    // With one recursion level, i = 4, f(i) = 32, and the remaining value is 40 - 32 = 8.
    // In the second recursion level, the 4-th bit is fixed, and we need to find the largest j
    // such that j < i, and g(j) = f(j) + count of fixed bits <= 8.
    // It turns out j = 2, and the binary representations are: 10000, 10001, 10010, 10011
    //
    // If `fixedBits` = m, the corrected function is:
    // g(i) = f(i) + 2 ^ i * m
    //      = 2 ^ i * f(0) + 2 ^ (i - 1) * i + 2 ^ i * m
    //      = 2 ^ i * (f(0) + m) + 2 ^ (i - 1) * i
    // The only difference between f(i) and g(i) is the initial value.
    // By setting f(0) = m, we convert f(i) to g(i).
    //
    // For x > 1, tracks the position p of the rightmost bit
    // and adds it only if p % x == 0.
    long long addend = 1, sumOfPrices = fixedBits, p = 1;
    while (2 * sumOfPrices + (p % x == 0 ? addend : 0) <= k) {
        sumOfPrices = 2 * sumOfPrices + (p % x == 0 ? addend : 0);
        addend *= 2;
        p++;
    }

    // The target bit p is fixed in the next recursion level if p % x == 0.
    return findMaximumNumber(k - sumOfPrices, fixedBits + (p % x == 0), x) + addend;
}

public:
long long findMaximumNumber(long long k, int x) {
    return findMaximumNumber(k, 0, x) - 1;
}
```

## Bitwise Operators

### And

[Find a Value of a Mysterious Function Closest to Target][find-a-value-of-a-mysterious-function-closest-to-target]

```java
public int closestToTarget(int[] arr, int target) {
    int n = arr.length, min = Integer.MAX_VALUE;

    // ands[i]: unique AND values of arr[i...]
    // the size of ands[i] is at most ceil(log(arr[i])), i.e. arr[i].bitCount()
    Set<Integer>[] ands = new Set[n];
    for (int i = 0; i < n; i++) {
        ands[i] = new HashSet<>();
    }

    ands[n - 1].add(arr[n - 1]);

    // computes ands[i] from ands[i + 1]
    for (int i = n - 2; i >= 0; i--) {
        ands[i].add(arr[i]);
        for (int v: ands[i + 1]) {
            ands[i].add(arr[i] & v);
        }
    }

    for (var a : ands) {
        for (int v : a) {
            min = Math.min(min, Math.abs(v - target));
        }
    }

    return min;
}
```

[Split Array Into Maximum Number of Subarrays][split-array-into-maximum-number-of-subarrays]

```c++
int maxSubarrays(vector<int>& nums) {
    // Each subarray has score 0
    int ways = 0, curr = 0;
    for (int num : nums) {
        curr = curr == 0 ? num : curr & num;
        ways += curr == 0;
    }

    // 1 means no split
    // & monitonically decreases
    // so the score(any subarray) >= score(nums)
    // If score(nums) is non-zero, the sum of score(subarrays) must be greater than score(num)
    return max(ways, 1);
}
```

### Exclusive Or

Distributive Property:

```
(a + b) * (c + d) = (a * c) + (a * d) + (b * c) + (b * d)
(a ^ b) & (c ^ d) = (a & c) ^ (a & d) ^ (b & c) ^ (b & d)
```

[Maximum Xor Product][maximum-xor-product]

```c++
int maximumXorProduct(long long a, long long b, int n) {
    const int mod = 1e9 + 7;
    if (n) {
        // Consider the i-th bit.
        // If a[i] == b[i], x[i] should make (a ^ x)[i] == (b ^ x)[i] == 1
        // Otherwise,
        //   if x[i] == 0, a and b don't change
        //   otherwise, x[i] transfers bit 1 from one number to the other
        //
        // Denote a ^ x as a', b ^ x as b'
        // In either case, a' + b' = a + b remains unchanged
        // so we should make a' and b' as close as possible to maximize a' * b'
        for (long long bt = 1ll << (n - 1); bt > 0; bt >>= 1) {
            if ((min(a, b) & bt) == 0) {
                a ^= bt;
                b ^= bt;
            }
        }
    }
    return a % mod * (b % mod) % mod;
}
```

#### MSB -> LSB

Process the numbres bit by bit from msb to lsb.

[Maximum XOR of Two Numbers in an Array][maximum-xor-of-two-numbers-in-an-array]

```java
public int findMaximumXOR(int[] nums) {
    int max = 0, mask = 0;
    // Builds the max xor bit by bit from msb to lsb.
    for (int i = 31; i >= 0; i--) {
        // Most significant (32 - i) bits
        mask = mask | (1 << i);

        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num & mask);
        }

        int candidate = max | (1 << i);

        // Finds a and b in the set so that a ^ b == candidate
        // => b == candidate ^ a
        // If there's no such (a, b), `max` remains the same,
        // i.e. the bit at this index in `max` is 0
        for (int a : set) {
            if (set.contains(candidate ^ a)) {
                max = candidate;
                break;
            }
        }
    }
    return max;
}
```

#### Trie

It's more intuitive to process the numbers bit by bit.

[Maximum XOR of Two Numbers in an Array][maximum-xor-of-two-numbers-in-an-array]

```java
public int findMaximumXOR(int[] nums) {
    TrieNode root = new TrieNode();
    int max = 0;
    for (int num : nums) {
        TrieNode node = root;
        int xor = 0;

        // Searches for such an existing number x in the trie that makes
        // `xor` = num ^ x the max so far.
        for (int i = 31; i >= 0; i--) {
            // num[i]: the i-th bit of num
            int b = (num >> i) & 1;
            // v = xor[i]
            // First, checks whether there is an existing number that makes xor[i] = 1.
            // Otherwise, unless the trie is empty, there is an existing number that makes xor[i] = 0.
            for (int v = 1; v >= 0; v--) {
                if (node.children[b ^ v] != null) {
                    // There exists an number in the trie whose i-th bit is num[i] ^ v
                    node = node.children[b ^ v];
                    // xor[i] = v
                    xor = (xor << 1) + v;
                    break;
                }
            }
        }

        // Inserts the num into the trie
        node = root;
        for (int i = 31; i >= 0; i--) {
            int b = (num >> i) & 1;
            if (node.children[b] == null) {
                node.children[b] = new TrieNode();
            }
            node = node.children[b];
        }
        max = Math.max(max, xor);
    }
    return max;
}

class TrieNode {
    TrieNode[] children = new TrieNode[2];
}
```

A more compact approach is as follows, which inserts the node and finds the optimal complement in one loop:

```java
public int findMaximumXOR(int[] nums) {
    TrieNode root = new TrieNode();
    int max = 0;
    for (int num : nums) {
        // num ^ complement yields max possible value
        TrieNode node = root, complement = root;
        int xor = 0;

        for (int i = 31; i >= 0; i--) {
            // num[i]: the i-th bit of num
            int b = (num >> i) & 1;

            // Inserts node into the trie
            if (node.children[b] == null) {
                node.children[b] = new TrieNode();
            }
            node = node.children[b];

            if (complement.children[b ^ 1] != null) {
                // There exists an element in the trie whose i-th bit is 1 - num[i]
                complement = complement.children[b ^ 1];
                // xor[i] = 1
                xor += 1 << i;
            } else {
                // In this case, complement.children[b] != null because num is already inserted.
                // Otherwise, xor[i] = 0
                complement = complement.children[b];
            }
        }
        max = Math.max(max, xor);
    }
    return max;
}

class TrieNode {
    TrieNode[] children = new TrieNode[2];
}
```

[Count Pairs With XOR in a Range][count-pairs-with-xor-in-a-range]

```java
private static final int NUM_BITS = 15;

public int countPairs(int[] nums, int low, int high) {
    TrieNode root = new TrieNode();
    int lowCount = 0, highCount = 0;
    for (int num : nums) {
        lowCount += countSmallerPairs(root, num, low);
        highCount += countSmallerPairs(root, num, high + 1);
        insert(root, num);
    }
    return highCount - lowCount;
}

// counts elements in the trie that xor num < x
private int countSmallerPairs(TrieNode root, int num, int x) {
    TrieNode node = root;
    int count = 0;
    for (int i = NUM_BITS - 1; i >= 0 && node != null; i--) {
        int a = (num >> i) & 1, b = (x >> i) & 1;

        // compares the i-th bits of num and x
        if (b == 0) {
            // finds the bit == a, so they xor to 0
            node = node.children[a];
        } else {
            if (node.children[a] != null) {
                // finds the bit == a, so they xor to 0
                // so the xor < x
                count += node.children[a].count;
            }
            // keeps searching
            node = node.children[1 - a];
        }
    }
    return count;
}

class TrieNode {
    TrieNode[] children;
    int count;

    TrieNode() {
        children = new TrieNode[2];
        count = 0;
    }
}

private void insert(TrieNode root, int num) {
    TrieNode node = root;
    for (int i = NUM_BITS - 1; i >= 0; i--) {
        int b = (num >> i) & 1;
        if (node.children[b] == null) {
            node.children[b] = new TrieNode();
        }
        node = node.children[b];
        node.count++;
    }
}
```

Simplied version: no Trie, but similarly, level-traverse all the numbers

```java
public int countPairs(int[] nums, int low, int high) {
    return countSmallerPairs(nums, high + 1) - countSmallerPairs(nums, low);
}

// it's a variant of the trie solution
// the search starts from leaves
// counts pairs in nums that xor < x
private int countSmallerPairs(int[] nums, int x) {
    Map<Integer, Long> count = Arrays.stream(nums).boxed()
        .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
    Map<Integer, Long> count2 = new HashMap<>();

    int pairs = 0;
    // iterates through each bit of x, from lsb to msb
    while (x > 0) {
        for (int k : count.keySet()) {
            // counts next level of nums to check
            // by right shifting all nums
            long v = count.get(k);
            count2.put(k >> 1, count2.getOrDefault(k >> 1, 0l) + v);

            // looks for pairs that, after XORing, have the same bits to the left
            // but have a 0 instead of a 1 at lsb.
            // if x & 1 == 0, then there can be no such pairs
            // if lsb == 1
            if ((x & 1) > 0) {
                // k ^ (x - 1) ^ k == x - 1 < x
                pairs += v * count.getOrDefault((x - 1) ^ k, 0l);
            }
        }
        count = count2;
        count2 = new HashMap<>();
        x >>= 1;
    }

    // i < j
    return pairs / 2;
}
```

#### Backtracking/DFS

[Maximum Genetic Difference Query][maximum-genetic-difference-query]

```java
private static final int NUM_BITS = 17;  // 10 ^ 5
private TrieNode trieRoot = new TrieNode();
private List<Integer>[] tree, queryIndexes;
private int[][] queries;

public int[] maxGeneticDifference(int[] parents, int[][] queries) {
    int n = parents.length, m = queries.length, root = -1;
    // builds tree
    this.tree = new List[n];
    for (int i = 0; i < n; i++) {
        tree[i] = new ArrayList<>();
    }
    for (int i = 0; i < n; i++) {
        if (parents[i] == -1) {
            root = i;
        } else {
            tree[parents[i]].add(i);
        }
    }

    this.queries = queries;
    this.queryIndexes = new List[n];
    for (int i = 0; i < n; i++) {
        queryIndexes[i] = new ArrayList<>();
    }
    for (int i = 0; i < m; i++) {
        queryIndexes[queries[i][0]].add(i);
    }

    int[] ans = new int[m];
    backtrack(root, ans);
    return ans;
}

private void backtrack(int node, int[] ans) {
    // adds the node to the trie
    trieRoot.insert(node, true);

    for (int i : queryIndexes[node]) {
        ans[i] = trieRoot.maxDiff(queries[i][1]);
    }

    for (int child: tree[node]) {
        backtrack(child, ans);
    }

    // removes the node from the trie (by decrementing the counter of each node on the path)
    trieRoot.insert(node, false);
}

class TrieNode {
    TrieNode[] child = new TrieNode[2];
    int countOfTreeNodes = 0; // count of tree nodes go through this trie node

    public void insert(int number, boolean toAdd) {
        TrieNode node = this;
        for (int i = NUM_BITS; i >= 0; i--) {
            int bit = (number >> i) & 1;
            if (node.child[bit] == null) {
                node.child[bit] = new TrieNode();
            }
            node = node.child[bit];
            node.countOfTreeNodes += toAdd ? 1 : -1;
        }
    }

    public int maxDiff(int val) {
        TrieNode node = this;
        int ans = 0;
        for (int i = NUM_BITS; i >= 0; i--) {
            int bit = (val >> i) & 1;
            if (node.child[1 - bit] != null && node.child[1 - bit].countOfTreeNodes > 0) {
                node = node.child[1 - bit];
                ans |= (1 << i);
            } else {
                node = node.child[bit];
            }
        }
        return ans;
    }
}
```

## Hybrid

[Apply Operations on Array to Maximize Sum of Squares][apply-operations-on-array-to-maximize-sum-of-squares]

`(a, b) -> (a & b, a | b)`

```
(1, 0) -> (0, 1)  // bit transferred
// all the other pairs
(a, b) -> (a, b)
```

## Gray Code

[Gray code](https://en.wikipedia.org/wiki/Gray_code): an ordering of the binary numeral system such that two successive values differ in only one bit (binary digit).

Formula:

```java
int g(int n) {
    return n ^ (n >> 1);
}
```

[Circular Permutation in Binary Representation][circular-permutation-in-binary-representation]

```java
public List<Integer> circularPermutation(int n, int start) {
    List<Integer> list = new ArrayList<>();
    for (int i = 0; i < (1 << n); i++) {
        list.add(start ^ i ^ (i >> 1));
    }
    return list;
}
```

### Inverse Gray Code

[Minimum One Bit Operations to Make Integers Zero][minimum-one-bit-operations-to-make-integers-zero]

```java
public int minimumOneBitOperations(int n) {
    // oeis A006068
    // inverse Gray code
    int count = 0;
    while (n > 0) {
        count ^= n;
        n >>= 1;
    }
    return count;
}
```

[Flip Columns For Maximum Number of Equal Rows][flip-columns-for-maximum-number-of-equal-rows]

```java
public int maxEqualRowsAfterFlips(int[][] matrix) {
    Map<String, Integer> map = new HashMap<>();
    for (int[] row : matrix) {
        // Flipping a subset of columns is like doing a bitwise XOR of some number k onto each row
        // if row ^ k == 0 or row ^ k == 1
        // then k == row or k == row ^ 1
        StringBuilder sb1 = new StringBuilder(), sb2 = new StringBuilder();
        for (int e : row) {
            sb1.append(e);
            sb2.append(e ^ 1);
        }
        map.compute(sb1.toString(), (k, v) -> v == null ? 1 : v + 1);
        map.compute(sb2.toString(), (k, v) -> v == null ? 1 : v + 1);
    }
    return Collections.max(map.values());
}
```

[Find Root of N-Ary Tree][find-root-of-n-ary-tree]

```java
public Node findRoot(List<Node> tree) {
    // visits all nodes
    // the root node would be the only node that is visited once
    // the rest of the nodes would be visited twice.
    int xor = 0;
    for (Node node : tree) {
        xor = xor ^ node.val;
        for (Node child : node.children) {
            xor = xor ^ child.val;
        }
    }

    for (Node node : tree) {
        if (node.val == xor) {
            return node;
        }
    }
    return null;
}
```

[Integer Replacement][integer-replacement]

```java
public int integerReplacement(int n) {
    int count = 0;
    // not n > 1, because of Integer.MAX_VALUE
    while (n != 1) {
        if (n % 2 == 0) {
            n >>>= 1;
        } else {
            // checks the last two digits
            if (n == 3 || (n & 3) == 1) {
                n--;
            } else {
                n++;
            }
        }
        count++;
    }
    return count;
}
```

[Missing Number][missing-number]

```java
public int missingNumber(int[] nums) {
    int missing = nums.length;
    for (int i = 0; i < nums.length; i++) {
        missing ^= i ^ nums[i];
    }
    return missing;
}
```

[UTF-8 Validation][utf-8-validation]

```java
public boolean validUtf8(int[] data) {
    int count = 0;
    for (int d : data) {
        d = d & 255;
        if (count == 0) {
            if ((d >> 5) == 0b110) {
                count = 1;
            } else if ((d >> 4) == 0b1110) {
                count = 2;
            } else if ((d >> 3) == 0b11110) {
                count = 3;
            } else if ((d >> 7) != 0) {
                return false;
            }
        } else {
            if ((d >> 6) != 0b10) {
                return false;
            }
            count--;
        }
    }
    return count == 0;
}
```

[Maximum Length of a Concatenated String with Unique Characters][maximum-length-of-a-concatenated-string-with-unique-characters]

```c++
int maxLength(vector<string>& arr) {
    {% raw %}vector<bitset<26>> dp{{}};{% endraw %}
    int res = 0;
    for (const string& s : arr) {
        bitset<26> mask;
        for (const char& ch : s) {
            mask.set(ch - 'a');
        }

        int n = mask.count();
        // Duplicates exist; skip
        if (n < s.size()) {
            continue;
        }

        for (int i = dp.size() - 1; i >= 0; i--) {
            // An visited element that has no overlap with mask is a candidate
            if ((dp[i] & mask).none()) {
                dp.push_back(dp[i] | mask);
                res = max(res, static_cast<int>(dp[i].count()) + n);
            }
        }
    }
    return res;
}
```

[apply-operations-on-array-to-maximize-sum-of-squares]: https://leetcode.com/problems/apply-operations-on-array-to-maximize-sum-of-squares/
[binary-number-with-alternating-bits]: https://leetcode.com/problems/binary-number-with-alternating-bits/
[bitwise-and-of-numbers-range]: https://leetcode.com/problems/bitwise-and-of-numbers-range/
[concatenation-of-consecutive-binary-numbers]: https://leetcode.com/problems/concatenation-of-consecutive-binary-numbers/
[circular-permutation-in-binary-representation]: https://leetcode.com/problems/circular-permutation-in-binary-representation/
[count-pairs-with-xor-in-a-range]: https://leetcode.com/problems/count-pairs-with-xor-in-a-range/
[counting-bits]: https://leetcode.com/problems/counting-bits/
[find-a-value-of-a-mysterious-function-closest-to-target]: https://leetcode.com/problems/find-a-value-of-a-mysterious-function-closest-to-target/
[find-root-of-n-ary-tree]: https://leetcode.com/problems/find-root-of-n-ary-tree/
[flip-columns-for-maximum-number-of-equal-rows]: https://leetcode.com/problems/flip-columns-for-maximum-number-of-equal-rows/
[integer-replacement]: https://leetcode.com/problems/integer-replacement/
[k-th-symbol-in-grammar]: https://leetcode.com/problems/k-th-symbol-in-grammar/
[maximum-genetic-difference-query]: https://leetcode.com/problems/maximum-genetic-difference-query/
[maximum-length-of-a-concatenated-string-with-unique-characters]: https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/
[maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k]: https://leetcode.com/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/
[maximum-xor-product]: https://leetcode.com/problems/maximum-xor-product/
[maximum-xor-of-two-numbers-in-an-array]: https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/
[minimum-one-bit-operations-to-make-integers-zero]: https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/
[minimum-operations-to-make-the-integer-zero]: https://leetcode.com/problems/minimum-operations-to-make-the-integer-zero/
[minimum-operations-to-reduce-an-integer-to-0]: https://leetcode.com/problems/minimum-operations-to-reduce-an-integer-to-0/
[missing-number]: https://leetcode.com/problems/missing-number/
[single-number]: https://leetcode.com/problems/single-number/
[single-number-ii]: https://leetcode.com/problems/single-number-ii/
[single-number-iii]: https://leetcode.com/problems/single-number-iii/
[split-array-into-maximum-number-of-subarrays]: https://leetcode.com/problems/split-array-into-maximum-number-of-subarrays/
[total-hamming-distance]: https://leetcode.com/problems/total-hamming-distance/
[utf-8-validation]: https://leetcode.com/problems/utf-8-validation/
[xor-operation-in-an-array]: https://leetcode.com/problems/xor-operation-in-an-array/
