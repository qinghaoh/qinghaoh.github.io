---
title:  "Permutation"
---
[Permutation Sequence][permutation-sequence]

```java
public String getPermutation(int n, int k) {
    List<Integer> num = new ArrayList<>();
    int fact = 1;
    for (int i = 1; i <= n; i++) {
        num.add(i);
        fact *= i;
    }

    StringBuffer sb = new StringBuffer();
    k--;  // list index is 0-based
    for (int i = n; i > 0; i--) {
        fact /= i;
        int index = k / fact;
        sb.append(num.remove(index));
        k -= index * fact;
    }
    return sb.toString();
}
```

# Lexicographically Sort

[Maximize Greatness of an Array][maximize-greatness-of-an-array]

```java
public int maximizeGreatness(int[] nums) {
    Arrays.sort(nums);

    // two pointers
    int count = 0;
    for (int num : nums) {
        // for each integer on its sorted position
        // checks if the leftmost unused integer is less than it
        // if not, continues
        if (num > nums[count]) {
            count++;
        }
    }
    return count;
}
```

## Next Permutation

[Next Permutation][next-permutation]

Algorithm: [Generation in lexicographic order](https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order)

```java
public void nextPermutation(int[] nums) {
    // Narayana Pandita
    // finds the largest index k such that a[k] < a[k + 1]
    int i = nums.length - 2;
    while (i >= 0 && nums[i + 1] <= nums[i]) {
        i--;
    }

    if (i >= 0) {
        // finds the largest index l greater than k such that a[k] < a[l]
        int j = nums.length - 1;
        while (j > i && nums[j] <= nums[i]) {
            j--;
        }
        // swaps the value of a[k] with that of a[l]
        swap(nums, i, j);
    }
    // reverses the sequence from a[k + 1] up to and including the final element a[n]
    reverse(nums, i + 1);
}

private void reverse(int[] nums, int start) {
    int i = start, j = nums.length - 1;
    while (i < j) {
        swap(nums, i++, j--);
    }
}

private void swap(int[] nums, int i, int j) {
    int tmp = nums[i];
    nums[i] = nums[j];
    nums[j] = tmp;
}
```

## Previous Permutation

[Minimum Number of Operations to Make String Sorted][minimum-number-of-operations-to-make-string-sorted]

$$\frac{n!}{\prod{\mathbf{card}(c)}!}$$

where \\(n\\) is the number of characters, and \\(\mathbf{card}(c)\\) is the count of each unique character.

https://www.geeksforgeeks.org/lexicographic-rank-string-duplicate-characters/

```java
private static final int MOD = (int)1e9 + 7;
private static final int MAX_LENGTH = 3000;

public int makeStringSorted(String s) {
    int[] factorial = new int[MAX_LENGTH + 1], inverse = new int[MAX_LENGTH + 1];
    factorial[0] = 1;
    inverse[0] = 1;

    for (int i = 1; i < factorial.length; i++) {
        factorial[i] = (int)((long)i * factorial[i - 1] % MOD);
        inverse[i] = (int)modPow(factorial[i], MOD - 2);
    }

    int n = s.length();
    long ops = 1;
    int[] count = new int[26];
    // for each s[i],
    // 1) counts the number of smaller characters on the right side of s[i] (less_than)
    // 2) computes the product of factorials of repetitions of each character (d_fac)
    // 3) computes (less_than * fac(n - i - 1)) / (d_fac).
    for (int i = n - 1; i >= 0; i--) {
        count[s.charAt(i) - 'a']++;
        long perm = (long)Arrays.stream(count).limit(s.charAt(i) - 'a').sum() * factorial[n - i - 1] % MOD;
        for (int c : count) {
            perm = perm * inverse[c] % MOD;
        }
        ops = (ops + perm) % MOD;
    }
    return (int)(ops - 1);
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
```

[Construct Smallest Number From DI String][construct-smallest-number-from-di-string]

```java
public String smallestNumber(String pattern) {
    // 1 2 3 4 5 6 7 8 9
    // D D I D D I D D
    StringBuilder res = new StringBuilder(), sb = new StringBuilder();
    for (int i = 0; i <= pattern.length(); i++) {
        sb.append(i + 1);
        // whenever encounters 'I', reverse the segment
        if (i == pattern.length() || pattern.charAt(i) == 'I') {
            res.append(sb.reverse());
            sb = new StringBuilder();
        }
    }
    return res.toString();
}
```

[Numbers With Repeated Digits][numbers-with-repeated-digits]

```java
public int numDupDigitsAtMostN(int n) {
    List<Integer> nums = new ArrayList<>();
    int tmp = n + 1;
    while (tmp != 0) {
        nums.add(0, tmp % 10);
        tmp /= 10;
    }

    // counts the number with digits < numDigits
    // e.g. 8765
    // xxx
    // xx
    // x
    int numDigits = nums.size(), noRepeats = 0;
    for (int i = 0; i < numDigits - 1; i++) {
        // excludes leading 0
        noRepeats += 9 * permutation(9, i);
    }

    // counts the number with same prefix
    // e.g. 8765
    // 1xxx ~ 7xxx
    // 80xx ~ 86xx
    // 870x ~ 875x
    // 8760 ~ 8765
    boolean[] used = new boolean[10];
    for (int i = 0; i < numDigits; i++) {
        int d = nums.get(i);
        // skips leading 0
        for (int j = i == 0 ? 1 : 0; j < d; j++) {
            // if the number j is not a part of the prefix
            if (!used[j]) {
                // prefix has (i + 1) digits
                noRepeats += permutation(10 - i - 1, numDigits - i - 1);
            }
        }

        // prefix has repeated number
        if (used[d]) {
            break;
        }
        used[d] = true;
    }
    return n - noRepeats;
}

// A(n, m)
private int permutation(int n, int m) {
    int p = 1;
    for (int i = 0; i < m; i++) {
        p *= n--;
    }
    return p;
}
```

# Dynamic Programming

[Valid Permutations for DI Sequence][valid-permutations-for-di-sequence]

```java
private static final int MOD = (int)1e9 + 7;

public int numPermsDISequence(String s) {
    int n = s.length();
    // dp[i][j]: number of permutations of [0, i], with DI-rule s.substring(0, i) and ending with digit j
    int[][] dp = new int[n + 1][n + 1];
    dp[0][0] = 1;

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= i; j++) {
            if (s.charAt(i - 1) == 'D') {
                // e.g. DID, 1032 -> DIDD that ends with 2
                // 1032 -> 1043 -> 10432
                // steps:
                // 1. increments digits that are larger than or equal to j
                // 2. appends j
                // k >= j, so k -> j is D
                for (int k = j; k < i; k++) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][k]) % MOD;
                }
            } else {
                // e.g. DID, 1032 -> DIDI that ends with 3
                // 1032 -> 1042 -> 10423 (append)
                // k < j, so k -> j is I
                for (int k = 0; k < j; k++) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][k]) % MOD;
                }
            }
        }
    }

    int count = 0;
    for (int i = 0; i <= n; i++) {
        count = (count + dp[n][i]) % MOD;
    }
    return count;
}
```

[construct-smallest-number-from-di-string]: https://leetcode.com/problems/construct-smallest-number-from-di-string/
[maximize-greatness-of-an-array]: https://leetcode.com/problems/maximize-greatness-of-an-array/
[minimum-number-of-operations-to-make-string-sorted]: https://leetcode.com/problems/minimum-number-of-operations-to-make-string-sorted/
[next-permutation]: https://leetcode.com/problems/next-permutation/
[numbers-with-repeated-digits]: https://leetcode.com/problems/numbers-with-repeated-digits/
[permutation-sequence]: https://leetcode.com/problems/permutation-sequence/
[valid-permutations-for-di-sequence]: https://leetcode.com/problems/valid-permutations-for-di-sequence/
