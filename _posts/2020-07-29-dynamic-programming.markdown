---
title:  "Dynamic Programming (Edit Distance)"
category: algorithm
tag: dynamic programming
---
[Edit Distance][edit-distance]

```java
public int minDistance(String word1, String word2) {
    int n1 = word1.length(), n2 = word2.length();

    // dp[i][j]: word1.substring(0, i) -> word2.substring(0, j)
    int[][] dp = new int[n1 + 1][n2 + 1];

    // word1.substring(0, i) -> empty string
    for (int i = 1; i <= n1; i++) {
        dp[i][0] = i;
    }

    // empty string -> word2.substring(0, j)
    for (int j = 1; j <= n2; j++) {
        dp[0][j] = j;
    }

    for (int i = 1; i <= n1; i++) {
        for (int j = 1; j <= n2; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                // replace: dp[i - 1][j - 1] + 1
                // delete/insert: dp[i - 1][j] + 1, dp[i][j - 1] + 1
                dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
            }
        }
    }
    return dp[n1][n2];
}
```

Visually, we initialize boarder cells.

For example, `word1 = "newton", word2 = "einstein"`, then `dp` is:
```
[0,1,2,3,4,5,6,7,8]
[1,1,2,2,3,4,5,6,7]
[2,1,2,3,3,4,4,5,6]
[3,2,2,3,4,4,5,5,6]
[4,3,3,3,4,4,5,6,6]
[5,4,4,4,4,5,5,6,7]
[6,5,5,4,5,5,6,6,6]
```

Notice `dp[i - 1][j - 1] <= dp[i][j - 1] + 1` and `dp[i - 1][j - 1] <= dp[i - 1][j] + 1`

Rolling array optimization:
* `dp[i - 1][j] -> prev[j]`
* `dp[i][j] -> curr[j]`

```java
public int minDistance(String word1, String word2) {
    int n1 = word1.length(), n2 = word2.length();
    int[] prev = new int[n2 + 1], curr = new int[n2 + 1];

    for (int j = 1; j <= n2; j++) {
        prev[j] = j;
    }

    for (int i = 1; i <= n1; i++) {
        curr[0] = i;
        for (int j = 1; j <= n2; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                curr[j] = prev[j - 1];
            } else {
                curr[j] = Math.min(prev[j - 1], Math.min(prev[j], curr[j - 1])) + 1;
            }
        }
        int[] tmp = prev;
        prev = curr;
        curr = tmp;
    }
    return prev[n2];
}
```

* `prev[j - 1] -> prev`
* `prev[j] -> curr[j]`

```java
public int minDistance(String word1, String word2) {
    int prev = 0, n1 = word1.length(), n2 = word2.length();
    int[] curr = new int[n2 + 1];

    for (int j = 1; j <= n2; j++) {
        curr[j] = j;
    }

    for (int i = 1; i <= n1; i++) {
        prev = curr[0];
        curr[0] = i;
        for (int j = 1; j <= n2; j++) {
            int tmp = curr[j];
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                curr[j] = prev;
            } else {
                curr[j] = Math.min(prev, Math.min(curr[j], curr[j - 1])) + 1;
            }
            prev = tmp;
        }
    }
    return curr[n2];
}
```

Similar problem: [Delete Operation for Two Strings][delete-operation-for-two-strings]

[Minimum ASCII Delete Sum for Two Strings][minimum-ascii-delete-sum-for-two-strings]

```java
char c1 = s1.charAt(i - 1), c2 = s2.charAt(j - 1);
if (c1 == c2) {
    dp[i][j] = dp[i - 1][j - 1];
} else {
    dp[i][j] = Math.min(dp[i][j - 1] + c2, dp[i - 1][j] + c1);
}
```

**Longest Common Subsequence**

[Longest Common Subsequence][longest-common-subsequence]

```java
if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
    dp[i][j] = dp[i - 1][j - 1] + 1;
} else {
    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
}
```

[Uncrossed Lines][uncrossed-lines] can be transformed to [Longest Common Subsequence][longest-common-subsequence]!

**Longest Common Subarray**

[Maximum Length of Repeated Subarray][maximum-length-of-repeated-subarray]

```java
// dp[i][j]: max length of repeated subarray ending with nums1[i - 1] and nums2[j - 1]
int[][] dp = new int[n1 + 1][n2 + 1];
for (int i = 1; i <= n1; i++) {
    for (int j = 1; j <= n2; j++) {
        if (nums1[i - 1] == nums2[j - 1]) {
            max = Math.max(max, dp[i][j] = dp[i - 1][j - 1] + 1);
        }
    }
}
```

[Distinct Subsequences][distinct-subsequences]

```java
// t == ""
for (int i = 0; i <= m; i++) {
    dp[i][0] = 1;
}

for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        dp[i + 1][j + 1] = dp[i][j + 1];

        if (s.charAt(i) == t.charAt(j)) {
            dp[i + 1][j + 1] += dp[i][j];
        }
    }
}
```

[Minimum Window Subsequence][minimum-window-subsequence]

```java
public String minWindow(String s1, String s2) {
    int n1 = s1.length(), n2 = s2.length();
    // dp[i][j] = k: (k - 1) is the start index of the shortest postfix of s1.substring(0, i)
    // such that s1.substring(k, i) is a supersequence of s2.substring(0, j)
    // notice k is 1-indexed
    int[][] dp = new int[n1 + 1][n2 + 1];

    for (int i = 0; i <= n1; i++) {
        dp[i][0] = i + 1;
    }

    for (int i = 1; i <= n1; i++) {
        for (int j = 1; j <= n2; j++) {
            // e.g. s1 = "abbcd"
            //      s2 =    "bd"
            // s1[4] == s2[1]
            // dp[5][2] = dp[4][1] = 3 (points the second 'b' in s1)
            if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }

    int start = 0, len = n1 + 1;
    for (int i = 1; i <= n1; i++) {
        // dp[i][n2] == 0 means not found
        if (dp[i][n2] != 0 && i - dp[i][n2] + 1 < len) {
            start = dp[i][n2] - 1;
            // [start, end)
            // len = end - start
            //     = i   - (dp[i][n2] - 1)
            len = i - dp[i][n2] + 1;
        }
    }
    return s1.substring(start, start + (len > n1 ? 0 : len));
}
```

The space complex can be optimized to `O(n)`.

[Interleaving String][interleaving-string]

```java
public boolean isInterleave(String s1, String s2, String s3) {
    int n1 = s1.length(), n2 = s2.length();
    if (s3.length() != n1 + n2) {
        return false;
    }

    // dp[i][j]: s1.substring(0, i) and s2.substring(0, j)
    boolean dp[][] = new boolean[n1 + 1][n2 + 1];
    dp[0][0] = true;

    for (int j = 0; j < n2; j++) {
        dp[0][j + 1] = dp[0][j] && s2.charAt(j) == s3.charAt(j);
    }

    for (int i = 0; i < n1; i++) {
        dp[i + 1][0] = dp[i][0] && s1.charAt(i) == s3.charAt(i);
    }

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            dp[i + 1][j + 1] = (dp[i][j + 1] && s1.charAt(i) == s3.charAt(i + j + 1)) || (dp[i + 1][j] && s2.charAt(j) == s3.charAt(i + j + 1));
        }
    }

    return dp[n1][n2];
}
```

Reduced to 1D:

```java
boolean dp[] = new boolean[n2 + 1];
dp[0] = true;

// initializes first row
for (int j = 0; j < n2; j++) {
    dp[j + 1] = dp[j] && s2.charAt(j) == s3.charAt(j);
}

for (int i = 0; i < n1; i++) {
    // initializes first column in this row
    dp[0] = dp[0] && s1.charAt(i) == s3.charAt(i);
    for (int j = 0; j < n2; j++) {
        dp[j + 1] = (dp[j + 1] && s1.charAt(i) == s3.charAt(i + j + 1)) || (dp[j] && s2.charAt(j) == s3.charAt(i + j + 1));
    }
}

return dp[n2];
```

[Freedom Trail][freedom-trail]

```java
// O(mn^2)
public int findRotateSteps(String ring, String key) {
    // dp[i][j]: key.substring(i) and ring.substring(j)
    int n = ring.length(), m = key.length();
    int[][] dp = new int[m + 1][n];

    // reversely scans key
    for (int i = m - 1; i >= 0; i--) {
        // ring points to j
        for (int j = 0; j < n; j++) {
            dp[i][j] = Integer.MAX_VALUE;
            for (int k = 0; k < n; k++) {
                if (key.charAt(i) == ring.charAt(k)) {
                    int diff = Math.abs(j - k);
                    int step = Math.min(diff, n - diff);
                    dp[i][j] = Math.min(dp[i][j], step + dp[i + 1][k]);
                }
            }
        }
    }

    // if we process from 0 to (m - 1)
    // there could be multiple indefinite final states dp[i][j]
    return dp[0][0] + m;
}
```

Precomputation:

```java
// O(mn)
public int findRotateSteps(String ring, String key) {
    int n = ring.length(), m = key.length();
    // dp[i][j]: key.substring(i) and ring.substring(j)
    int[][] dp = new int[m + 1][n];

    int[][] clock = preProcess(ring, 1), anti = preProcess(ring, -1);
    for (int i = m - 1; i >= 0; i--) {
        int index = key.charAt(i) - 'a';
        for (int j = 0; j < n; j++) {
            int p = clock[j][index], q = anti[j][index];
            dp[i][j] = Math.min(dp[i + 1][p] + (j + n - p) % n, dp[i + 1][q] + (q + n - j) % n);
        }
    }
    return dp[0][0] + m;
}

/**
 * Precomputes the last index memo array.
 * @param r ring
 * @param orientation: clockwise (1) or anticlockwise (-1)
 * @return last index memo array
 */
private int[][] preProcess(String r, int orientation) {
    int n = r.length();
    // lastIndex[i][j]: last index of character (j + 'a') appears before (or at) the i-th position of r (wrapped)
    int[][] lastIndex = new int[n][26];

    // map[i]: last index of character (i + 'a')
    int[] map = new int[26];

    // "abc" -> "abcab"
    for (int i = 0, j = 0; j < n * 2 - 1; j++) {
        map[r.charAt(i) - 'a'] = i;

        // all the indexes before (n - 1) will be written twice
        // i.e. i and i + n
        // so the map copied at i will be overwritten at (i + n)
        // therefore, we can skip the first copy and start from (n - 1) directly
        if (j >= n - 1) {
            System.arraycopy(map, 0, lastIndex[i], 0, 26);
        }

        i = (i + orientation + n) % n;
    }
    return lastIndex;
}
```

[delete-operation-for-two-strings]: https://leetcode.com/problems/delete-operation-for-two-strings/
[distinct-subsequences]: https://leetcode.com/problems/distinct-subsequences/
[edit-distance]: https://leetcode.com/problems/edit-distance/
[freedom-trail]: https://leetcode.com/problems/freedom-trail/
[interleaving-string]: https://leetcode.com/problems/interleaving-string/
[longest-common-subsequence]: https://leetcode.com/problems/longest-common-subsequence/
[maximum-length-of-repeated-subarray]: https://leetcode.com/problems/maximum-length-of-repeated-subarray/
[minimum-ascii-delete-sum-for-two-strings]: https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/
[minimum-window-subsequence]: https://leetcode.com/problems/minimum-window-subsequence/
[uncrossed-lines]: https://leetcode.com/problems/uncrossed-lines/
