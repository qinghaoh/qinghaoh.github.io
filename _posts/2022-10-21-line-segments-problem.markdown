---
title:  "Line Segment Problem"
category: algorithm
---
[Number of Sets of K Non-Overlapping Line Segments][number-of-sets-of-k-non-overlapping-line-segments]

```java
private static final int MOD = (int)1e9 + 7;

public int numberOfSets(int n, int k) {
    long[][] dp = new long[n][k + 1];
    dp[0][0] = 1;
    for (int i = 1; i < n; i++) {
        dp[i][0] = 1;
        for (int j = 1; j <= Math.min(k, i); j++) {
            // all segments are within [0, i - 1]
            dp[i][j] = dp[i - 1][j];

            // or, the last segment is [h, i]
            for (int h = 0; h < i; h++) {
                dp[i][j] = (dp[i][j] + dp[h][j - 1]) % MOD;
            }
        }
    }
    return (int)dp[n - 1][k];
}
```

```java
private static final int MOD = (int)1e9 + 7;

public int numberOfSets(int n, int k) {
    // dp[i][j][]:
    // 0: segments don't start from i
    // 1: segments start from i
    int[][][] dp = new int[n][k + 1][2];
    for (int i = 0; i < n; i++) {
        dp[i][0][0] = dp[i][0][1] = 1;
    }

    for (int i = n - 2; i >= 0; i--) {
        for (int j = 1; j <= k; j++) {
            dp[i][j][0] = (dp[i + 1][j][0] + dp[i + 1][j][1]) % MOD;
            dp[i][j][1] = (dp[i][j - 1][0] + dp[i + 1][j][1]) % MOD;
        }
    }
    return (dp[0][k][0] + dp[0][k][1]) % MOD;
}
```

[number-of-sets-of-k-non-overlapping-line-segments]: https://leetcode.com/problems/number-of-sets-of-k-non-overlapping-line-segments/
