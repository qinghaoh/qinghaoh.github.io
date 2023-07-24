---
title:  "Dynamic Programming (Rolling)"
category: algorithm
tags: dynamic programming
---
[Check if There is a Valid Partition For The Array][check-if-there-is-a-valid-partition-for-the-array]

```java
public boolean validPartition(int[] nums) {
    boolean[] dp = {true, false, nums[0] == nums[1], false};
    int n = nums.length;
    for (int i = 2; i < n; i++) {
        boolean two = nums[i] == nums[i - 1];
        boolean three = (two && nums[i] == nums[i - 2]) || (nums[i] - 1 == nums[i - 1] && nums[i] - 2 == nums[i - 2]);
        // we just need to record the values in 4 cases: i - 2, i - 1, i, i + 1
        // so a DP array of size 4 is enough
        dp[(i + 1) % 4] = (two && dp[(i - 1) % 4]) || (three && dp[(i - 2) % 4]);
    }
    return dp[n % 4];
}
```

[Number of People Aware of a Secret][number-of-people-aware-of-a-secret]

```java
private static final int MOD = (int)1e9 + 7;

public int peopleAwareOfSecret(int n, int delay, int forget) {
    int[] dp = new int[forget];
    dp[0] = 1;

    // people who know the secret on the i-th day
    int people = 0;
    // circular array, sliding window
    for (int i = 1; i < n; i++) {
        people = ((people + dp[(i - delay + forget) % forget]) % MOD - dp[i % forget] + MOD) % MOD;
        dp[i % forget] = people;
    }

    int sum = 0;
    for (int d : dp) {
        sum = (sum + d) % MOD;
    }
    return sum;
}
```

[Number of Distinct Roll Sequences][number-of-distinct-roll-sequences]

```java
private static final int MOD = (int)1e9 + 7;

public int distinctSequences(int n) {
    if (n == 1) {
        return 6;
    }

    // dp[i][j]: number of distinct roll sequences from value j to value i
    // the intial array presents the result of the first two rolls
{% raw %}
    int[][] dp = {{0, 1, 1, 1, 1, 1},
                  {1, 0, 1, 0, 1, 0},
                  {1, 1, 0, 1, 1, 0},
                  {1, 0, 1, 0, 1, 0},
                  {1, 1, 1, 1, 0, 1},
                  {1, 0, 0, 0, 1, 0}}, dp1 = new int[6][6];
{% endraw %}

    for (int i = 3; i <= n; i++) {
        for (int d = 0; d < 6; d++) {
            // (i - 1)th roll
            for (int p = 0; p < 6; p++) {
                dp1[d][p] = 0;
                if (dp[d][p] > 0) {
                    // (i - 2)th roll
                    for (int pp = 0; pp < 6; pp++) {
                        // condition #2
                        if (d != pp) {
                            dp1[d][p] = (dp1[d][p] + dp[p][pp]) % MOD;
                        }
                    }
                }
            }
        }

        // rolling by swapping
        int[][] tmp = dp;
        dp = dp1;
        dp1 = tmp;
    }

    int sum = 0;
    for (int i = 0; i < dp.length; i++) {
        for (int j = 0; j < dp[i].length; j++) {
            sum = (sum + dp[i][j]) % MOD;
        } 
    }
    return sum;
}
```

[check-if-there-is-a-valid-partition-for-the-array]: https://leetcode.com/problems/check-if-there-is-a-valid-partition-for-the-array/
[number-of-distinct-roll-sequences]: https://leetcode.com/problems/number-of-distinct-roll-sequences/
[number-of-people-aware-of-a-secret]: https://leetcode.com/problems/number-of-people-aware-of-a-secret/
