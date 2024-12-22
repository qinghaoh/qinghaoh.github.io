---
title:  "Dynamic Programming (Dependent)"
category: algorithm
tag: dynamic programming
---

The dynamic programming problems are not self-contained, i.e. external information are required.

[Burst Balloons][burst-balloons]

```java
public int maxCoins(int[] nums) {
    int n = nums.length;

    // dp[i][j]: max coins after bursting all ballons in the range nums[i...j]
    //   and the rest balloons are NOT bursted. (reverse thinking)
    int[][] dp = new int[n][n];

    for (int len = 1; len <= n; len++) {
        for (int left = 0; left + len - 1 < n; left++) {
            int right = left + len - 1;
            // reverse thinking: for each balloon i that's last to burst
            // every element of the range [left, right] could be the last balloon to burst
            for (int i = left; i <= right; i++) {
                int leftNum = (left == 0) ? 1 : nums[left - 1];
                int rightNum = (right == n - 1) ? 1 : nums[right + 1];

                int leftSum = (i == left) ? 0 : dp[left][i - 1];
                int rightSum = (i == right) ? 0 : dp[i + 1][right];
                dp[left][right] = Math.max(dp[left][right], leftNum * nums[i] * rightNum + leftSum + rightSum);
            }
        }
    }

    return dp[0][n - 1];
}
```

[Remove Boxes][remove-boxes]

```java
public int removeBoxes(int[] boxes) {
    int n = boxes.length;
    // dp[i][j][k]: the maximum points by removing the boxes of subarray boxes[i...j]
    //   with k boxes attached to its left of the same color as boxes[i]
    int[][][] dp = new int[n][n][n];

    // boxes[i]
    for (int i = 0; i < n; i++) {
        for (int k = 0; k <= i; k++) {
            dp[i][i][k] = (k + 1) * (k + 1);
        }
    }

    for (int len = 1; len < n; len++) {
        for (int j = len; j < n; j++) {
            int i = j - len;

            for (int k = 0; k <= i; k++) {
                // Option 1: removes boxes[i]
                dp[i][j][k] = (k + 1) * (k + 1) + dp[i + 1][j][0];

                // Option 2: defers the removal of boxes[i] until boxes[(i + 1)...(m - 1)] are removed
                // if boxes[m] has the same color as boxes[i]
                for (int m = i + 1; m <= j; m++) {
                    if (boxes[m] == boxes[i]) {
                        dp[i][j][k] = Math.max(dp[i][j][k], dp[i + 1][m - 1][0] + dp[m][j][k + 1]);
                    }
                }
            }
        }
    }

    return dp[0][n - 1][0];
}
```

[Minimum Cost to Merge Stones][minimum-cost-to-merge-stones]

```java
public int mergeStones(int[] stones, int k) {
    int n = stones.length;

    // k + (k - 1) * m == n
    if ((n - 1) % (k - 1) != 0) {
        return -1;
    }

    // prefix sum
    int[] p = new int[n + 1];
    for (int i = 0; i < n; i++) {
        p[i + 1] = p[i] + stones[i];
    }

    // dp[i][j]: minimum cost to merge k consecutive piles in stones[i...j]
    // into as few piles as possible (one or more piles, not exactly into one pile)
    // the final number of piles is dependent on the range length
    // e.g. dp[1][8], k = 5, then the number of piles = 3
    int[][] dp = new int[n][n];

    for (int len = k; len <= n; len++) {
        for (int i = 0; i + len <= n; i++) {
            int j = i + len - 1;
            dp[i][j] = Integer.MAX_VALUE;
            // step == k - 1, because it ensures stones[i...m] can be merged to one pile
            for (int m = i; m < j; m += k - 1) {
                dp[i][j] = Math.min(dp[i][j], dp[i][m] + dp[m + 1][j]);
            }

            // (j - i) % (k - 1) stones remained from the above loop
            // if it's 0, merges stones[i...j] into one pile
            if ((j - i) % (k - 1) == 0) {
                dp[i][j] += p[j + 1] - p[i];
            }
        }
    }

    return dp[0][n - 1];
}
```

[Minimum Cost to Cut a Stick][minimum-cost-to-cut-a-stick]

```java
public int minCost(int n, int[] cuts) {
    List<Integer> list = new ArrayList<>();
    Arrays.stream(cuts).forEach(list::add);
    list.add(0);
    list.add(n);

    Collections.sort(list);

    int m = list.size();
    int[][] dp = new int[m][m];
    for (int i = m - 1; i >= 0; i--) {
        for (int j = i + 1; j < m; j++) {
            for (int k = i + 1; k < j; k++) {
                dp[i][j] = Math.min(dp[i][j] == 0 ? Integer.MAX_VALUE : dp[i][j], dp[i][k] + dp[k][j] + list.get(j) - list.get(i));
            }
        }
    }
    return dp[0][m - 1];
}
```

[burst-balloons]: https://leetcode.com/problems/burst-balloons/
[minimum-cost-to-cut-a-stick]: https://leetcode.com/problems/minimum-cost-to-cut-a-stick/
[minimum-cost-to-merge-stones]: https://leetcode.com/problems/minimum-cost-to-merge-stones/
[remove-boxes]: https://leetcode.com/problems/remove-boxes/
