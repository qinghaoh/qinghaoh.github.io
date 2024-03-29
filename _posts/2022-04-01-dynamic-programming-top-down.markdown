---
title:  "Dynamic Programming (Top-down)"
category: algorithm
tag: dynamic programming
---

Divide and conquer + Memoization.

[Maximum Value of K Coins From Piles][maximum-value-of-k-coins-from-piles]

```java
private Integer[][] memo;

public int maxValueOfCoins(List<List<Integer>> piles, int k) {
    this.memo = new Integer[piles.size() + 1][k + 1];
    return dfs(piles, 0, k);
}

private int dfs(List<List<Integer>> piles, int i, int k) {
    if (k == 0 || i == piles.size()) {
        return 0;
    }

    if (memo[i][k] != null) {
        return memo[i][k];
    }

    int total = 0, max = dfs(piles, i + 1, k);
    for (int j = 0; j < Math.min(piles.get(i).size(), k); j++) {
        total += piles.get(i).get(j);
        max = Math.max(max, total + dfs(piles, i + 1, k - j - 1));
    }
    return memo[i][k] = max;
}
```

[Largest Sum of Averages][largest-sum-of-averages]

```java
private double[][] memo;
private double[] sum;

public double largestSumOfAverages(int[] nums, int k) {
    int n = nums.length;
    memo = new double[n + 1][k + 1];
    sum = new double[n + 1];

    for (int i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + nums[i];
    }
    return largestSumOfAverages(nums, n, k);
}

private double largestSumOfAverages(int[] nums, int end, int k) {
    if (memo[end][k] != 0) {
        return memo[end][k];
    }

    if (k == 1) {
        memo[end][1] = sum[end] / end;
        return memo[end][1];
    }

    // "at most k groups" is equivalent to "exact k groups"
    // so we don't need to consider largestSumOfAverages(nums, end, k - 1)
    //
    // see https://en.wikipedia.org/wiki/Mediant_(mathematics)#Properties
    double max = 0;
    for (int i = end - 1; i >= k - 1; i--) {
        max = Math.max(max, (sum[end] - sum[i]) / (end - i) + largestSumOfAverages(nums, i, k - 1));
    }
    memo[end][k] = max;
    return max;
}
```

The result of each loop can be written to `memo` directly:

```java
for (int i = end - 1; i >= k - 1; i--) {
    memo[end][k] = Math.max(memo[end][k], (sum[end] - sum[i]) / (end - i) + largestSumOfAverages(nums, i, k - 1));
}
return memo[end][k];
```

Bottom-up DP can be derived in the following way:

```java
public double largestSumOfAverages(int[] nums, int k) {
    int n = nums.length;

    double[] sum = new double[n + 1];
    for (int i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + nums[i];
    }

    double[][] dp = new double[n + 1][k + 1];
    for (int i = 1; i <= n; i++) {
        dp[i][1] = sum[i] / i;
    }

    for (int m = 2; m <= k; m++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i][m] = Math.max(dp[i][m], (sum[i] - sum[j]) / (i - j) + dp[j][m - 1]);
            }
        }
    }

    return dp[n][k];
}
```

```
[0.0,0.0,0.0,0.0]
[0.0,9.0,9.0,9.0]
[0.0,5.0,10.0,10.0]
[0.0,4.0,10.5,12.0]
[0.0,3.75,11.0,13.5]
[0.0,4.8,12.75,20.0]
```

1D:

```java
public double largestSumOfAverages(int[] nums, int k) {
    int n = nums.length;

    double[] sum = new double[n + 1];
    for (int i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + nums[i];
    }

    double[] dp = new double[n + 1];
    for (int i = 1; i <= n; i++) {
        dp[i] = sum[i] / i;
    }

    for (int m = 2; m <= k; m++) {
        for (int i = n; i > 0; i--) {
            for (int j = 0; j < i; j++) {
                dp[i] = Math.max(dp[i], (sum[i] - sum[j]) / (i - j) + dp[j]);
            }
        }
    }

    return dp[n];
}
```

DP scans array from right to left:

[Maximum Score from Performing Multiplication Operations][maximum-score-from-performing-multiplication-operations]

```java
public int maximumScore(int[] nums, int[] multipliers) {
    int n = nums.length, m = multipliers.length;
    // dp[i][j]: left of array elements has index i, and j is the index of multipliers 
    int[][] dp = new int[m + 1][m + 1];

    for (int i = m - 1; i >= 0; i--) {
        for (int j = m - 1; j >= 0; j--) {
            // len = n - j
            // right = left + len - 1
            int right = i + (n - j) - 1;
            if (right >= 0 && right < n) {
                // chooses start or end
                dp[i][j] = Math.max(dp[i + 1][j + 1] + nums[i] * multipliers[j], dp[i][j + 1] + nums[right] * multipliers[j]);
            }
        }
    }
    return dp[0][0];
}
```

[Minimum Score Triangulation of Polygon][minimum-score-triangulation-of-polygon]

```java
private int[][] memo;

public int minScoreTriangulation(int[] values) {
    int n = values.length;
    memo = new int[n][n];
    return minScore(values, 0, n - 1);
}

private int minScore(int[] values, int start, int end) {
    if (start + 1 == end) {
        return 0;
    }

    if (memo[start][end] > 0) {
        return memo[start][end];
    }

    int min = Integer.MAX_VALUE;
    for (int i = start + 1; i < end; i++) {
        int sum = values[start] * values[end] * values[i];
        sum += minScore(values, start, i);
        sum += minScore(values, i, end);
        min = Math.min(min, sum);
    }
    return memo[start][end] = min;
}
```

```java
public int minScoreTriangulation(int[] values) {
    int n = values.length;
    int[][] dp = new int[n][n];
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 2; j < n; j++) {
            dp[i][j] = Integer.MAX_VALUE;
            for (int k = i + 1; k < j; k++) {
                dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k][j] + values[i] * values[j] * values[k]);
            }
        }
    }
    return dp[0][n - 1];
}
```

```java
    for (int j = 2; j < n; j++) {
        for (int i = j - 2; i >= 0; i--) {

        }
    }
```

[Minimum Total Distance Traveled][minimum-total-distance-traveled]

```java
private Long[][][] memo;

public long minimumTotalDistance(List<Integer> robot, int[][] factory) {
    int n = robot.size(), m = factory.length;
    // memo[i][j][k]: the cost to fix robot[i...] at factory[j] which has already fixed k robots
    this.memo = new Long[n + 1][m + 1][n + 1];

    Collections.sort(robot);
    Arrays.sort(factory, Comparator.comparingInt(f -> f[0]));

    return helper(robot, factory, 0, 0, 0);
}

public long helper(List<Integer> robot, int[][] factory, int i, int j, int k) {
    // all robots are fixed
    if (i == robot.size()) {
        return 0;
    }

    // no more factory to fix the remaining robots
    if (j == factory.length) {
        return Long.MAX_VALUE;
    }

    if (memo[i][j][k] != null) {
        return memo[i][j][k];
    }

    // skips current factory
    long res1 = helper(robot, factory, i, j + 1, 0);

    // fixes current robot at current factory
    long res2 = Long.MAX_VALUE;
    if (factory[j][1] > k) {
        long val = helper(robot, factory, i + 1, j, k + 1);
        if (val != Long.MAX_VALUE) {
            res2 = val + Math.abs(robot.get(i) - factory[j][0]);
        }
    }
    return memo[i][j][k] = Math.min(res1, res2);
}
```

```java
public long minimumTotalDistance(List<Integer> robot, int[][] factory) {
    int n = robot.size(), m = factory.length;
    long[] dp = new long[n + 1];
    Arrays.fill(dp, (long)1e12);  // > 100 * 2 * 10 ^ 9
    dp[n] = 0;

    Collections.sort(robot);
    Arrays.sort(factory, Comparator.comparingInt(f -> f[0]));

    for (int j = m - 1; j >= 0; j--) {
        for (int i = 0; i < n; i++) {
            long d = 0;
            for (int k = 1; k <= Math.min(factory[j][1], n - i); k++) {
                // the (i + k - 1)-th robot is fixed by current factory
                d += Math.abs(robot.get(i + k - 1) - factory[j][0]);
                dp[i] = Math.min(dp[i], dp[i + k] + d);
            }
        }
    }
    return dp[0];
}
```

[largest-sum-of-averages]: https://leetcode.com/problems/largest-sum-of-averages/
[maximum-score-from-performing-multiplication-operations]: https://leetcode.com/problems/maximum-score-from-performing-multiplication-operations/
[maximum-value-of-k-coins-from-piles]: https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/
[minimum-score-triangulation-of-polygon]: https://leetcode.com/problems/minimum-score-triangulation-of-polygon/
[minimum-total-distance-traveled]: https://leetcode.com/problems/minimum-total-distance-traveled/

