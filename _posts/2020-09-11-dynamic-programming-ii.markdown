---
layout: post
title:  "Dynamic Programming II"
tag: dynamic programming
---
[Largest Sum of Averages][largest-sum-of-averages]

## Top-down DP: divide and conquer + memoization

{% highlight java %}
private double[][] memo;
private double[] sum;

public double largestSumOfAverages(int[] A, int K) {
    memo = new double[A.length + 1][K + 1];
    sum = new double[A.length + 1];

    for (int i = 0; i < A.length; i++) {
        sum[i + 1] = sum[i] + A[i];
    }
    return largestSumOfAverages(A, A.length, K);
}

private double largestSumOfAverages(int[] A, int end, int K) {
    if (memo[end][K] != 0) {
        return memo[end][K];
    }

    if (K == 1) {
        memo[end][1] = sum[end] / end;
        return memo[end][1];
    }

    // "at most K groups" is equivalent to "exact K groups"
    // so we don't need to consider largestSumOfAverages(A, end, K - 1)
    //  
    // see https://en.wikipedia.org/wiki/Mediant_(mathematics)#Properties
    double max = 0;
    for (int i = end - 1; i >= K - 1; i--) {
        max = Math.max(max, (sum[end] - sum[i]) / (end - i) + largestSumOfAverages(A, i, K - 1));
    }
    memo[end][K] = max;
    return max;
}
{% endhighlight %}

The result of each loop can be written to `memo` directly:

{% highlight java %}
    for (int i = end - 1; i >= K - 1; i--) {
        memo[end][K] = Math.max(memo[end][K], (sum[end] - sum[i]) / (end - i) + largestSumOfAverages(A, i, K - 1));
    }
    return memo[end][K];
{% endhighlight %}

## Bottom-up DP

{% highlight java %}
public double largestSumOfAverages(int[] A, int K) {
    double[][] dp = new double[A.length + 1][K + 1];

    double[] sum = new double[A.length + 1];
    for (int i = 0; i < A.length; i++) {
        sum[i + 1] = sum[i] + A[i];
    }

    for (int i = 1; i < dp.length; i++) {
        dp[i][1] = sum[i] / i;
    }

    for (int k = 2; k <= K; k++) {
        for (int i = 1; i <= A.length; i++) {
            for (int j = 0; j < i; j++) {
                dp[i][k] = Math.max(dp[i][k], (sum[i] - sum[j]) / (i - j) + dp[j][k - 1]);
            }
        }
    }

    return dp[A.length][K];
}
{% endhighlight %}

```
[0.0,0.0,0.0,0.0]
[0.0,9.0,9.0,9.0]
[0.0,5.0,10.0,10.0]
[0.0,4.0,10.5,12.0]
[0.0,3.75,11.0,13.5]
[0.0,4.8,12.75,20.0]
```

1D:

{% highlight java %}
public double largestSumOfAverages(int[] A, int K) {
    double[] dp = new double[A.length + 1];

    double[] sum = new double[A.length + 1];
    for (int i = 0; i < A.length; i++) {
        sum[i + 1] = sum[i] + A[i];
    }

    for (int i = 1; i < dp.length; i++) {
        dp[i] = sum[i] / i;
    }

    for (int k = 2; k <= K; k++) {
        for (int i = A.length; i > 0; i--) {
            for (int j = 0; j < i; j++) {
                dp[i] = Math.max(dp[i], (sum[i] - sum[j]) / (i - j) + dp[j]);
            }
        }
    }

    return dp[A.length];
}
{% endhighlight %}

DP scans array from right to left:

[Maximum Score from Performing Multiplication Operations][maximum-score-from-performing-multiplication-operations]

{% highlight java %}
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
{% endhighlight %}

[Minimum Score Triangulation of Polygon][minimum-score-triangulation-of-polygon]

{% highlight java %}
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
{% endhighlight %}

{% highlight java %}
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
{% endhighlight %}

{% highlight java %}
    for (int j = 2; j < n; j++) {
        for (int i = j - 2; i >= 0; i--) {

        }
    }
{% endhighlight %}

[Minimum Total Distance Traveled][minimum-total-distance-traveled]

{% highlight java %}
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
{% endhighlight %}

{% highlight java %}
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
{% endhighlight %}

[largest-sum-of-averages]: https://leetcode.com/problems/largest-sum-of-averages/
[maximum-score-from-performing-multiplication-operations]: https://leetcode.com/problems/maximum-score-from-performing-multiplication-operations/
[minimum-score-triangulation-of-polygon]: https://leetcode.com/problems/minimum-score-triangulation-of-polygon/
[minimum-total-distance-traveled]: https://leetcode.com/problems/minimum-total-distance-traveled/

