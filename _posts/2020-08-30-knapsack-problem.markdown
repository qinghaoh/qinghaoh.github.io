---
title:  "Knapsack Problem"
tag: dynamic programming
---
# Fundamentals

[Knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem)

Backtracking takes \\(O(2^n)\\) time, so it's less preferable. Dynamic Programming is better, and its form is like `dp[i][j]`, which means the first i elements sums to j.

# 0-1 Knapsack Problem

maximize $$ \sum _{i=1}^{n}v_{i}x_{i} $$

subject to $$ \sum _{i=1}^{n}w_{i}x_{i}\leq W $$ and $$ x_{i}\in \{0,1\} $$

## Template

```java
for (int elment : elements) {
    for (int i = weight; i >= element; i--) {
        dp[i] = Math.max(dp[i], dp[weight - element] + element);
    }
}
```

The problem [Parition Equal Subset Sum][partition-equal-subset-sum] below demonstrates how the template is derived. Mapping the template to the Knapsack model: `weight` is the upper bound of the picked element sum, and `dp[i]` is the optimal `value` when the weight sum is `i`.

The key point of solving Knapsack Problem is to identify the meaning of `weight` and `value` under a specific condition and apply the template.

## Subset Sum Problem

The [subset sum problem (SSP)](https://en.wikipedia.org/wiki/Subset_sum_problem):  there is a multiset `S` of integers and a target-sum `T`, and the question is to decide whether any subset of the integers sum to precisely `T`. It's NP-complete.

[Partition Equal Subset Sum][partition-equal-subset-sum]

With full dimensionality (no reduction), we can backtrace.

```java
public boolean canPartition(int[] nums) {
    int n = nums.length, sum = Arrays.stream(nums).sum();

    if (sum % 2 == 1) {
        return false;
    }

    // dp[i][j]: whether the first i elements can sum up to j
    boolean[][] dp = new boolean[n + 1][sum / 2 + 1];
    dp[0][0] = true;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= sum / 2; j++) {
            if (j < nums[i]) {
                dp[i + 1][j] = dp[i][j];
            } else {
                // no picks || picks nums[i]
                dp[i + 1][j] = dp[i][j] || dp[i][j - nums[i]];
            }
        }
    }
    return dp[n][sum / 2];
}
```

For example, `nums = [1,2,5,1]`, then `dp` is:

```
[true,false,false,false,false,false]
[true,true,false,false,false,false]
[true,true,true,true,false,false]
[true,true,true,true,false,true]
[true,true,true,true,true,true]
```

![2D](/assets/img/algorithm/knapsack_partition_equal_subset_sum_2d.png)

```java
public boolean canPartition(int[] nums) {
    int sum = Arrays.stream(nums).sum();

    if (sum % 2 == 1) {
        return false;
    }

    boolean[] dp = new boolean[sum / 2 + 1];
    dp[0] = true;

    for (int num : nums) {
        // dp[k + 1][i] depends on dp[k][i - num]
        //  so the iteration order is reversed
        for (int i = sum / 2; i >= num; i--) {
            dp[i] = dp[i] || dp[i - num];
        }
    }
    return dp[sum / 2];
}
```

In 2D, `dp[i + 1][j] = dp[i][j] || dp[i][j - nums[i]]`. The reverse iteration ensures `dp[i][j - nums[i]]` is not updated to `dp[i + 1][j - nums[i]]` before we update `dp[i][j]` to `dp[i + 1][j]`.

![1D](/assets/img/algorithm/knapsack_partition_equal_subset_sum_1d.png)

[Target Sum][target-sum]

```java
public int findTargetSumWays(int[] nums, int target) {
    int sum = Arrays.stream(nums).sum();

    // sum(P) - sum(N) == target
    // sum(P) - (sum - sum(P)) == target
    // 2 * sum(P) == target + sum
    // sum(P) == (target + sum) / 2
    return target + sum < 0 || (target + sum) % 2 > 0 ? 0 : subsetSum(nums, (target + sum) >>> 1); 
}   

private int subsetSum(int[] nums, int target) {
    int[] dp = new int[target + 1];
    dp[0] = 1;
    for (int num : nums) {
        for (int i = target; i >= num; i--) {
            dp[i] += dp[i - num]; 
        }
    }
    return dp[target];
}
```

[Number of Great Partitions][number-of-great-partitions]

```java
private static final int MOD = (int)1e9 + 7;

public int countPartitions(int[] nums, int k) {
    long sum = 0, count = 1;
    long[] dp = new long[k];
    dp[0] = 1;
    for (int num : nums) {
        for (int i = k - 1; i >= num; i--) {
            dp[i] = (dp[i] + dp[i - num]) % MOD;
        }

        count = count * 2 % MOD;
        sum += num;
    }

    // now count == 2 ^ n is the total number of distinct partitions
    for (int i = 0; i < k; i++) {
        // sa = sum(group_a) = i
        // sb = sum(group_b) = sum - i
        //
        // inclusion–exclusion principle
        // set A: sa < k
        // set B: sb < k
        // set A and B (intersection): sa < k && sb < k
        // set A or B (union): total count
        //
        // if only one of the groups has sum < k, say sa = i < k
        // its inverse pair (sa = sum - i, sb = i) also needs to be deducted from total
        // so count -= 2 * dp[i]
        //
        // if both groups have sum < k, i.e. sa = i < k, sb = sum - i < k
        // it's the intersection, so count -= dp[i]
        count -= dp[i] * (sum - i < k ? 1 : 2);
    }
    return (int)((count % MOD + MOD) % MOD);
}
```

## Variants

**Max**

[Last Stone Weight II][last-stone-weight-ii]

```java
public int lastStoneWeightII(int[] stones) {
    int sum = Arrays.stream(stones).sum();
    boolean[] dp = new boolean[sum / 2 + 1];
    dp[0] = true;

    for (int stone : stones) {
        // considers smaller group only
        for (int i = sum / 2; i >= stone; i--) {
            dp[i] = dp[i] || dp[i - stone];
        }
    }

    // smaller group
    for (int i = sum / 2; i >= 0; i--) {
        if (dp[i]) {
            return sum - 2 * i;
        }
    }
    return 0;
}
```

```java
public int lastStoneWeightII(int[] stones) {
    int sum = Arrays.stream(stones).sum();
    // dp[i]: weight sum of stones that is closest to i
    //   it's possible that no group of stones can sum to i
    //   e.g. stones = [1,2,5]
    //   dp = [0, 1, 2, 3, 3]
    //   dp[sum / 2] = dp[4] = 3, i.e. stone 1 and 2
    int[] dp = new int[sum / 2 + 1];
    for (int stone : stones) {
        // the smaller group has a sum no greater than sum / 2
        for (int i = sum / 2; i >= stone; i--) {
            dp[i] = Math.max(dp[i], dp[i - stone] + stone);
        }
    }
    return sum - 2 * dp[sum / 2];
}
```

[Maximum Profit From Trading Stocks][maximum-profit-from-trading-stocks]

```java
public int maximumProfit(int[] present, int[] future, int budget) {
    int[] dp = new int[budget + 1];
    for (int i = 0; i < present.length; i++) {
        for (int j = budget; j >= present[i]; j--) {
            dp[j] = Math.max(dp[j], dp[j - present[i]] + future[i] - present[i]);
        }
    }
    return dp[budget];
}
```

**Probability**

[Toss Strange Coins][toss-strange-coins]

```java
public double probabilityOfHeads(double[] prob, int target) {
    int n = prob.length;

    // dp[i][j]: probability that the number of the first i coins equals j
    double[][] dp = new double[n + 1][target + 1];
    dp[0][0] = 1d;

    for (int i = 0; i < n; i++) {
        dp[i + 1][0] = dp[i][0] * (1 - prob[i]);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 1; j <= target; j++) {
            dp[i + 1][j] = dp[i][j] * (1 - prob[i]) + dp[i][j - 1] * prob[i];
        }
    }
    return dp[n][target];
}
```

**3D**

[Ones and Zeroes][ones-and-zeroes]

```java
public int findMaxForm(String[] strs, int m, int n) {
    int len = strs.length;
    int[][][] dp = new int[len + 1][m + 1][n + 1];

    for (int i = 0; i < len; i++) {
        int zeroes = (int)strs[i].chars().filter(ch -> ch == '0').count();
        int ones = strs[i].length() - zeroes;

        for (int j = 0; j <= m; j++) {
            for (int k = 0; k <= n; k++) {
                if (j < zeroes || k < ones) {
                    dp[i + 1][j][k] = dp[i][j][k];
                } else {
                    dp[i + 1][j][k] = Math.max(dp[i][j][k], dp[i][j - zeroes][k - ones] + 1);
                }
            }
        }
    }

    return dp[len][m][n];
}
```

2D:

```java
public int findMaxForm(String[] strs, int m, int n) {
    int[][] dp = new int[m + 1][n + 1];

    for (String str : strs) {
        int zeroes = (int)str.chars().filter(ch -> ch == '0').count();
        int ones = str.length() - zeroes;

        for (int i = m; i >= zeroes; i--) {
            for (int j = n; j >= ones; j--) {
                dp[i][j] = Math.max(dp[i][j], dp[i - zeroes][j - ones] + 1);
            }
        }
    }

    return dp[m][n];
}
```

[Profitable Schemes][profitable-schemes]

```java
private static final int MOD = (int)1e9 + 7;

public int profitableSchemes(int n, int minProfit, int[] group, int[] profit) {
    // dp[i][j]: count of schemes with profit >= j done by exactly i members
    int[][] dp = new int[n + 1][minProfit + 1];
    dp[0][0] = 1;

    for (int k = 0; k < group.length; k++) {
        for (int i = n; i >= group[k]; i--) {
            for (int j = minProfit; j >= 0; j--) {
                dp[i][j] = (dp[i][j] + dp[i - group[k]][Math.max(0, j - profit[k])]) % MOD;
            }
        }
    }

    int count = 0;
    for (int i = 0; i < dp.length; i++) {
        count = (count + dp[i][minProfit]) % MOD;
    }
    return count;
}
```

**Count of Selected Elements**

[Split Array With Same Average][split-array-with-same-average]

```java
public boolean splitArraySameAverage(int[] nums) {
int n = nums.length, sum = Arrays.stream(nums).sum();

    // assumes size(A) <= size(B)
    // dp[i][j]: whether it's possible to sum to i with j elements in array A
    boolean[][] dp = new boolean[sum + 1][n / 2 + 1];
    dp[0][0] = true;

    for (int num : nums) {
        for (int i = sum; i >= num; i--) {
            // the second dimension is used to record the count of selected elements
            for (int j = 1; j <= n / 2; j++) {
                dp[i][j] = dp[i][j] || dp[i - num][j - 1];
            }
        }
    }

    // when avg(A) == avg(B) == avg(nums)
    // sum(A) = avg(nums) * size(A)
    //        = sum * size(A) / n
    // iterates size(A) from 1 through n / 2
    for (int i = 1; i <= n / 2; i++)  {
        if (sum * i % n == 0 && dp[sum * i / n][i]) {
            return true;
        }
    }
    return false;
}
```

**Multi-layer**

[Number of Ways to Earn Points][number-of-ways-to-earn-points]

```java
private static final int MOD = (int)1e9 + 7;

public int waysToReachTarget(int target, int[][] types) {
    int[] dp = new int[target + 1];
    dp[0] = 1;

    for (int[] t : types) {
        for (int j = target; j >= 0; j--) {
            for (int k = 1; k <= t[0] && k * t[1] <= j; k++) {
                dp[j] = (dp[j] + dp[j - k * t[1]]) % MOD;
            }
        }
    }

    return dp[target];
}
```

[Painting the Walls][painting-the-walls]

```java
private static final int MAX_COST = (int)5e8;

public int paintWalls(int[] cost, int[] time) {
    int n = cost.length;
    // dp[i]: min amount of money required to paint i walls
    int[] dp = new int[n + 1];
    Arrays.fill(dp, MAX_COST);
    dp[0] = 0;

    for (int i = 0; i < n; i++) {
        // j is the number of remaining walls
        for (int j = n; j > 0; j--) {
            // if a paid painter is selected for the current wall
            // then in the time period time[i]:
            // * 1 wall is painted by the paid painter
            // * at most time[i] walls are painted by free painter
            dp[j] = Math.min(dp[j], dp[Math.max(j - time[i] - 1, 0)] + cost[i]);
        }
    }
    return dp[n];
}
```

## Top-down

[Maximize Total Tastiness of Purchased Fruits][maximize-total-tastiness-of-purchased-fruits]

```java
private int[] price, tastiness;
private int maxAmount, maxCoupons;
private Integer[][][] memo;

public int maxTastiness(int[] price, int[] tastiness, int maxAmount, int maxCoupons) {
    this.price = price;
    this.tastiness = tastiness;
    this.maxAmount = maxAmount;
    this.maxCoupons = maxCoupons;
    this.memo = new Integer[price.length + 1][maxAmount + 1][maxCoupons + 1];

    return dfs(0, maxAmount, maxCoupons);
}

private int dfs(int i, int amount, int coupon) {
    if (i == price.length) {
        return 0;
    }

    if (memo[i][amount][coupon] != null) {
        return memo[i][amount][coupon];
    }

    int max = dfs(i + 1, amount, coupon);
    if (price[i] <= amount) {
        max = Math.max(max, dfs(i + 1, amount - price[i], coupon) + tastiness[i]);
    }

    if (coupon > 0 && price[i] / 2 <= amount) {
        max = Math.max(max, dfs(i + 1, amount - price[i] / 2, coupon - 1) + tastiness[i]);
    }

    return memo[i][amount][coupon] = max;
}
```

# Unbounded Knapsack Problem (UKP)

maximize $$ \sum _{i=1}^{n}v_{i}x_{i} $$

subject to $$ \sum _{i=1}^{n}w_{i}x_{i}\leq W $$ and $$ x_{i}\geq 0,\ x_{i}\in \mathbb {Z} $$

[Coin Change 2][coin-change-2]

```java
public int change(int amount, int[] coins) {
    int n = coins.length;

    // dp[i][j]: combinations to make up amount j by using the first i kinds of coins
    int[][] dp = new int[n + 2][amount + 1];
    dp[0][0] = 1;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= amount; j++) {
            if (j < coins[i]) {
                dp[i + 1][j] = dp[i][j];
            } else {
                dp[i + 1][j] = dp[i][j] + dp[i + 1][j - coins[i]];
            }
        }
    }
    return dp[n][amount];
}
```

For example, `amount = 5, coins = [1, 2, 5]`, then `dp` is:

```
[1,0,0,0,0,0]
[1,1,1,1,1,1]
[1,1,2,2,3,3]
[1,1,2,2,3,4]
```

![2D](/assets/img/algorithm/knapsack_coin_change_2_2d.png)

```java
public int change(int amount, int[] coins) {
    int[] dp = new int[amount + 1];
    dp[0] = 1;

    for (int coin : coins) {  
        for (int i = coin; i <= amount; i++) {
            dp[i] += dp[i - coin];
        }
    }
    return dp[amount];
}
```

In 2D, `dp[i + 1][j] = dp[i][j] + dp[i + 1][j - nums[i]]`. The natural iteration ensures `dp[i][j - nums[i]]` is updated to `dp[i + 1][j - nums[i]]` before we update `dp[i][j]` to `dp[i + 1][j]`.

![1D](/assets/img/algorithm/knapsack_coin_change_2_1d.png)

[Number of Ways to Build House of Cards][number-of-ways-to-build-house-of-cards]

```java
public int houseOfCards(int n) {
    int[] dp = new int[n + 1];
    dp[0] = 1;
    // it takes 2 * k - 1 cards to build a row
    // 2, 5, 8, ...
    for (int cards = 2; cards <= n; cards += 3) {
        for (int i = n; i >= cards; i--) {
            dp[i] += dp[i - cards];
        }
    }
    return dp[n];
}
```

[Form Largest Integer With Digits That Add up to Target][form-largest-integer-with-digits-that-add-up-to-target]

```java
public String largestNumber(int[] cost, int target) {
    int n = 9;

    String[][] dp = new String[n + 1][target + 1];
    dp[0][0] = "";

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= target; j++) {
            if (j < cost[i] || dp[i + 1][j - cost[i]] == null) {
                dp[i + 1][j] = dp[i][j];
            } else {
                // i is increasing, and that ensures (i + 1) + dp[i + 1][j - cost[i]] is always the largest
                // among all possible combinations with the same set of digits
                // therefore, no sorting is required
                dp[i + 1][j] = max(dp[i][j], (i + 1) + dp[i + 1][j - cost[i]]);
            }
        }
    }
    return dp[n][target] == null ? "0" : dp[n][target];
}

private String max(String a, String b) {
    return a == null ? b : (a.length() == b.length() ?
        (a.compareTo(b) > 0 ? a : b) :
        (a.length() > b.length() ? a : b));
}
```

## Permutation Sum

The below permutation sum (yes it's permutation, ignore the wrong problem name) is not a knapsack problem, but the only difference is the loop order:

[Combination Sum IV][combination-sum-iv]

```java
public int combinationSum4(int[] nums, int target) {
    int[] dp = new int[target + 1];
    dp[0] = 1;

    for (int i = 0; i <= target; i++) {
        for (int num : nums) {
            if (i >= num) {
                dp[i] += dp[i - num];
            }
        }
    }
    return dp[target];
}
```

In essence, it's bottom-up DP.

Similar problem: [Count Ways To Build Good Strings][count-ways-to-build-good-strings]

```java
    int sum = 0;
    for (int i = 0; i <= high; i++) {
        for (int num : new int[]{zero, one}) {
            if (i >= num) {
                dp[i] = (dp[i] + dp[i - num]) % MOD;
            }
        }
        if (i >= low) {
            sum = (sum + dp[i]) % MOD;
        }
    }
```

Even if `zero` and `one` are equal, they represent different base values. This is a bit different from the requirement of [Combination Sum IV][combination-sum-iv] that all base numbers are distinct.

## Change-making Problem

[Change-making problem](https://en.wikipedia.org/wiki/Change-making_problem): Weakly NP-hard. Find the minimum number of coins (of certain denominations) that add up to a given amount of money. It is a special case of the integer knapsack problem.

minimize $$ f(W)=\sum _{j=1}^{n}x_{j} $$

subject to $$ \sum _{j=1}^{n}w_{j}x_{j}=W $$

[Coin Change][coin-change]

```java
public int coinChange(int[] coins, int amount) {
    int n = coins.length;

    int[][] dp = new int[n + 1][amount + 1];
    for (int i = 0; i < dp.length; i++) {
        Arrays.fill(dp[i], amount + 1);
    }
    for (int i = 0; i < dp.length; i++) {
        dp[i][0] = 0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= amount; j++) {
            if (j < coins[i]) {
                dp[i + 1][j] = dp[i][j];
            } else {
                dp[i + 1][j] = Math.min(dp[i][j], dp[i + 1][j - coins[i]] + 1);
            }
        }
    }
    return dp[n][amount] > amount ? -1 : dp[n][amount];
}
```

```java
public int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1);
    dp[0] = 0;

    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] = Math.min(dp[i], dp[i - coin] + 1);
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}
```

# Summary

|       | 2D     | 1D order |
|-------|--------|----------|
| 0-1 | `dp[i + 1][j] = dp[i][j] + dp[i][j - num]` | j: num <- target |
| UKP | `dp[i + 1][j] = dp[i][j] + dp[i + 1][j - num]` | j: num -> target |

[coin-change]: https://leetcode.com/problems/coin-change/
[coin-change-2]: https://leetcode.com/problems/coin-change-2/
[combination-sum-iv]: https://leetcode.com/problems/combination-sum-iv/
[count-ways-to-build-good-strings]: https://leetcode.com/problems/count-ways-to-build-good-strings/
[form-largest-integer-with-digits-that-add-up-to-target]: https://leetcode.com/problems/form-largest-integer-with-digits-that-add-up-to-target/
[last-stone-weight-ii]: https://leetcode.com/problems/last-stone-weight-ii/
[maximum-profit-from-trading-stocks]: https://leetcode.com/problems/maximum-profit-from-trading-stocks/
[maximize-total-tastiness-of-purchased-fruits]: https://leetcode.com/problems/maximize-total-tastiness-of-purchased-fruits/
[number-of-great-partitions]: https://leetcode.com/problems/number-of-great-partitions/
[number-of-ways-to-earn-points]: https://leetcode.com/problems/number-of-ways-to-earn-points/
[number-of-ways-to-build-house-of-cards]: https://leetcode.com/problems/number-of-ways-to-build-house-of-cards/
[ones-and-zeroes]: https://leetcode.com/problems/ones-and-zeroes/
[painting-the-walls]: https://leetcode.com/problems/painting-the-walls/
[partition-equal-subset-sum]: https://leetcode.com/problems/partition-equal-subset-sum/
[profitable-schemes]: https://leetcode.com/problems/profitable-schemes/
[split-array-with-same-average]: https://leetcode.com/problems/split-array-with-same-average/
[target-sum]: https://leetcode.com/problems/target-sum/
[toss-strange-coins]: https://leetcode.com/problems/toss-strange-coins/
