---
layout: post
title:  "Dynamic Programming III"
tag: dynamic programming
---
[Best Time to Buy and Sell Stock IV][best-time-to-buy-and-sell-stock-iv]

```
dp[k][i] = max(dp[k][i - 1], prices[i] - prices[j] + dp[k - 1][j - 1]), j = [0, 1, ..., i - 1]
```

{% highlight java %}
public int maxProfit(int k, int[] prices) {
    int n = prices.length;

    // we can make maximum number of transactions
    if (k >= n / 2) {
        // 122. Best Time to Buy and Sell Stock II
        int profit = 0;
        for (int i = 1; i < n; i++) {
            profit += Math.max(0, prices[i] - prices[i - 1]);
        }
        return profit;
    }

    // dp[i][j]: max profit on j-th day with i transactions
    int[][] dp = new int[k + 1][n];
    for (int m = 1; m <= k; m++) {
        for (int i = 1; i < n; i++) {
            int min = prices[0];
            for (int j = 1; j <= i; j++) {
                min = Math.min(min, prices[j] - dp[m - 1][j - 1]);
            }
            dp[m][i] = Math.max(dp[m][i - 1], prices[i] - min);
        }
    }
    return dp[k][n - 1];
}
{% endhighlight %}

Reduces the repetitive calculation of min:

{% highlight java %}
for (int m = 1; m <= k; m++) {
    int min = prices[0];
    for (int i = 1; i < n; i++) {
	min = Math.min(min, prices[i] - dp[m - 1][i - 1]);
	dp[m][i] = Math.max(dp[m][i - 1], prices[i] - min);
    }
}
{% endhighlight %}

Swaps the two for-loops. `min` becomes an array to store min of each transaction.

{% highlight java %}
int[] min = new int[k + 1];
Arrays.fill(min, prices[0]);      
for (int i = 1; i < n; i++) {
    for (int j = 1; j <= k; j++) {
	min[j] = Math.min(min[j], prices[i] - dp[j - 1][i - 1]);
	dp[j][i] = Math.max(dp[j][i - 1], prices[i] - min[j]);
    }
}
{% endhighlight %}

Reduces to 1D:

{% highlight java %}
int[] dp = new int[k + 1], min = new int[k + 1];
Arrays.fill(min, prices[0]);      

for (int i = 1; i < prices.length; i++) {
    for (int j = 1; j <= k; j++) {
	min[j] = Math.min(min[j], prices[i] - dp[j - 1]);
	dp[j] = Math.max(dp[j], prices[i] - min[j]);
    }
}        
return dp[k];
{% endhighlight %}

[Best Time to Buy and Sell Stock III][best-time-to-buy-and-sell-stock-iii]

In the last solution of [Best Time to Buy and Sell Stock IV][best-time-to-buy-and-sell-stock-iv], replaces `k` with `2`:

{% highlight java %}
public int maxProfit(int[] prices) {
    int buy1 = Integer.MAX_VALUE, buy2 = Integer.MAX_VALUE;
    int sell1 = 0, sell2 = 0;
    for (int price : prices) {
        buy1 = Math.min(buy1, price);
        sell1 = Math.max(sell1, price - buy1);
        buy2 = Math.min(buy2, price - sell1);
        sell2 = Math.max(sell2, price - buy2);
    }
    return sell2;
}
{% endhighlight %}

[Best Time to Buy and Sell Stock with Cooldown][best-time-to-buy-and-sell-stock-with-cooldown]

```
dp[i] = max(dp[i - 1], prices[i] - prices[j] + dp[j - 2]), j = [0, 1, ..., i - 1]
```

{% highlight java %}
public int maxProfit(int[] prices) {
    if (prices.length < 2) {
        return 0;
    }

    int[] dp = new int[prices.length + 1];
    int min = prices[0];
    for (int i = 1; i < prices.length; i++) {
        min = Math.min(min, prices[i] - dp[i - 1]);
        dp[i + 1] = Math.max(dp[i], prices[i] - min);
    }
    return dp[prices.length];
}
{% endhighlight %}

Reduced to 0D:

{% highlight java %}
int prev = 0, curr = 0;
int min = prices[0];
for (int i = 1; i < prices.length; i++) {
    min = Math.min(min, prices[i] - prev);
    prev = curr;
    curr = Math.max(curr, prices[i] - min);
}
return curr;
{% endhighlight %}

[Number of Dice Rolls With Target Sum][number-of-dice-rolls-with-target-sum]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int numRollsToTarget(int d, int f, int target) {
    int[][] dp = new int[d + 1][target + 1];
    dp[d][0] = 1;
    for (int i = d - 1; i >= 0; i--) {
        for (int j = 1; j <= f; j++) {
            for (int k = 0; k < target; k++) {
                if (j + k <= target) {
                    dp[i][j + k] = (dp[i][j + k] + dp[i + 1][k]) % MOD;
                }
            }
        }
    }
    return dp[0][target];
}
{% endhighlight %}

[Strange Printer][strange-printer]

{% highlight java %}
public int strangePrinter(String s) {
    int n = s.length();
    int[][] dp = new int[n][n];

    for (int i = 0; i < n; i++) {
        dp[i][i] = 1;
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int len = 1; i + len < n; len++) {
            int j = i + len;
            if (s.charAt(i) == s.charAt(j)) {
                // skips same characters
                dp[i][j] = dp[i][j - 1];
            } else {
                // splits the string in many ways
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = i; k + 1 <= j; k++) {
                    dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k + 1][j]);
                }
            }
        }
    }

    return dp[0][n - 1];
}
{% endhighlight %}

# Cumulative Sum

[Dice Roll Simulation][dice-roll-simulation]

![Reduction](/assets/dice_roll_simulation.png)

{% highlight java %}
private final int MOD = (int)1e9 + 7;

public int dieSimulator(int n, int[] rollMax) {
    // dp[i][j]: number of distinct sequences at i-th roll and the last number is (j + 1)
    // if j == 6, it's the total number of distinct sequences at i-th roll
    int[][] dp = new int[n + 1][7];

    // initialization
    dp[0][6] = 1;

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < 6; j++) {
            // if there's no constraint
            dp[i][j] = dp[i - 1][6];

            if (i - rollMax[j] > 0) {
                // e.g. rollMax[j] = 2, and the rolls so far are: a, x, x, b
                // if b == 1, then we should exclude all possible cases of a, 1, 1
                // where a != 1
                int reduction = dp[i - rollMax[j] - 1][6] - dp[i - rollMax[j] - 1][j];
                dp[i][j] = ((dp[i][j] - reduction) % MOD + MOD) % MOD;
            }

            dp[i][6] = (dp[i][6] + dp[i][j]) % MOD;
        }
    }

    return dp[n][6];
}
{% endhighlight %}

[K Inverse Pairs Array][k-inverse-pairs-array]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int kInversePairs(int n, int k) {
    // total number of displacements == number of inverse pairs
    //
    // e.g. a0 = [1, 2, 3, 4]
    // 1) left shifts 2 by 1 position
    //   [2, 1, 3, 4], #inv == 1
    // 2) left shits 2 by 1 position and then 4 by 2 positions
    //   [2, 4, 1, 3], #inv == 3
    //
    // dp[n - 1][] -> dp[n][]:
    // appends the new number to the end
    // e.g. [2, 4, 1, 3] -> [2, 4, 1, 3, 5]
    // it adds no inverse pairs
    // then left shifts 5 to achieve k
    //
    int[][] dp = new int[n + 1][k + 1];
    for (int i = 1; i <= n; i++) {
        // there's only one array with no inverse pairs
        dp[i][0] = 1;
        for (int j = 1; j <= k; j++) {
            // p is the number of displacements
            for (int p = 0; p <= Math.min(j, i - 1); p++) {
                dp[i][j] = (dp[i][j] + dp[i - 1][j - p]) % MOD;
            }
        }
    }
    return dp[n][k];
}
{% endhighlight %}

If `j >= i - 1`,
```
dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1] + ... + dp[i - 1][j - i + 1]
```

Therefore if `j >= i`,
```
dp[i][j - 1] = dp[i - 1][j - 1] + dp[i - 1][j - 2] + ... + dp[i - 1][j - i]
```
```
dp[i][j] - dp[i][j - 1] = dp[i - 1][j] - dp[i - 1][j - i]
dp[i][j] = dp[i][j - 1] + dp[i - 1][j] - dp[i - 1][j - i]
```

{% highlight java %}
int[][] dp = new int[n + 1][k + 1];
dp[0][0] = 1;
for (int i = 1; i <= n; i++) {
    // there's only one array with no inverse pairs
    dp[i][0] = 1;
    for (int j = 1; j <= k; j++) {
        dp[i][j] = (dp[i][j - 1] + dp[i - 1][j]) % MOD;
        if (j - i >= 0) {
            dp[i][j] = (dp[i][j] - dp[i - 1][j - i] + MOD) % MOD; 
        }
    }
}
return dp[n][k];
{% endhighlight %}

[best-time-to-buy-and-sell-stock-iii]: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
[best-time-to-buy-and-sell-stock-iv]: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
[best-time-to-buy-and-sell-stock-with-cooldown]: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
[dice-roll-simulation]: https://leetcode.com/problems/dice-roll-simulation/
[k-inverse-pairs-array]: https://leetcode.com/problems/k-inverse-pairs-array/
[number-of-dice-rolls-with-target-sum]: https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/
[strange-printer]: https://leetcode.com/problems/strange-printer/
