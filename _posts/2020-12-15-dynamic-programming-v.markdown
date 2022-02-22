---
layout: post
title:  "Dynamic Programming V"
tag: dynamic programming
---
# At i

[Paint Fence][paint-fence]

![Paint Fence](/assets/paint_fence.png)

{% highlight java %}
public int numWays(int n, int k) {
    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        return k;
    }

    int[] dp = new int[n + 1];
    dp[1] = k;
    dp[2] = k * k;

    for (int i = 3; i <= n; i++) {
        dp[i] = (dp[i - 1] + dp[i - 2]) * (k - 1);
    }
    return dp[n];
}
{% endhighlight %}

Reduced to 0D:

{% highlight java %}
public int numWays(int n, int k) {
    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        return k;
    }

    int prev = k * k, prevPrev = k;
    for (int i = 3; i <= n; i++) {
        int tmp = (prev + prevPrev) * (k - 1);
        prevPrev = prev;
        prev = tmp;
    }
    return prev;
}
{% endhighlight %}

[House Robber][house-robber]

{% highlight java %}
public int rob(int[] nums) {
    int n = nums.length;
    int[] dp = new int[n + 1];
    dp[1] = nums[0];
    for (int i = 1; i < n; i++) {
        dp[i + 1] = Math.max(dp[i], dp[i - 1] + nums[i]);
    }       
    return dp[n];
}
{% endhighlight %}

Reduced to 0D:

{% highlight java %}
public int rob(int[] nums) {
    int prev = 0, curr = 0;
    for (int num : nums) {
        int tmp = curr;
        curr = Math.max(curr, prev + num);
        prev = tmp;
    }
    return curr;
}
{% endhighlight %}

[House Robber II][house-robber-ii]

{% highlight java %}
public int rob(int[] nums) {
    if (nums.length == 1) {
        return nums[0];
    }
    return Math.max(rob(nums, 0, nums.length - 1), rob(nums, 1, nums.length));
}

// 198. House Robber
private int rob(int[] nums, int start, int end) {
    int prev = 0, curr = 0;
    for (int i = start; i < end; i++) {
        int tmp = curr;
        curr = Math.max(curr, prev + nums[i]);
        prev = tmp;
    }
    return curr;
}
{% endhighlight %}

[Pizza With 3n Slices][pizza-with-3n-slices]

{% highlight java %}
public int maxSizeSlices(int[] slices) {
    int m = slices.length, n = m / 3;

    // picks n non-adjacent elements from circular array 3n
    // slices[0] and slices[m - 1] can't be chosen at the same time
    int[] slices1 = Arrays.copyOfRange(slices, 0, m - 1);
    int[] slices2 = Arrays.copyOfRange(slices, 1, m);

    return Math.max(maxSizeSlices(slices1, n), maxSizeSlices(slices2, n));
}

private int maxSizeSlices(int[] slices, int n) {
    int m = slices.length;
    // dp[i][j]: maximum sum of j elements from slices[0...(i - 1)]
    int[][] dp = new int[m + 1][n + 1];

    // dp[i][0] = 0
    // dp[0][j] = 0
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (i == 1) {
                // slices has only one element
                dp[i][j] = slices[0];
            } else {
                // skips slices[i - 1]
                // or picks slices[i - 1]
                dp[i][j] = Math.max(dp[i - 1][j], dp[i - 2][j - 1] + slices[i - 1]);
            }
        }
    }
    return dp[m][n];
}
{% endhighlight %}

[Delete and Earn][delete-and-earn]

{% highlight java %}
public int deleteAndEarn(int[] nums) {
    int[] sum = new int[10001];
    for (int num : nums) {
        sum[num] += num;
    }

    int take = 0, skip = 0;
    for (int s : sum) {
        int tmp = skip;
        skip = Math.max(skip, take);
        take = tmp + s;
    }
    return Math.max(take, skip);
}
{% endhighlight %}

[Wiggle Subsequence][wiggle-subsequence]

{% highlight java %}
public int wiggleMaxLength(int[] nums) {
    // max wiggle sequence length so far at index i
    int up = 1, down = 1;
    for (int i = 1; i < nums.length; i++) {
        if (nums[i] > nums[i - 1]) {
            up = down + 1;
        } else if (nums[i] < nums[i - 1]) {
            down = up + 1;
        }
    }
    return Math.max(up, down);
}
{% endhighlight %}

[Flip String to Monotone Increasing][flip-string-to-monotone-increasing]

{% highlight java %}
public int minFlipsMonoIncr(String s) {
    // count of '1's and flips so far
    // to make [0...i] monotone increasing
    int one = 0, flip = 0;
    for (int i = 0; i < s.length(); i++) {
        // [0...i] is monotone increasing
        if (s.charAt(i) == '1') {
            one++;
        } else {
            flip++;
        }

        // either flips all '1's
        // or keeps as is
        flip = Math.min(one, flip);
    }
    return flip;
}
{% endhighlight %}

Similar: [Minimum Deletions to Make String Balanced][minimum-deletions-to-make-string-balanced]

[Number of Sets of K Non-Overlapping Line Segments][number-of-sets-of-k-non-overlapping-line-segments]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int numberOfSets(int n, int k) {
    // dp[][i][j]
    // 0: lines don't start from i
    // 1: lines start from i
    int[][][] dp = new int[2][n][k + 1];
    for (int j = 0; j < n; j++) {
        dp[0][j][0] = 1;
        dp[1][j][0] = 1;
    }

    for (int i = n - 2; i >= 0; i--) {
        for (int j = 1; j <= k; j++) {
            dp[0][i][j] = (dp[0][i + 1][j] + dp[1][i + 1][j]) % MOD;
            dp[1][i][j] = (dp[1][i + 1][j] + dp[0][i][j - 1]) % MOD;
        }
    }

    return (dp[0][0][k] + dp[1][0][k]) % MOD;
}
{% endhighlight %}

[Decode Ways][decode-ways]

{% highlight java %}
public int numDecodings(String s) {
    int n = s.length();
    // dp[i]: number of ways ending at s[i - 1]
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = s.charAt(0) == '0' ? 0 : 1;

    for (int i = 2; i < dp.length; i++) {
        // one digit
        if (s.charAt(i - 1) != '0') {
            dp[i] += dp[i - 1];
        }

        // two digits
        int twoDigits = Integer.valueOf(s.substring(i - 2, i));
        if (twoDigits >= 10 && twoDigits <= 26) {
            dp[i] += dp[i - 2];
        }
    }

    return dp[n];
}
{% endhighlight %}

Reduced to 1D:

{% highlight java %}
public int numDecodings(String s) {
    if (s.charAt(0) == '0') {
        return 0;
    }

    int oneBack = 1, twoBack = 1;
    for (int i = 1; i < s.length(); i++) {
        int curr = 0;
        if (s.charAt(i) != '0') {
            curr = oneBack;
        }

        int twoDigits = Integer.parseInt(s.substring(i - 1, i + 1));
        if (twoDigits >= 10 && twoDigits <= 26) {
            curr += twoBack;
        }

        twoBack = oneBack;
        oneBack = curr;
    }
    return oneBack;
}
{% endhighlight %}

[Number of Ways to Paint N × 3 Grid][number-of-ways-to-paint-n-3-grid]

{% highlight java %}
public int numDecodings(String s) {
    if (s.charAt(0) == '0') {
        return 0;
    }

    int oneBack = 1, twoBack = 1;
    for (int i = 1; i < s.length(); i++) {
        int curr = 0;
        if (s.charAt(i) != '0') {
            curr = oneBack;
        }

        int twoDigits = Integer.parseInt(s.substring(i - 1, i + 1));
        if (twoDigits >= 10 && twoDigits <= 26) {
            curr += twoBack;
        }

        twoBack = oneBack;
        oneBack = curr;
    }
    return oneBack;
}
{% endhighlight %}

[Painting a Grid With Three Different Colors][painting-a-grid-with-three-different-colors]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;
private int m, n;
// memo[j][mask]: the number of ways in the j-th column,
//   while the mask is for the m rows in the previous column.
//   It only stores result when r == 0
private int[][] memo;

public int colorTheGrid(int m, int n) {
    this.m = m;
    this.n = n;
    // for each row, the color is stored in 2 bits
    this.memo = new int[n][1 << (m * 2)];

    return dfs(0, 0, 0, 0);
}


private int dfs(int r, int c, int prev, int curr) {
    // found a valid way
    if (c == n) {
        return 1;
    }

    if (r == 0 && memo[c][prev] > 0) {
        return memo[c][prev];
    }

    // completes the current column
    // proceeds to the next column
    if (r == m) {
        return dfs(0, c + 1, curr, 0);
    }

    int count = 0;
    // color mask:
    // - r: 1
    // - g: 2
    // - b: 3
    for (int color = 1; color <= 3; color++) {
        // - same row in the previous column
        // - same column in the previous row (or the first row)
        if (getColor(prev, r) != color && (r == 0 || getColor(curr, r - 1) != color)) {
            // current row picks this color
            // then dfs the next row in the same column
            count = (count + dfs(r + 1, c, prev, setColor(curr, r, color))) % MOD;
        }
    }

    if (r == 0) {
        memo[c][prev] = count;
    }
    return count;
}

private int getColor(int mask, int pos) {
    return (mask >> (pos * 2)) & 0b11;
}

private int setColor(int mask, int pos, int color) {
    return mask | (color << (pos * 2));
}
{% endhighlight %}

[Paint House II][paint-house-ii]

{% highlight java %}
public int minCostII(int[][] costs) {
    int n = costs.length, k = costs[0].length;
    // min1: 1st smallest cost so far
    // min2: 2nd smallest cost so far
    // it's possible that min1 == min2
    int min1 = 0, min2 = 0;
    // index of min1
    int minIndex = -1;

    // O(nk)
    for (int i = 0; i < n; i++) {
        int currMin1 = Integer.MAX_VALUE, currMin2 = Integer.MAX_VALUE, currMinIndex = 0;
        for (int j = 0; j < k; j++) {
            // if current color j is different from previous min1, picks min1
            // otherwise, picks min2
            int cost = costs[i][j] + (j == minIndex ? min2 : min1);

            // curr becomes min1
            if (cost < currMin1) {
                currMin2 = currMin1;
                currMin1 = cost;
                currMinIndex = j;
            } else if (cost < currMin2) {
                // curr becomes min2
                currMin2 = cost;
            }
        }
        min1 = currMin1;
        min2 = currMin2;
        minIndex = currMinIndex;
    }

    return min1;
}
{% endhighlight %}

[Number of Ways to Stay in the Same Place After Some Steps][number-of-ways-to-stay-in-the-same-place-after-some-steps]

{% highlight java %}
int[][] dp = new int[arrLen][steps];
// reduced to 1D
int[] dp = new int[arrLen];
{% endhighlight %}

[Number of Ways to Form a Target String Given a Dictionary][number-of-ways-to-form-a-target-string-given-a-dictionary]

2D:

{% highlight java %}
// dp[i][j]: number of ways to form target[0...j - 1] with words by the i-th index
long[][] dp = new long[m + 1][n + 1];
{% endhighlight %}

Reduced to 1D:

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int numWays(String[] words, String target) {
    int n = target.length(), m = words[0].length();
    // dp[j]: number of ways to form target.substring(j)
    long[] dp = new long[n + 1];
    dp[0] = 1;

    for (int i = 0; i < m; i++) {
        // freq[j]: total frequency of letter j in the current position i
        int[] freq = new int[26];
        for (int j = 0; j < words.length; j++){
            freq[words[j].charAt(i)-'a']++;
        }

        // iterates backwards
        for (int j = Math.min(i + 1, n); j > 0; j--) {
            dp[j] = (dp[j] + dp[j - 1] * freq[target.charAt(j - 1) -'a']) % MOD;
        }
    }
    return (int)dp[n];
}
{% endhighlight %}

# By i

[Maximum Earnings From Taxi][maximum-earnings-from-taxi]

{% highlight java %}
public long maxTaxiEarnings(int n, int[][] rides) {
    // we can use treemap to find the previous dp value, too
    long[] dp = new long[n + 1];
    Arrays.sort(rides, Comparator.comparingInt(r -> r[1]));

    int prev = 0;
    for (int[] r : rides) {
        if (r[1] != prev) {
            Arrays.fill(dp, prev + 1, r[1] + 1, dp[prev]);
        }
        dp[r[1]] = Math.max(dp[r[1]], dp[r[0]] + r[1] - r[0] + r[2]);
        prev = r[1];
    }
    return dp[prev];
}
{% endhighlight %}

[Minimum Time to Remove All Cars Containing Illegal Goods][minimum-time-to-remove-all-cars-containing-illegal-goods]

{% highlight java %}
public int minimumTime(String s) {
    // implicit DP
    // left: number of cars in s[0...i] that contain illegal goods
    int n = s.length(), left = 0, min = n;
    for (int i = 0; i < n; i++) {
        // previous min left + current cost, or
        // removes from start to current consecutively
        left = Math.min(left + (s.charAt(i) - '0') * 2, i + 1);

        // removes s[i + 1] to end consecutively costs n - 1 - i
        // to make things easier, Operation #3 is always considered in left,
        // and right is always Operation #2
        min = Math.min(min, left + n - 1 - i);
    }
    return min;
}
{% endhighlight %}

[decode-ways]: https://leetcode.com/problems/decode-ways/
[delete-and-earn]: https://leetcode.com/problems/delete-and-earn/
[flip-string-to-monotone-increasing]: https://leetcode.com/problems/flip-string-to-monotone-increasing/
[house-robber]: https://leetcode.com/problems/house-robber/
[house-robber-ii]: https://leetcode.com/problems/house-robber-ii/
[maximum-earnings-from-taxi]: https://leetcode.com/problems/maximum-earnings-from-taxi/
[minimum-deletions-to-make-string-balanced]: https://leetcode.com/problems/minimum-deletions-to-make-string-balanced/
[minimum-time-to-remove-all-cars-containing-illegal-goods]: https://leetcode.com/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/
[number-of-sets-of-k-non-overlapping-line-segments]: https://leetcode.com/problems/number-of-sets-of-k-non-overlapping-line-segments/
[number-of-ways-to-form-a-target-string-given-a-dictionary]: https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/
[number-of-ways-to-paint-n-3-grid]: https://leetcode.com/problems/number-of-ways-to-paint-n-3-grid/
[number-of-ways-to-stay-in-the-same-place-after-some-steps]: https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/
[paint-fence]: https://leetcode.com/problems/paint-fence/
[paint-house-ii]: https://leetcode.com/problems/paint-house-ii/
[painting-a-grid-with-three-different-colors]: https://leetcode.com/problems/painting-a-grid-with-three-different-colors/
[pizza-with-3n-slices]: https://leetcode.com/problems/pizza-with-3n-slices/
[wiggle-subsequence]: https://leetcode.com/problems/wiggle-subsequence/
