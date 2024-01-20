---
title:  "Dynamic Programming (Linear Scan)"
category: algorithm
tag: dynamic programming
---

In this type of problem, we linear scan the elements one by one, and use `dp[i]` to store the state at a certain position. The DP array (1D) can usually be reduced to a DP variable (0D).

The general form of the transaction function is a linear expression, i.e., \\(\sum_{0 \le k \le i}{c_k \cdot dp[i - k]}\\). With a linear scan from left to right, we don't need to care about indices to the right (`k < 0`), because they have been equivalently included when we deal with their counterparts to the left (`-k`). The key is to find the recurrence relation.

The state `dp[i]` can be a single value, or an array (rolling array).

# Single DP Value

[Decode Ways][decode-ways]

```java
public int numDecodings(String s) {
    int n = s.length();
    // dp[i]: number of ways ending at s[i]
    int[] dp = new int[n + 1];
    dp[0] = s.charAt(0) == '0' ? 0 : 1;

    for (int i = 1; i < n; i++) {
        // one digit
        if (s.charAt(i) != '0') {
            dp[i] += dp[i - 1];
        }

        // two digits
        int twoDigits = Integer.valueOf(s.substring(i - 1, i + 1));
        if (twoDigits >= 10 && twoDigits <= 26) {
            dp[i] += i == 1 ? 1 : dp[i - 2];
        }
    }

    return dp[n - 1];
}
```

Reduced to 1D:

```java
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
```

# Rolling DP Array

[Greatest Sum Divisible by Three][greatest-sum-divisible-by-three]

This problem can be generalized to "Divisible by k":

```java
public int maxSumDivThree(int[] nums) {
    int k = 3;
    int[] dp = new int[k];
    for (int num : nums) {
        int[] tmp = Arrays.copyOf(dp, k);
        for (int i = 0; i < k; i++) {
            int r = (dp[i] + num) % k;
            tmp[r] = Math.max(tmp[r], dp[i] + num);
        }
        dp = tmp;
    }
    return dp[0];
}
```

[Sorting Three Groups][sorting-three-groups]

```c++
int minimumOperations(vector<int>& nums) {
    return minimumOperations(nums, 3);
}

int minimumOperations(vector<int>& nums, int k) {
    // dp[i]: min operations if the array constructed so far is non-decreasing
    // and the max group number is at most i + 1
    // (assuming the unconstructed trailing array has all k's)
    vector<int> dp(k, nums.size());

    for (int num : nums) {
        dp[num - 1]--;
        // updates the dp vector in ascending order of group number
        for (int i = 1; i < k; i++) {
            dp[i] = min(dp[i], dp[i - 1]);
        }
    }
    return dp[k - 1];
}
```

[Apply Operations to Make Two Strings Equal][apply-operations-to-make-two-strings-equal]

```c++
int minOperations(string s1, string s2, int x) {
    int n = s1.size(), res = 0, prev = 500, parity = 0;
    for (int i = 0; i < s1.size(); i++) {
        if (s1[i] != s2[i]) {
            int tmp = res;
            res = min(res + x, prev);
            prev = tmp;
            parity ^= 1;
        }
        // prev stores the distance to the previous diff
        prev += 2;
    }
    return parity ? -1 : res / 2;
}
```

# House Robber

In this model, at each position there are multiple choices (e.g. to skip or rob). We need to find out the recurrence relation in each case and combine them.

## Linear

[House Robber][house-robber]

```c++
int rob(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n + 1);
    dp[1] = nums[0];
    for (int i = 1; i < n; i++) {
        // dp[i] = max(dpRob[i], dpSkip[i])
        //       = max(dpRob[i], dp[i - 1])
        //       = max(dpSkip[i - 1] + curr, dp[i - 1])
        //       = max(dp[i - 2] + curr, dp[i - 1])
        dp[i + 1] = max(dp[i], dp[i - 1] + nums[i]);
    }
    return dp[n];
}
```

Reduced to 0D:

```c++
int rob(vector<int>& nums) {
    int prev = 0, curr = 0;
    for (const int& num : nums) {
        int tmp = curr;
        curr = max(curr, prev + num);
        prev = tmp;
    }
    return curr;
}
```

[Count Number of Ways to Place Houses][count-number-of-ways-to-place-houses]

```java
// dp[i] = dpPick[i] + dpSkip[i]
//       = dpSkip[i - 1] + dp[i - 1]
//       = dp[i - 2] + dp[i - 1]
// --> Fibonacci
```

[The Number of Beautiful Subsets][the-number-of-beautiful-subsets]

```java
private Map<Integer, Integer> freqs = new HashMap<>();
private Map<Integer, Integer> memo = new HashMap<>();
private int k;

public int beautifulSubsets(int[] nums, int k) {
    this.k = k;

    for (int num : nums) {
        freqs.put(num, freqs.getOrDefault(num, 0) + 1);
    }

    int res = 1;
    for (var e : freqs.entrySet()) {
        int num = e.getKey();
        if (!freqs.containsKey(num + k)) {
            // +1 because of the empty set
            res *= (dfs(num) + 1);
        }
    }

    return res - 1;
}

private int dfs(int num) {
    if (memo.containsKey(num)) {
        return memo.get(num);
    }

    // the key set of the frequency map is deduped input set.
    // we partition the key set into a few arithmetic sequences,
    // and the difference in each sequence is k
    // we process each sequence with house robber algorithm,
    // and the final result is the product of the results of all sequences.

    // house robber:
    // denotes v = map[num], c = 2 ^ v - 1,
    // where c is the count of all subsets of this "bucket", empty set excluded
    // dp[num] = dpPick[num] * c + dpSkip[num]
    //         = (dpSkip[num - k] + 1) * c + dp[num - k]    // +1 because of possible empty set before num
    //         = dp[num - 2 * k] * c + dp[num - k] + c
    int c = (int)Math.pow(2, freqs.get(num)) - 1;
    int res = c;
    if (!freqs.containsKey(num - k)) {
        memo.put(num, res);
        return res;
    }

    res += dfs(num - k);
    if (freqs.containsKey(num - 2 * k)) {
        res += dfs(num - 2 * k) * c;
    }
    memo.put(num, res);
    return res;
}
```

An alternative solution is backtracking, which is way slow than this `O(n)` solution.

[Paint Fence][paint-fence]

```java
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
        // dp[i] = dpSameAsPrev[i] + dpDiffFromPrev[i]
        //       = dpSameAsPrev[i] + dp[i - 1] * (k - 1)
        //       = dpDiffFromPrev[i - 1] + dp[i - 1] * (k - 1)
        //       = dp[i - 2] * (k - 1) + dp[i - 1] * (k - 1)
        dp[i] = (dp[i - 1] + dp[i - 2]) * (k - 1);
    }
    return dp[n];
}
```

Reduced to 0D:

```java
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
```

Two dimensional fence painting problem:

[Number of Ways to Paint N × 3 Grid][number-of-ways-to-paint-n-3-grid]

```java
private static final int MOD = (int)1e9 + 7;

public int numOfWays(int n) {
    // pattern 1: aba
    // pattern 2: abc
    // next level:
    // - ryr -> yry, yrg, ygy, gry, grg (3 aba, 2 abc)
    // - ryg -> yry, ygr, ygy, gry (2 aba, 2 abc)
    long prev1 = 6, prev2 = 6, curr1 = 0, curr2 = 0;
    for (int i = 1; i < n; i++) {
        curr1 = prev1 * 3 + prev2 * 2;
        curr2 = prev1 * 2 + prev2 * 2;
        prev1 = curr1 % MOD;
        prev2 = curr2 % MOD;
    }
    return (int)((prev1 + prev2) % MOD);
}
```

[Minimum Increment Operations to Make Array Beautiful][minimum-increment-operations-to-make-array-beautiful]

```c++
long long minIncrementOperations(vector<int>& nums, int k) {
    // Consider the last 3 elements a1, a2 and a3 from left to right
    // If we denote "picked" as 1 and "not picked" as 0, and wildcard * means either 0 or 1, then
    //      a1 a2 a3
    // dp1: 1  *  *
    // dp2: 0  1  *
    // dp3: 0  0  1
    long long dp1 = 0, dp2 = 0, dp3 = 0, dp;
    for (int& num : nums) {
        dp = min(dp1, min(dp2, dp3)) + max(k - num, 0);
        dp1 = dp2;
        dp2 = dp3;
        dp3 = dp;
    }
    return min(dp1, min(dp2, dp3));
}
```

**2D**

[Number of Ways to Stay in the Same Place After Some Steps][number-of-ways-to-stay-in-the-same-place-after-some-steps]

```java
private static final int MOD = (int)1e9 + 7;

public int numWays(int steps, int arrLen) {
    // updates arrLen
    arrLen = Math.min(arrLen, steps / 2 + 1);

    // dp[i][j]: number of ways to back index i to index 0 using exactly j steps
    int[][] dp = new int[arrLen][steps + 1];
    dp[0][0] = 1;

    int ways = 0;
    for (int j = 0; j < steps; j++) {
        for (int i = 0; i < arrLen; i++) {
            dp[i][j + 1] = dp[i][j];
            if (i > 0) {
                dp[i][j + 1] = (dp[i][j + 1] + dp[i - 1][j]) % MOD;
            }
            if (i < arrLen - 1) {
                dp[i][j + 1] = (dp[i][j + 1] + dp[i + 1][j]) % MOD;
            }
        }
    }
    return dp[0][steps];
}
```

Reduced to 1D:

```java
private static final int MOD = (int)1e9 + 7;

public int numWays(int steps, int arrLen) {
    // updates arrLen
    arrLen = Math.min(arrLen, steps / 2 + 1);

    // dp[i]: number of ways to back index i to index 0
    int[] dp = new int[arrLen];
    dp[0] = 1;

    int[] tmp = new int[arrLen];
    for (int j = 0; j < steps; j++) {
        System.arraycopy(dp, 0, tmp, 0, arrLen);
        for (int i = 0; i < arrLen; i++) {
            if (i > 0) {
                dp[i] = (dp[i] + tmp[i - 1]) % MOD;
            }
            if (i < arrLen - 1) {
                dp[i] = (dp[i] + tmp[i + 1]) % MOD;
            }
        }
    }
    return dp[0];
}
```

## Circular

[House Robber II][house-robber-ii]

```java
public int rob(int[] nums) {
    int n = nums.length;
    if (n == 1) {
        return nums[0];
    }
    // house[0] and house[n - 1] can't be robbed together, so it can be divided into two subproblems:
    // - nums[0...(n - 2)]
    // - nums[1...(n - 1)]
    // (break the chain!)
    return Math.max(rob(nums, 0, n - 1), rob(nums, 1, n));
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
```

**2D**

[Pizza With 3n Slices][pizza-with-3n-slices]

```java
public int maxSizeSlices(int[] slices) {
    int m = slices.length;

    // picks n non-adjacent elements from circular array 3n
    // slices[0] and slices[m - 1] can't be chosen at the same time
    return Math.max(maxSizeSlices(slices, 0, m - 1), maxSizeSlices(slices, 1, m));
}

private int maxSizeSlices(int[] slices, int start, int end) {
    int m = slices.length, n = m / 3;
    // dp[i][j]: maximum sum of j elements from slices[start...(start + i)]
    int[][] dp = new int[m][n + 1];

    // dp[i][0] = 0
    for (int i = start; i < end; i++) {
        for (int j = 1; j <= n; j++) {
            if (i == start) {
                // slices has only one element
                dp[i][j] = slices[i];
            } else if (i == start + 1) {
                dp[i][j] = Math.max(slices[i - 1], slices[i]);
            } else {
                // skips or picks slices[i]
                dp[i][j] = Math.max(dp[i - 1][j], dp[i - 2][j - 1] + slices[i]);
            }
        }
    }
    return dp[end - 1][n];
}
```

## Tree

[Choose Edges to Maximize Score in a Tree][choose-edges-to-maximize-score-in-a-tree]

```java
public long maxScore(int[][] edges) {
    int n = edges.length;
    // {child, weight}
    List<int[]>[] tree = new List[n];
    for (int i = 0; i < n; i++) {
        tree[i] = new ArrayList<>();
    }

    int root = 0;
    for (int i = 1; i < n; i++) {
        tree[edges[i][0]].add(new int[]{i, edges[i][1]});
    }

    // {with parent edge, without parent edge}
    long[] res = dfs(root, tree);
    return Math.max(res[0], res[1]);
}

private long[] dfs(int node , List<int[]>[] tree) {
    long[] curr = new long[2];
    for (int[] child : tree[node]) {
        long[] next = dfs(child[0], tree);
        curr[0] += next[1];
        // child[1] + next[0] - next[1]
        // = weight - (next[1] - next[0])
        // = weight - (next contains one child edge)
        curr[1] = Math.max(curr[1], child[1] + next[0] - next[1]);
    }

    // curr[1] means no parent edge, so there can be one or no child edge
    // so far, curr[1] contains one child edge
    // now adds the scenario of no child edge (curr[0]) to curr[1]
    curr[1] += curr[0];
    return curr;
}
```

# Multiple DP Variables

In this model, there are multiple choices at the current position, and we assign each choice a dp variable, which essentially is a 0-dimensional dp array. Then we find the relations between the dp variables.

[Wiggle Subsequence][wiggle-subsequence]

```java
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
```

[Flip String to Monotone Increasing][flip-string-to-monotone-increasing]

```java
public int minFlipsMonoIncr(String s) {
    // count of 1's and flips so far
    // to make [0...i] monotone increasing
    int ones = 0, flips = 0;
    for (char ch : s.toCharArray()) {
        if (ch == '1') {
            // no need to flip
            ones++;
        } else {
            // keeps current number as 0, and flips all preceding 1's
            // or flips the current 0 to 1
            flips = Math.min(ones, flips + 1);
        }
    }
    return flips;
}
```

Similar: [Minimum Deletions to Make String Balanced][minimum-deletions-to-make-string-balanced]

[Delete and Earn][delete-and-earn]

```java
private static final int MAX_NUM = (int)1e4;

public int deleteAndEarn(int[] nums) {
    int[] sum = new int[MAX_NUM + 1];
    for (int num : nums) {
        sum[num] += num;
    }

    // to take or skip the prev bucket value
    int take = 0, skip = 0;
    for (int s : sum) {
        int tmp = skip;
        skip = Math.max(skip, take);
        take = tmp + s;
    }
    return Math.max(take, skip);
}
```

[Painting a Grid With Three Different Colors][painting-a-grid-with-three-different-colors]

```java
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
```

[Paint House II][paint-house-ii]

```java
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
```

[Number of Ways to Form a Target String Given a Dictionary][number-of-ways-to-form-a-target-string-given-a-dictionary]

2D:

```java
// dp[i][j]: number of ways to form target[0...j - 1] with words by the i-th index
long[][] dp = new long[m + 1][n + 1];
```

Reduced to 1D:

```java
private static final int MOD = (int)1e9 + 7;

public int numWays(String[] words, String target) {
    int n = target.length(), m = words[0].length();
    // dp[j]: number of ways to form target.substring(j)
    long[] dp = new long[n + 1];
    dp[0] = 1;

    for (int i = 0; i < m; i++) {
        // freq[j]: total frequency of letter j in the current position i
        int[] freq = new int[26];
        for (int j = 0; j < words.length; j++) {
            freq[words[j].charAt(i) - 'a']++;
        }

        // iterates backwards
        for (int j = Math.min(i + 1, n); j > 0; j--) {
            dp[j] = (dp[j] + dp[j - 1] * freq[target.charAt(j - 1) - 'a']) % MOD;
        }
    }
    return (int)dp[n];
}
```

# By i

[Maximum Earnings From Taxi][maximum-earnings-from-taxi]

```java
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
```

[Minimum Time to Remove All Cars Containing Illegal Goods][minimum-time-to-remove-all-cars-containing-illegal-goods]

```java
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
```

[apply-operations-to-make-two-strings-equal]: https://leetcode.com/problems/apply-operations-to-make-two-strings-equal/
[choose-edges-to-maximize-score-in-a-tree]: https://leetcode.com/problems/choose-edges-to-maximize-score-in-a-tree/
[count-number-of-ways-to-place-houses]: https://leetcode.com/problems/count-number-of-ways-to-place-houses/
[decode-ways]: https://leetcode.com/problems/decode-ways/
[delete-and-earn]: https://leetcode.com/problems/delete-and-earn/
[flip-string-to-monotone-increasing]: https://leetcode.com/problems/flip-string-to-monotone-increasing/
[greatest-sum-divisible-by-three]: https://leetcode.com/problems/greatest-sum-divisible-by-three/
[house-robber]: https://leetcode.com/problems/house-robber/
[house-robber-ii]: https://leetcode.com/problems/house-robber-ii/
[maximum-earnings-from-taxi]: https://leetcode.com/problems/maximum-earnings-from-taxi/
[minimum-deletions-to-make-string-balanced]: https://leetcode.com/problems/minimum-deletions-to-make-string-balanced/
[minimum-increment-operations-to-make-array-beautiful]: https://leetcode.com/problems/minimum-increment-operations-to-make-array-beautiful/
[minimum-number-of-coins-for-fruits]: https://leetcode.com/problems/minimum-number-of-coins-for-fruits/
[minimum-time-to-remove-all-cars-containing-illegal-goods]: https://leetcode.com/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/
[number-of-ways-to-form-a-target-string-given-a-dictionary]: https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/
[number-of-ways-to-paint-n-3-grid]: https://leetcode.com/problems/number-of-ways-to-paint-n-3-grid/
[number-of-ways-to-stay-in-the-same-place-after-some-steps]: https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/
[paint-fence]: https://leetcode.com/problems/paint-fence/
[paint-house-ii]: https://leetcode.com/problems/paint-house-ii/
[painting-a-grid-with-three-different-colors]: https://leetcode.com/problems/painting-a-grid-with-three-different-colors/
[pizza-with-3n-slices]: https://leetcode.com/problems/pizza-with-3n-slices/
[sorting-three-groups]: https://leetcode.com/problems/sorting-three-groups/
[the-number-of-beautiful-subsets]: https://leetcode.com/problems/the-number-of-beautiful-subsets/
[wiggle-subsequence]: https://leetcode.com/problems/wiggle-subsequence/
