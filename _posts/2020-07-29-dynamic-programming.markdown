---
layout: post
title:  "Dynamic Programming"
tag: dynamic programming
---
[Edit Distance][edit-distance]

{% highlight java %}
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
{% endhighlight %}

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
* `dp[i - 1][j] -> pre[j]`
* `dp[i][j] -> cur[j]`

{% highlight java %}
public int minDistance(String word1, String word2) {
    int n1 = word1.length(), n2 = word2.length();
    int[] pre = new int[n2 + 1], cur = new int[n2 + 1];

    for (int j = 1; j <= n2; j++) {
        pre[j] = j;
    }

    for (int i = 1; i <= n1; i++) {
        cur[0] = i;
        for (int j = 1; j <= n2; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                cur[j] = pre[j - 1];
            } else {
                cur[j] = Math.min(pre[j - 1], Math.min(pre[j], cur[j - 1])) + 1;
            }
        }
        int[] tmp = pre;
        pre = cur;
        cur = tmp;
    }
    return pre[n2];
}
{% endhighlight %}

* `pre[j - 1] -> pre`
* `pre[j] -> cur[j]`
{% highlight java %}
public int minDistance(String word1, String word2) {
    int pre = 0, n1 = word1.length(), n2 = word2.length();
    int[] cur = new int[n2 + 1];

    for (int j = 1; j <= n2; j++) {
        cur[j] = j;
    }

    for (int i = 1; i <= n1; i++) {
        pre = cur[0];
        cur[0] = i;
        for (int j = 1; j <= n2; j++) {
            int tmp = cur[j];
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                cur[j] = pre;
            } else {
                cur[j] = Math.min(pre, Math.min(cur[j], cur[j - 1])) + 1;
            }
            pre = tmp;
        }
    }
    return cur[n2];
}
{% endhighlight %}

[Minimum ASCII Delete Sum for Two Strings][minimum-ascii-delete-sum-for-two-strings]

{% highlight java %}
char c1 = s1.charAt(i - 1), c2 = s2.charAt(j - 1);
if (c1 == c2) {
    dp[i][j] = dp[i - 1][j - 1];
} else {
    dp[i][j] = Math.min(dp[i][j - 1] + c2, dp[i - 1][j] + c1);
}
{% endhighlight %}

**Longest Common Subsequence**

[Longest Common Subsequence][longest-common-subsequence]

{% highlight java %}
if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
    dp[i][j] = dp[i - 1][j - 1] + 1;
} else {
    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
}
{% endhighlight %}

[Uncrossed Lines][uncrossed-lines] can be transformed to [Longest Common Subsequence][longest-common-subsequence]!

**Longest Common Subarray**

[Maximum Length of Repeated Subarray][maximum-length-of-repeated-subarray]

{% highlight java %}
// dp[i][j]: max length of repeated subarray ending with nums1[i - 1] and nums2[j - 1]
int[][] dp = new int[n1 + 1][n2 + 1];
for (int i = 1; i <= n1; i++) {
    for (int j = 1;j <= n2; j++) {
        if (nums1[i - 1] == nums2[j - 1]) {
            max = Math.max(max, dp[i][j] = dp[i - 1][j - 1] + 1);
        }
    }
}
{% endhighlight %}

[Distinct Subsequences][distinct-subsequences]

{% highlight java %}
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
{% endhighlight %}

[Interleaving String][interleaving-string]

{% highlight java %}
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
{% endhighlight %}

Reduced to 1D:

{% highlight java %}
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
{% endhighlight %}

[Longest String Chain][longest-string-chain]

{% highlight java %}
public int longestStrChain(String[] words) {
    Arrays.sort(words, Comparator.comparingInt(s -> s.length()));

    Map<String, Integer> dp = new HashMap<>();
    int max = 0;
    for (String word : words) {
        int length = 0;
        for (int i = 0; i < word.length(); ++i) {
            String predecessor = word.substring(0, i) + word.substring(i + 1);
            length = Math.max(length, dp.getOrDefault(predecessor, 0) + 1);
        }
        dp.put(word, length);
        max = Math.max(max, length);
    }
    return max;
}
{% endhighlight %}

[Make Array Strictly Increasing][make-array-strictly-increasing]

{% highlight java %}
public int makeArrayIncreasing(int[] arr1, int[] arr2) {
    Arrays.sort(arr2);

    // rolling dp
    // dp[i]: i is the element we choose for the current position.
    // this element can be from either arr1 or arr2.
    Map<Integer, Integer> dp = new HashMap<>();
    dp.put(-1, 0);

    for (int a1: arr1) {
        // builds temporary dp map for i-th element of arr1
        Map<Integer, Integer> tmp = new HashMap<>();
        for (int key : dp.keySet()) {
            int val = dp.get(key);
            // path one
            // no assignment for key -> a1
            if (a1 > key) {
                tmp.put(a1, Math.min(tmp.getOrDefault(a1, Integer.MAX_VALUE), val));
            }

            int index = Arrays.binarySearch(arr2, key + 1);
            if (index < 0) {
                index = ~index;
            }

            // path two
            // one assignment for key -> arr2[index]
            if (index < arr2.length) {
                tmp.put(arr2[index], Math.min(tmp.getOrDefault(arr2[index], Integer.MAX_VALUE), val + 1));
            }
        }
        dp = tmp;
    }

    return dp.isEmpty() ? - 1: Collections.min(dp.values());
}
{% endhighlight %}

# Reverse

[Coin Path][coin-path]

{% highlight java %}
public List<Integer> cheapestJump(int[] coins, int maxJump) {
    int n = coins.length;
    List<Integer> path = new ArrayList<>();
    if (coins[n - 1] < 0){
        return path;
    }

    // dp[i]: cost from coins[i] to coins[n - 1]
    int[] dp = new int[n], next = new int[n];
    Arrays.fill(dp, Integer.MAX_VALUE);
    Arrays.fill(next, -1);

    dp[n - 1] = coins[n - 1];

    // reverse order
    for (int i = n - 2; i >= 0; i--) {
        if (coins[i] == -1) {
            continue;
        }

        for (int j = i + 1; j <= Math.min(i + maxJump, n - 1); j++) {
            // strict > guarantees lexicographical order
            if (dp[i] > dp[j] + coins[i] && dp[j] != Integer.MAX_VALUE) {
                dp[i] = dp[j] + coins[i];
                next[i] = j;
            }
        }
    }

    if (dp[0] == Integer.MAX_VALUE) {
        return path;
    }

    int index = 0;
    while (index != -1) {
        path.add(index + 1);
        index = next[index];
    }
    return path;
}
{% endhighlight %}


[Freedom Trail][freedom-trail]

{% highlight java %}
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
{% endhighlight %}

Precomputation:

{% highlight java %}
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
{% endhighlight %}

[coin-path]: https://leetcode.com/problems/coin-path/
[delete-operation-for-two-strings]: https://leetcode.com/problems/delete-operation-for-two-strings/
[distinct-subsequences]: https://leetcode.com/problems/distinct-subsequences/
[edit-distance]: https://leetcode.com/problems/edit-distance/
[freedom-trail]: https://leetcode.com/problems/freedom-trail/
[interleaving-string]: https://leetcode.com/problems/interleaving-string/
[longest-common-subsequence]: https://leetcode.com/problems/longest-common-subsequence/
[longest-string-chain]: https://leetcode.com/problems/longest-string-chain/
[make-array-strictly-increasing]: https://leetcode.com/problems/make-array-strictly-increasing/
[maximum-length-of-repeated-subarray]: https://leetcode.com/problems/maximum-length-of-repeated-subarray/
[minimum-ascii-delete-sum-for-two-strings]: https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/
[uncrossed-lines]: https://leetcode.com/problems/uncrossed-lines/
