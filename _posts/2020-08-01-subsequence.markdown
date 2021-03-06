---
layout: post
title:  "Subsequence"
tags: array
---
# Definition
```
a[i_0], a[i_1], ..., a[i_k]
```
Where `0 <= i_0 < i_1 < ... < i_k <= a.length`

# Algorithm

## Sort

[Smallest Range II][smallest-range-ii]

{% highlight java %}
public int smallestRangeII(int[] A, int K) {
    Arrays.sort(A);

    int max = A[A.length - 1], min = A[0], diff = max - min;
    for (int i = 0; i < A.length - 1; i++) {
        max = Math.max(A[A.length - 1], A[i] + 2 * K);
        min = Math.min(A[0] + 2 * K, A[i + 1]);
        diff = Math.min(diff, max - min);
    }
    return diff;
}
{% endhighlight %}

[Sum of Subsequence Widths][sum-of-subsequence-widths]

{% highlight java %}
private final int MOD = (int)1e9 + 7;

public int sumSubseqWidths(int[] A) {
    Arrays.sort(A);

    long sum = 0, c = 1;
    for (int i = 0; i < A.length; i++, c = c * 2 % MOD) {
        // 2 ^ i subsequences where A[i] is max
        // 2 ^ (n - i - 1) subsequences where A[i] is min
        // sum_i (2 ^ (n - i - 1) * A[i]) == sum_i (2 ^ i * A[n - i - 1])
        sum = (sum + c * (A[i] - A[A.length - i - 1])) % MOD;
    }
    return (int)((sum + MOD) % MOD);
}
{% endhighlight %}

## Binary Search

[Is Subsequence][is-subsequence]

It's easy to come up with a `O(n)` solution with two poiners.

Follow up:

If there are lots of incoming `s` (e.g. more than one billion), how to check one by one to see if `t` has its subsequence?

Binary Search:

{% highlight java %}
public boolean isSubsequence(String s, String t) {
    Map<Integer, List<Integer>> map = new HashMap<>();
    for (int i = 0; i < t.length(); i++) {
        map.computeIfAbsent(t.charAt(i) - 'a', k -> new ArrayList<>()).add(i);
    }

    int index = 0;
    for (char c : s.toCharArray()) {
        if (!map.containsKey(c - 'a')) {
            return false;
        }

        int i = Collections.binarySearch(map.get(c - 'a'), index);
        if (i < 0) {
            i = ~i;
        }
        if (i == map.get(c - 'a').size()) {
            return false;
        }
        index = map.get(c - 'a').get(i) + 1;
    }
    return true;
}
{% endhighlight %}

## Dynamic Programming

[Longest Increasing][longest-increasing-subsequence]

{% highlight java %}
// O(n ^ 2)
public int lengthOfLIS(int[] nums) {
    int n = nums.length;
    // dp[i]: LIS ends at index i
    int[] dp = new int[n];

    int max = 0;
    for (int i = 0; i < n; i++) {
        dp[i] = 1;
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }
        max = Math.max(max, dp[i]);
    }

    return max;
}
{% endhighlight %}

Similar problems:
* [Largest Divisible Subset][largest-divisible-subset]
* [Russian Doll Envelopes][russian-doll-envelopes]: 2D

A quicker solution is [Patience sorting](https://en.wikipedia.org/wiki/Patience_sorting). [This](https://www.cs.princeton.edu/courses/archive/spring13/cos423/lectures/LongestIncreasingSubsequence.pdf) is a Princeton lecture for it.

1. Initially, there are no piles. The first card dealt forms a new pile consisting of the single card.
1. Each subsequent card is placed on the leftmost existing pile whose top card has a value greater than or equal to the new card's value, or to the right of all of the existing piles, thus forming a new pile.
1. When there are no more cards remaining to deal, the game ends.

{% highlight java %}
// O(nlog(n))
public int lengthOfLIS(int[] nums) {
    List<Integer> piles = new ArrayList<>(nums.length);
    for (int num : nums) {
        int pile = Collections.binarySearch(piles, num);
        if (pile < 0) {
            pile = ~pile;
        }

        if (pile == piles.size()) {
            piles.add(num);
        } else {
            piles.set(pile, num);
        }
    }
    return piles.size();
}
{% endhighlight %}

Not as intuitive, we can use array instead:

{% highlight java %}
public int lengthOfLIS(int[] nums) {
    int[] piles = new int[nums.length];
    int count = 0;
    for (int num : nums) {
        int i = Arrays.binarySearch(piles, 0, count, num);
        if (i < 0) {
            i = ~i;
        }

        piles[i] = num;
        if (i == count) {
            count++;
        }
    }
    return count;
}
{% endhighlight %}

[Russian Doll Envelopes][russian-doll-envelopes]

{% highlight java %}
public int maxEnvelopes(int[][] envelopes) {
    // ascending in the first dimension and descending in the second
    // so when the first dimension are equal, two envelopes won't be in the same increasing subsequence
    Arrays.sort(envelopes, (a, b) -> a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);

    // extracts the second dimension
    int n = envelopes.length;
    int[] h = new int[n];
    for (int i = 0; i < n; i++) {
        h[i] = envelopes[i][1];
    }

    return lengthOfLIS(h);
}
{% endhighlight %}

[Shortest Common Supersequence][shortest-common-supersequence]

{% highlight java %}
public String shortestCommonSupersequence(String str1, String str2) {
    // longest common subsequence
    int[][] dp = new int[str1.length() + 1][str2.length() + 1];

    for (int i = 1; i < dp.length; i++) {
        for (int j = 1; j < dp[i].length; j++) {
            if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
            }
        }
    }

    StringBuilder sb = new StringBuilder();
    int len = dp[str1.length()][str2.length()];
    int i = str1.length(), j = str2.length(), pi = str1.length(), pj = str2.length();
    while (i > 0 || j > 0) {
        while (j > 0 && dp[i][j - 1] == dp[i][j]) {
            j--;
        }
        if (j < pj) {
            sb.insert(0, str2.substring(j, pj));
        }

        while (i > 0 && dp[i - 1][j] == dp[i][j]) {
            i--;
        }
        if (i < pi) {
            sb.insert(0, str1.substring(i, pi));
        }

        pi = --i;
        pj = --j;
        if (pj >= 0) {
            sb.insert(0, str2.charAt(pj));
        }
    }

    return sb.toString();
}
{% endhighlight %}

[Longest Arithmetic Subsequence][longest-arithmetic-subsequence]

{% highlight java %}
public int longestArithSeqLength(int[] A) {
    // diff : max length
    List<Map<Integer, Integer>> dp = new ArrayList<>();
    int max = 2;
    for (int j = 0; j < A.length; j++) {
        dp.add(new HashMap<>());
        for (int i = 0; i < j; i++) {
            int d = A[j] - A[i];
            dp.get(j).put(d, dp.get(i).getOrDefault(d, 1) + 1);
            max = Math.max(max, dp.get(j).get(d));
        }
    }
    return max;
}
{% endhighlight %}

[Length of Longest Fibonacci Subsequence][length-of-longest-fibonacci-subsequence]

{% highlight java %}
public int lenLongestFibSubseq(int[] A) {
    int[][] dp = new int[A.length][A.length];
    Map<Integer, Integer> index = new HashMap<>();
    // (k, i, j)
    int max = 0;
    for (int j = 0; j < A.length; j++) {
        index.put(A[j], j);
        for (int i = 0; i < j; i++) {
            int k = index.getOrDefault(A[j] - A[i], -1);
            dp[i][j] = (A[j] - A[i] < A[i] && k >= 0) ? dp[k][i] + 1 : 2;
            max = Math.max(max, dp[i][j]);
        }
    }
    return max > 2 ? max : 0;
}
{% endhighlight %}

[Longest Arithmetic Subsequence of Given Difference][longest-arithmetic-subsequence-of-given-difference]

{% highlight java %}
public int longestSubsequence(int[] arr, int difference) {
    Map<Integer, Integer> dp = new HashMap<>();
    int max = 1;
    for (int a : arr) {
        int value = dp.getOrDefault(a - difference, 0) + 1;
        dp.put(a, value);
        max = Math.max(max, value);
    }
    return max;
}
{% endhighlight %}

# Buckets

[Number of Matching Subsequences][number-of-matching-subsequences]

{% highlight java %}
public int numMatchingSubseq(String s, String[] words) {
    int count = 0;
    List<Deque<Character>>[] buckets = new List[26];
    for (int i = 0; i < 26; i++) {
        buckets[i] = new ArrayList();
    }

    // adds words to buckets based on the first letter
    for (String w : words) {
        Deque<Character> q = new ArrayDeque<>();
        for (char c: w.toCharArray()) {
            q.offer(c);
        }
        buckets[w.charAt(0) - 'a'].add(q);
    }

    for (char c : s.toCharArray()) {
        List<Deque<Character>> list = buckets[c - 'a'];
        buckets[c - 'a'] = new ArrayList();
        for (Deque<Character> q : list) {
            // O(1) removes the first letter
            q.pollFirst();

            if (q.size() == 0) {  // no more letters
                count++;
            } else {
                buckets[q.peekFirst() - 'a'].add(q);
            }
        }
    }
    return count;
}
{% endhighlight %}

For example, `s = "abcde", words = ["a","bb","acd","ace"]`

```
'a':  ["(a)", "(a)cd", "(a)ce"]
'b':  ["(b)b"]
```
```
'b':  ["(b)b"]
'c':  ["a(c)d", "a(c)e"]
count: 1
```
```
'b':  ["b(b)"]
'c':  ["a(c)d", "a(c)e"]
count: 1
```
```
'b':  ["b(b)"]
'd':  ["ac(d)"]
'e':  ["ac(e)"]
count: 1
```
```
'b':  ["b(b)"]
'e':  ["ac(e)"]
count: 2
```
```
'b':  ["b(b)"]
count: 3
```

[largest-divisible-subset]: https://leetcode.com/problems/largest-divisible-subset/
[length-of-longest-fibonacci-subsequence]: https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/
[longest-arithmetic-subsequence]: https://leetcode.com/problems/longest-arithmetic-subsequence/
[longest-arithmetic-subsequence-of-given-difference]: https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/
[longest-increasing-subsequence]: https://leetcode.com/problems/longest-increasing-subsequence/
[is-subsequence]: https://leetcode.com/problems/is-subsequence/
[number-of-matching-subsequences]: https://leetcode.com/problems/number-of-matching-subsequences/
[russian-doll-envelopes]: https://leetcode.com/problems/russian-doll-envelopes/
[shortest-common-subsequence]: https://leetcode.com/problems/shortest-common-subsequence/
[smallest-range-ii]: https://leetcode.com/problems/smallest-range-ii/
[sum-of-subsequence-widths]: https://leetcode.com/problems/sum-of-subsequence-widths/
[shortest-common-supersequence]: https://leetcode.com/problems/shortest-common-supersequence/
