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

## Greedy

[Shortest Impossible Sequence of Rolls][shortest-impossible-sequence-of-rolls]

{% highlight java %}
public int shortestSequence(int[] rolls, int k) {
    Set<Integer> seen = new HashSet<>();
    int len = 1;
    for (int r : rolls) {
        seen.add(r);
        // 1. finds the min index such that all sequences of length 1 can be formed until this index
        // 2. finds the min index such that all sequences of length 2
        // by finding all numbers of [1, k] after the previous min index
        // 3. repeats Step #2
        if (seen.size() == k) {
            len++;
            seen.clear();
        }
    }
    return len;
}
{% endhighlight %}

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
    for (char ch : s.toCharArray()) {
        if (!map.containsKey(ch - 'a')) {
            return false;
        }

        int i = Collections.binarySearch(map.get(ch - 'a'), index);
        if (i < 0) {
            i = ~i;
        }
        if (i == map.get(ch - 'a').size()) {
            return false;
        }
        index = map.get(ch - 'a').get(i) + 1;
    }
    return true;
}
{% endhighlight %}

The above map pattern (character: list of indices) is very useful in many problems.

## Dynamic Programming

### LIS (Longest Increasing Subsequence)

[Longest Increasing Subsequence][longest-increasing-subsequence]

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

[Largest Divisible Subset][largest-divisible-subset]

{% highlight java %}
public List<Integer> largestDivisibleSubset(int[] nums) {
    Arrays.sort(nums);

    int n = nums.length;
    int[] count = new int[n], prev = new int[n];
    int max = 0, index = -1;
    for (int i = 0; i < n; i++) {
        count[i] = 1;
        prev[i] = -1;
        for (int j = i - 1; j >= 0; j--) {
            if (nums[i] % nums[j] == 0) {
                if (count[j] + 1 > count[i]) {
                    count[i] = count[j] + 1;
                    prev[i] = j;
                }
            }
        }
        if (count[i] > max) {
            max = count[i];
            index = i;
        }
    }

    List<Integer> list = new ArrayList<>();
    while (index != -1) {
        list.add(nums[index]);
        index = prev[index];
    }
    return list;
}
{% endhighlight %}

[Delete Columns to Make Sorted III][delete-columns-to-make-sorted-iii]

{% highlight java %}
public int minDeletionSize(String[] strs) {
    int m = strs.length, n = strs[0].length(), min = n - 1;
    int[] dp = new int[n];
    Arrays.fill(dp, 1);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            int k = 0;
            for (k = 0; k < m; k++) {
                if (strs[k].charAt(j) > strs[k].charAt(i)) {
                    break;
                }
            }
            if (k == m) {
                dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }
        min = Math.min(min, n - dp[i]);
    }
    return min;
}
{% endhighlight %}

[Russian Doll Envelopes][russian-doll-envelopes]: 2D

#### Patience Sorting

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

#### Variants

**Longest Non-Decreasing Sequence**

[Find the Longest Valid Obstacle Course at Each Position][find-the-longest-valid-obstacle-course-at-each-position]

{% highlight java %}
public int[] longestObstacleCourseAtEachPosition(int[] obstacles) {
    int n = obstacles.length;
    int[] piles = new int[n], ans = new int[n];
    int count = 0;
    for (int i = 0; i < n; i++) {
        int index = binarySearch(piles, count, obstacles[i]);
        piles[index] = obstacles[i];
        ans[i] = index + 1;
        if (index == count) {
            count++;
        }
    }
    return ans;
}

// finds the first element > target
private int binarySearch(int[] piles, int end, int target) {
    int low = 0, high = end;
    while (low < high) {
        int mid = (low + high) >>> 1;
        if (piles[mid] > target) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}
{% endhighlight %}

[Minimum Operations to Make the Array K-Increasing][minimum-operations-to-make-the-array-k-increasing]

**Mountain Array**

[Minimum Number of Removals to Make Mountain Array][minimum-number-of-removals-to-make-mountain-array]

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

[Minimum Operations to Make a Subsequence][minimum-operations-to-make-a-subsequence]

{% highlight java %}
public int minOperations(int[] target, int[] arr) {
    int n = target.length;
    // since target has distinct elements,
    // stores the index of each target element
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < n; i++) {
        map.put(target[i], i);
    }

    // finds Longest Increasing Subsequence of target element indices in arr
    int lis = 0;
    int[] piles = new int[n];
    for (int a : arr) {
        // ignores arr element that is not in target
        if (map.containsKey(a)) {
            int i = Arrays.binarySearch(piles, 0, lis, map.get(a));
            if (i < 0) {
                i = ~i;
            }

            piles[i] = map.get(a);
            if (i == lis) {
                lis++;
            }
        }
    }
    return n - lis;
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
public int longestArithSeqLength(int[] nums) {
    int n = nums.length;
    // dp[i]: all subsequences in nums[0...i]
    // map -> diff : max length
    Map<Integer, Integer>[] dp = new Map[n];

    int max = 2;
    for (int i = 0; i < n; i++) {
        dp[i] = new HashMap<>();
        for (int j = 0; j < i; j++) {
            int d = nums[i] - nums[j];
            dp[i].put(d, dp[j].getOrDefault(d, 1) + 1);
            max = Math.max(max, dp[i].get(d));
        }
    }

    return max;
}
{% endhighlight %}

[Arithmetic Slices II - Subsequence][arithmetic-slices-ii-subsequence]

{% highlight java %}
public int numberOfArithmeticSlices(int[] nums) {
    int n = nums.length;
    // dp[i]: all subsequences in nums[0...i]
    // map -> diff : count
    Map<Integer, Integer>[] dp = new Map[n];

    int count = 0;
    for (int i = 0; i < n; i++) {
        dp[i] = new HashMap<>(i);
        for (int j = 0; j < i; j++) {
            // not (long)(nums[i] - nums[j])
            // e.g. [0,2000000000,-294967296]
            long diff = (long)nums[i] - nums[j];

            // out of 32 bits
            if (diff <= Integer.MIN_VALUE || diff > Integer.MAX_VALUE) {
                continue;
            }

            int d = (int)diff;
            // sub: number of subsequences in nums[0...j] with difference d
            int sub = dp[j].getOrDefault(d, 0);

            // 1: 2-element slice -> [nums[j], nums[i]]
            dp[i].put(d, dp[i].getOrDefault(d, 0) + sub + 1);

            // accumulates sub would yield all indexes with all differences
            count += sub;
        }
    }

    return count;
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
    // array element : length of longest arithmetic subsequence ending at the key
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

[Number of Unique Good Subsequences][number-of-unique-good-subsequences]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int numberOfUniqueGoodSubsequences(String binary) {
    // count of subsequences ending with 0 and 1, respectively
    int dp0 = 0, dp1 = 0;
    boolean has0 = false;
    for (int i = 0; i < binary.length(); i++) {
        if (binary.charAt(i) == '0') {
            // appends '0'
            dp0 = (dp0 + dp1) % MOD;
            has0 = true;
        } else {
            // appends '1'
            // +1 means adding this new char as the subsequence "1"
            dp1 = (dp0 + dp1 + 1) % MOD;
        }
    }
    return (dp0 + dp1 + (has0 ? 1 : 0)) % MOD;
}
{% endhighlight %}

## Buckets

[Number of Matching Subsequences][number-of-matching-subsequences]

{% highlight java %}
public int numMatchingSubseq(String s, String[] words) {
    List<Deque<Character>>[] buckets = new List[26];
    for (int i = 0; i < buckets.length; i++) {
        buckets[i] = new ArrayList();
    }

    // adds words to buckets based on the first letter
    for (String w : words) {
        buckets[w.charAt(0) - 'a'].add(w.chars().mapToObj(ch -> (char)ch)
                                       .collect(Collectors.toCollection(LinkedList::new)));
    }

    int count = 0;
    for (char ch : s.toCharArray()) {
        List<Deque<Character>> list = buckets[ch - 'a'];
        buckets[ch - 'a'] = new ArrayList();
        for (Deque<Character> q : list) {
            // O(1) removes the first letter
            q.pollFirst();

            if (q.isEmpty()) {  // no more letters
                count++;
            } else {
                // reallocate the truncated word to (other) buckets
                buckets[q.peekFirst() - 'a'].add(q);
            }
        }
    }
    return count;
}
{% endhighlight %}

For example, `s = "abcde", words = ["a","bb","acd","ace"]`. In each iteration:

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

## Backtracking

[Longest Subsequence Repeated k Times][longest-subsequence-repeated-k-times]

{% highlight java %}
private List<String> candidates = new ArrayList<>();

public String longestSubsequenceRepeatedK(String s, int k) {
    int n = s.length(), total = 0;
    int[] freq = new int[26];
    for (int i = 0; i < n; i++) {
        freq[s.charAt(i) - 'a']++;
    }
    for (int i = 0; i < freq.length; i++) {
        freq[i] /= k;
        total += freq[i];
    }

    // generates candidates in length decreasing order
    for (int len = total; len >= 0; len--) {
        backtrack(freq, len, new StringBuilder());
    }

    for (String c : candidates) {
        if (isKSubSequence(s, c, k)) {
            return c;
        }
    }

    return "";
}

private boolean isKSubSequence(String s, String sub, int k) {
    if (sub.length() == 0) {
        return true;
    }

    int count = 0, j = 0;
    for (int i = 0; i < s.length(); i++) {
        if (s.charAt(i) == sub.charAt(j)) {
            j++;
        }

        // a candidate subsequence is found, reset p to find next candidate.
        if (j == sub.length()) {
            count++;
            j = 0;
        }

        if (count >= k) {
            return true;
        }
    }
    return false;
}

private void backtrack(int[] freq, int len, StringBuilder sb) {
    if (len == 0) {
        candidates.add(sb.toString());
        return;
    }

    // generates candidates in lexicographically decreasing order
    for (int i = freq.length - 1; i >= 0; i--) {
        if (freq[i] > 0) {
            sb.append((char)(i + 'a'));
            freq[i]--;

            backtrack(freq, len - 1, sb);

            sb.deleteCharAt(sb.length() - 1);
            freq[i]++;
        }
    }
}
{% endhighlight %}

[arithmetic-slices-ii-subsequence]: https://leetcode.com/problems/arithmetic-slices-ii-subsequence/
[delete-columns-to-make-sorted-iii]: https://leetcode.com/problems/delete-columns-to-make-sorted-iii/
[find-the-longest-valid-obstacle-course-at-each-position]: https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/
[is-subsequence]: https://leetcode.com/problems/is-subsequence/
[largest-divisible-subset]: https://leetcode.com/problems/largest-divisible-subset/
[length-of-longest-fibonacci-subsequence]: https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/
[longest-arithmetic-subsequence]: https://leetcode.com/problems/longest-arithmetic-subsequence/
[longest-arithmetic-subsequence-of-given-difference]: https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/
[longest-increasing-subsequence]: https://leetcode.com/problems/longest-increasing-subsequence/
[longest-subsequence-repeated-k-times]: https://leetcode.com/problems/longest-subsequence-repeated-k-times/
[minimum-number-of-removals-to-make-mountain-array]: https://leetcode.com/problems/minimum-number-of-removals-to-make-mountain-array/
[minimum-operations-to-make-a-subsequence]: https://leetcode.com/problems/minimum-operations-to-make-a-subsequence/
[minimum-operations-to-make-the-array-k-increasing]: https://leetcode.com/problems/minimum-operations-to-make-the-array-k-increasing/
[number-of-matching-subsequences]: https://leetcode.com/problems/number-of-matching-subsequences/
[number-of-unique-good-subsequences]: https://leetcode.com/problems/number-of-unique-good-subsequences/
[russian-doll-envelopes]: https://leetcode.com/problems/russian-doll-envelopes/
[shortest-common-subsequence]: https://leetcode.com/problems/shortest-common-subsequence/
[shortest-impossible-sequence-of-rolls]: https://leetcode.com/problems/shortest-impossible-sequence-of-rolls/
[smallest-range-ii]: https://leetcode.com/problems/smallest-range-ii/
[sum-of-subsequence-widths]: https://leetcode.com/problems/sum-of-subsequence-widths/
[shortest-common-supersequence]: https://leetcode.com/problems/shortest-common-supersequence/
