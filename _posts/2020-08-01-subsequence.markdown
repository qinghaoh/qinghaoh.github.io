---
title:  "Subsequence"
category: algorithm
tags: sequence
---
# Definition
```
a[i_0], a[i_1], ..., a[i_k]
```
Where `0 <= i_0 < i_1 < ... < i_k <= a.length`

# Algorithm

## Greedy

[Shortest Impossible Sequence of Rolls][shortest-impossible-sequence-of-rolls]

```java
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
```

## Sort

[Sum of Subsequence Widths][sum-of-subsequence-widths]

```java
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
```

## Binary Search

[Is Subsequence][is-subsequence]

It's easy to come up with a `O(n)` solution with two poiners.

Follow up:

If there are lots of incoming `s` (e.g. more than one billion), how to check one by one to see if `t` has its subsequence?

Binary Search:

```java
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
```

The above map pattern (character: list of indices) is very useful in many problems.

## LIS (Longest Increasing Subsequence)

### Dynamic Programming

[Longest Increasing Subsequence][longest-increasing-subsequence]

```java
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
```

Similar problems:

[Largest Divisible Subset][largest-divisible-subset]

```java
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
```

[Delete Columns to Make Sorted III][delete-columns-to-make-sorted-iii]

```java
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
```

[Russian Doll Envelopes][russian-doll-envelopes]: 2D

### Patience Sorting

A quicker solution is [Patience sorting](https://en.wikipedia.org/wiki/Patience_sorting). [This](https://www.cs.princeton.edu/courses/archive/spring13/cos423/lectures/LongestIncreasingSubsequence.pdf) is a Princeton lecture for it.

1. Initially, there are no piles. The first card dealt forms a new pile consisting of the single card.
1. Each subsequent card is placed on the leftmost existing pile whose top card has a value greater than or equal to the new card's value, or to the right of all of the existing piles, thus forming a new pile.
1. When there are no more cards remaining to deal, the game ends.

```c++
// O(nlog(n))
int lengthOfLIS(vector<int>& nums) {
    vector<int> piles;
    for (int num : nums) {
        auto it = lower_bound(piles.begin(), piles.end(), num);
        if (it == piles.end()) {
            piles.push_back(num);
        } else {
            *it = num;
        }
    }
    return piles.size();
}
```

### Monotonic Map

### Segment Tree/Fenwick Tree

### Variants

**Longest Non-Decreasing Sequence**

[Find the Longest Valid Obstacle Course at Each Position][find-the-longest-valid-obstacle-course-at-each-position]

```java
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
```

[Minimum Operations to Make the Array K-Increasing][minimum-operations-to-make-the-array-k-increasing]

[Make Array Empty][make-array-empty]

```java
public long countOperationsToEmptyArray(int[] nums) {
    int n = nums.length;
    Integer[] indices = new Integer[n];
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    Arrays.sort(indices, Comparator.comparingInt(i -> nums[i]));

    // without reordering, groups the elements into increasing subsequences
    // e.g. [1, 2, 4, 3, 5, 0] has 3 groups (in sorted order)
    //  - [0]
    //  - [1, 2, 3]
    //  - [4, 5]
    long operations = n;
    for (int i = 1; i < n; i++) {
        if (indices[i] < indices[i - 1]) {
            // the second operation occurs only at group transitions
            // and its count is the count of the remaining elements
            operations += n - i;
        }
    }
    return operations;
}
```

**Mountain Array**

[Minimum Number of Removals to Make Mountain Array][minimum-number-of-removals-to-make-mountain-array]

[Russian Doll Envelopes][russian-doll-envelopes]

```java
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
```

[Minimum Operations to Make a Subsequence][minimum-operations-to-make-a-subsequence]

```java
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
```

## Dynamic Programming

[Shortest Common Supersequence][shortest-common-supersequence]

```java
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
```

[Longest Arithmetic Subsequence][longest-arithmetic-subsequence]

```java
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
```

[Arithmetic Slices II - Subsequence][arithmetic-slices-ii-subsequence]

```c++
int numberOfArithmeticSlices(vector<int>& nums) {
    int n = nums.size();
    // dp[i]: all arithmetic subsequences (len > 1) in nums[0...i]
    // map: <diff, count>
    vector<unordered_map<int, int>> dp(n);

    int cnt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            // Not (long long)(nums[i] - nums[j])
            // e.g. [0,2000000000,-294967296]
            long long diff = static_cast<long long>(nums[i]) - nums[j];

            // Out of 32 bits
            if (diff <= numeric_limits<int>::min() || diff > numeric_limits<int>::max()) {
                continue;
            }

            int d = static_cast<int>(diff);
            // sub: number of subsequences in nums[0...j] with difference d
            int sub = dp[j][d];

            // 1: 2-element slice -> [nums[j], nums[i]]
            dp[i][d] += sub + 1;

            // Accumulates sub would yield all indexes with all differences
            cnt += sub;
        }
    }
    return cnt;
}
```

[Length of Longest Fibonacci Subsequence][length-of-longest-fibonacci-subsequence]

```java
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
```

[Longest Arithmetic Subsequence of Given Difference][longest-arithmetic-subsequence-of-given-difference]

```java
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
```

[Number of Unique Good Subsequences][number-of-unique-good-subsequences]

```java
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
```

[Subsequence With the Minimum Score][subsequence-with-the-minimum-score]

```java
public int minimumScore(String s, String t) {
    int n = s.length(), m = t.length(), index = m - 1;
    // dp[i]: rightmost index of s so that t.substring(i) is a subsequence of s.substring(dp[i])
    int[] dp = new int[m];
    Arrays.fill(dp, -1);
    for (int i = n - 1; i >= 0 && index >= 0; i--) {
        if (s.charAt(i) == t.charAt(index)) {
            dp[index--] = i;
        }
    }

    // t is subsequence of s
    if (index < 0) {
        return 0;
    }

    // as per the dp, if we remove t.substring(0, index + 1), the remaining substring is a subsequence of s
    // so the min score is no worse than (index + 1)
    int min = index + 1;
    // t' = t[:j] + t[k:]
    for (int i = 0, j = 0, k = index + 1; i < n && j < m; i++) {
        if (s.charAt(i) == t.charAt(j)) {
            // as j moves forward, k either stays or moves forward - it nevers goes backward
            // it can be proven by contradiction
            //
            // moves k until its rightmost index in s is greater than i
            // which means t' is a subsequence of s
            while (k < m && dp[k] <= i) {
                k++;
            }
            min = Math.min(min, k - ++j);
        }
    }
    return min;
}
```

## Buckets

[Number of Matching Subsequences][number-of-matching-subsequences]

```java
public int numMatchingSubseq(String s, String[] words) {
    List<Deque<Character>>[] buckets = new List[26];
    for (int i = 0; i < buckets.length; i++) {
        buckets[i] = new ArrayList<>();
    }

    // adds words to buckets based on the first letter
    for (String w : words) {
        buckets[w.charAt(0) - 'a'].add(w.chars().mapToObj(ch -> (char)ch)
                                       .collect(Collectors.toCollection(LinkedList::new)));
    }

    int count = 0;
    for (char ch : s.toCharArray()) {
        List<Deque<Character>> list = buckets[ch - 'a'];
        buckets[ch - 'a'] = new ArrayList<>();
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
```

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

```java
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
```

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
[make-array-empty]: https://leetcode.com/problems/make-array-empty/
[minimum-number-of-removals-to-make-mountain-array]: https://leetcode.com/problems/minimum-number-of-removals-to-make-mountain-array/
[minimum-operations-to-make-a-subsequence]: https://leetcode.com/problems/minimum-operations-to-make-a-subsequence/
[minimum-operations-to-make-the-array-k-increasing]: https://leetcode.com/problems/minimum-operations-to-make-the-array-k-increasing/
[number-of-matching-subsequences]: https://leetcode.com/problems/number-of-matching-subsequences/
[number-of-unique-good-subsequences]: https://leetcode.com/problems/number-of-unique-good-subsequences/
[russian-doll-envelopes]: https://leetcode.com/problems/russian-doll-envelopes/
[shortest-common-subsequence]: https://leetcode.com/problems/shortest-common-subsequence/
[shortest-impossible-sequence-of-rolls]: https://leetcode.com/problems/shortest-impossible-sequence-of-rolls/
[subsequence-with-the-minimum-score]: https://leetcode.com/problems/subsequence-with-the-minimum-score/
[sum-of-subsequence-widths]: https://leetcode.com/problems/sum-of-subsequence-widths/
[shortest-common-supersequence]: https://leetcode.com/problems/shortest-common-supersequence/
