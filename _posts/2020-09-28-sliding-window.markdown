---
title:  "Sliding Window"
category: algorithm
tag: sliding window
---
# Elastic-size Window

The constraint can be expressed as:

\\[f(v) \ge 0\\]

where \\(v\\) is called _contraint variable_. Constraint variable is determined by the position and size of the sliding window:

\\[v = g(s, m)\\]

where \\(s\\) is the start index and \\(m\\) is the size of the window.

In this type of problems, when \\(s\\) is fixed and \\(m\\) increases, \\(f(v)\\) monotonically increases or decreases. More formally, the following function \\(h(m)\\) is a monotonic function:

\\[f(v) = f(g(m)) = h(m) \ge 0\\]

## Monotonically Decreasing Function

\\(h(m)\\) is a monotonically decreasing function. For example, the constraint is "**at most** k elements". Denote the number of elements in the sliding window as \\(v\\), then \\(f(v) = k - v \ge 0\\). As the window grows, \\(v\\) tends to increase, so \\(f(v)\\) decreases.

### Max Length 

The common steps to resolve the problems:

1. Move the element referenced by the right pointer (`j`) into the sliding window and update related variables (counters, frequency array, etc.)
2. Increment `j` (move right by one)
3. Check if the constraint is satisfied. If not:
  - Move the element referenced by the left pointer (`i`) out of the sliding window and update related variables
  - Increment `i` (move right by one)
4. Repeat the above steps until `j` is out of boundary, the final answer is `j - i`

[Max Consecutive Ones III][max-consecutive-ones-iii]

```java
public int longestOnes(int[] nums, int k) {
    int i = 0, j = 0;
    // The sliding window never shrinks,
    // even if it doesn't meet the requirement at a certain moment
    while (j < nums.length) {
        if (nums[j++] == 0) {
            k--;
        }

        // if k < 0, both i, j move forward together
        // i.e. right shift by one
        if (k < 0 && nums[i++] == 0) {
            k++;
        }
    }

    // [i, j) is a sliding window.
    // Its span memorizes the max range so far
    return j - i;
}
```

[Fruit Into Baskets][fruit-into-baskets]

```java
public int totalFruit(int[] fruits) {
    int n = fruits.length, i = 0, j = 0, k = 2;
    // picks[i]: number of picked fruits of type i
    int[] picks = new int[n];
    while (j < n) {
        if (picks[fruits[j++]]++ == 0) {
            k--;
        }

        if (k < 0 && --picks[fruits[i++]] == 0) {
            k++;
        }
    }
    return j - i;
}
```

[Get Equal Substrings Within Budget][get-equal-substrings-within-budget]

```java
public int equalSubstring(String s, String t, int maxCost) {
    int i = 0, j = 0, cost = 0;
    while (j < s.length()) {
        cost += Math.abs(s.charAt(j) - t.charAt(j));
        j++;

        if (cost > maxCost) {
            cost -= Math.abs(s.charAt(i) - t.charAt(i));
            i++;
        }
    }
    return j - i;
}
```

Sometimes, we need to convert the problem to the "max consecutive ones" model:

[Longest Repeating Character Replacement][longest-repeating-character-replacement]

```java
public int characterReplacement(String s, int k) {
    int[] freqs = new int[26];
    int i = 0, j = 0, maxFreq = 0;
    while (j < s.length()) {
        // Grows the window when the count of the new char exceeds the historical max frequency.
        // So, max frequency is non-decreasing.
        maxFreq = Math.max(maxFreq, ++freqs[s.charAt(j++) - 'A']);

        // Constraint variable: j - i - maxFreq
        // After the element nums[j - 1] was added to the window in this iteration:
        // - If maxFreq remains unchanged, j - i - maxFreq increments by one due to j
        // - If maxFreq increases, it must have incremented by one, and nums[j - 1] must be the new, unique element whose frequency is maxFreq
        //   j - i - maxFreqs remains unchanged
        //
        // Therefore, the sliding window never shrinks, just like the "max consecutive ones" model.
        if (j - i - maxFreq > k) {
            freqs[s.charAt(i++) - 'A']--;
        }
    }
    return j - i;
}
```

[Frequency of the Most Frequent Element][frequency-of-the-most-frequent-element]

```java
public int maxFrequency(int[] nums, int k) {
    Arrays.sort(nums);

    int i = 0, j = 0;
    long availableOps = k;
    while (j < nums.length) {
        availableOps += nums[j];

        // Constraint variable: availableOps - max * length
        if (availableOps < (long)nums[j] * (++j - i)) {
            availableOps -= nums[i++];
        }
    }
    return j - i;
}
```

[Longest Substring with At Most K Distinct Characters][longest-substring-with-at-most-k-distinct-characters]

[Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit][longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit]

```java
public int longestSubarray(int[] nums, int limit) {
    Deque<Integer> maxd = new ArrayDeque<>(), mind = new ArrayDeque<>();
    int i = 0, j = 0;
    while (j < nums.length) {
        while (!maxd.isEmpty() && maxd.peekLast() < nums[j]) {
            maxd.pollLast();
        }
        maxd.add(nums[j]);

        while (!mind.isEmpty() && mind.peekLast() > nums[j]) {
            mind.pollLast();
        }
        mind.add(nums[j]);

        j++;

        if (maxd.peek() - mind.peek() > limit) {
            if (maxd.peek() == nums[i]) {
                maxd.poll();
            }
            if (mind.peek() == nums[i]) {
                mind.poll();
            }
            i++;
        }
    }
    return j - i;
}
```

### Count of Subarrays

The common steps to resolve the problems:

1. Move the element referenced by the right pointer (`j`) into the sliding window and update related variables (counters, frequency array, etc.)
2. Increment `j` (move right by one)
3. Do the following in a loop until the constraint is satisfied:
  - Move the element referenced by the left pointer (`i`) out of the sliding window and update related variables
  - Increment `i` (move right by one)
4. Add `j - i` to the final answer
5. Repeat the above steps until `j` is out of boundary

[Count Complete Subarrays in an Array][count-complete-subarrays-in-an-array]

```java
public int countCompleteSubarrays(int[] nums) {
    int k = Arrays.stream(nums).boxed().collect(Collectors.toSet()).size(), i = 0, j = 0, result = 0;
    Map<Integer, Integer> freqs = new HashMap<>();
    while (j < nums.length) {
        if (Optional.ofNullable(freqs.put(nums[j], freqs.getOrDefault(nums[j++], 0) + 1)).orElse(0) == 0) {
            k--;
        }
        while (k == 0) {
            if (freqs.put(nums[i], freqs.get(nums[i++]) - 1) == 1) {
                k++;
            }
        }
        result += i;
    }
    return result;
}
```

[Subarrays with K Different Integers][subarrays-with-k-different-integers]

```java
public int subarraysWithKDistinct(int[] nums, int k) {
    return atMost(nums, k) - atMost(nums, k - 1);
}

private int atMost(int[] nums, int k) {
    int n = nums.length, i = 0, j = 0, result = 0;
    int[] freqs = new int[n + 1];
    while (j < n) {
        if (freqs[nums[j++]]++ == 0) {
            k--;
        }

        while (k < 0) {
            if (--freqs[nums[i++]] == 0) {
                k++;
            }
        }

        // [i, j) is a sliding window
        // (j - i) represents the count of subarrays that has at most k different integers and end at index j
        // i.e., these subarrays [i, j), [i + 1, j), ..., [j - 1, j)
        // since each loop iteration moves j by one from start to end,
        // the final result will include the counts at all positions of j
        // Fomula: given an array of length n, it will produce (n * (n + 1)) / 2 total contiguous subarrays
        result += j - i;
    }
    return result;
}
```

[Count Number of Nice Subarrays][count-number-of-nice-subarrays]

```java
public int numberOfSubarrays(int[] nums, int k) {
    return atMost(nums, k) - atMost(nums, k - 1);
}

private int atMost(int[] nums, int k) {
    int i = 0, j = 0, result = 0;
    while (j < nums.length) {
        k -= nums[j++] % 2;

        while (k < 0) {
            k += nums[i++] % 2;
        }

        result += j - i;
    }
    return result;
}
```

If we apply `nums[i] -> nums[i] % 2`, the problem becomes [Subarray Sum Equals K][subarray-sum-equals-k]

[Count Subarrays With Score Less Than K][count-subarrays-with-score-less-than-k]

```java
public long countSubarrays(int[] nums, long k) {
    long sum = 0, count = 0;
    int i = 0, j = 0;
    while (j < nums.length) {
        sum += nums[j++];
        while (sum * (j - i) >= k) {
            sum -= nums[i++];
        }
        count += j - i;
    }
    return count;
}
```

[Count Vowel Substrings of a String][count-vowel-substrings-of-a-string]

```java
public int countVowelSubstrings(String word) {
    return atMost(word, 5) - atMost(word, 4);
}

private int atMost(String word, int k) {
    Map<Character, Integer> freq = new HashMap<>();
    int i = 0, j = 0, count = 0;
    while (j < word.length()) {
        char cj = word.charAt(j++);
        // relocates i if the current char is not vowel
        if ("aeiou".indexOf(cj) < 0) {
            freq.clear();
            i = j;
            continue;
        }

        freq.put(cj, freq.getOrDefault(cj, 0) + 1);
        while (freq.size() > k) {
            char ci = word.charAt(i++);
            freq.put(ci, freq.get(ci) - 1);
            freq.remove(ci, 0);
        }
        count += j - i;
    }
    return count;
}
```

[Subarray Product Less Than K][subarray-product-less-than-k]

```java
public int numSubarrayProductLessThanK(int[] nums, int k) {
    if (k <= 1) {
        return 0;
    }

    int i = 0, j = 0, prod = 1, count = 0;
    while (j < nums.length) {
        prod *= nums[j++];
        while (prod >= k) {
            prod /= nums[i++];
        }
        count += j - i;
    }
    return count;
}
```

[Kth Smallest Subarray Sum][kth-smallest-subarray-sum]

```java
private boolean condition(int[] nums, int upper, int k) {
    int i = 0, j = 0, sum = 0, count = 0;
    while (j < nums.length) {
        sum += nums[j++];
        while (sum > upper) {
            sum -= nums[i++];
        }
        count += j - i;
    }
    return count >= k;
}
```

[Find K-th Smallest Pair Distance][find-k-th-smallest-pair-distance]

```java
public int smallestDistancePair(int[] nums, int k) {
    Arrays.sort(nums);

    int low = 0, high = nums[nums.length - 1] - nums[0];
    while (low < high) {
        int mid = (low + high) >>> 1;
        if (condition(nums, mid, k)) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

private boolean condition(int[] nums, int distance, int k) {            
    int i = 0, j = 0, count = 0;
    while (j < nums.length) {
        while (nums[j] - nums[i] > distance) {
            i++;
        }
        // it's not `count += j - i` because it's the count of pairs (start != end)
        // rather than the count of subarrays (start == end is possible)
        count += j++ - i;
    }
    return count >= k;
}
```

[Count the Number of Good Subarrays][count-the-number-of-good-subarrays]

```java
public long countGood(int[] nums, int k) {
    long count = 0;
    int i = 0, j = 0;
    Map<Integer, Integer> freqs = new HashMap<>();
    while (j < nums.length) {
        k -= freqs.getOrDefault(nums[j], 0);
        freqs.put(nums[j], freqs.getOrDefault(nums[j], 0) + 1);
        j++;

        while (k <= 0) {
            freqs.put(nums[i], freqs.get(nums[i]) - 1);
            k += freqs.get(nums[i++]);
        }
        count += i;
    }
    return count;
}
```

## Monotonically Increasing Function

\\(h(m)\\) is a monotonically increasing function. For example, the constraint is "sum is **greater than or equal to** target". Denote the sum of elements in the sliding window as \\(v\\), then \\(f(v) = v - target \ge 0\\). As the window grows, \\(v\\) tends to increase, so \\(f(v)\\) increases.

The common steps to resolve the problems:

1. Move the element referenced by the right pointer (`j`) into the sliding window and update related variables (counters, frequency array, etc.)
2. Increment `j` (move right by one)
3. Do the following in a loop until the constraint is not satisfied (each iteration represents a valid window):
  - Move the element referenced by the left pointer (`i`) out of the sliding window and update related variables
  - Increment `i` (move right by one)
4. Repeat the above steps until `j` is out of boundary

**Min Length**

[Minimum Size Subarray Sum][minimum-size-subarray-sum]

```java
public int minSubArrayLen(int target, int[] nums) {
    int i = 0, j = 0, min = Integer.MAX_VALUE;
    while (j < nums.length) {
        target -= nums[j++];

        while (target <= 0) {
            min = Math.min(min, j - i);
            target += nums[i++];
        }
    }

    return min == Integer.MAX_VALUE ? 0 : min;
}
```

[Replace the Substring for Balanced String][replace-the-substring-for-balanced-string]

```java
public int balancedString(String s) {
    int[] freq = new int[26];
    for (char ch : s.toCharArray()) {
        freq[ch - 'A']++;
    }

    int i = 0, j = 0, n = s.length(), min = n;
    while (j < n) {
        // erases all chars inside the window
        freq[s.charAt(j++) - 'A']--;

        // outside the window, max(count[]) < n / 4
        while (i < n && "QWER".chars().allMatch(ch -> freq[ch - 'A'] <= n / 4)) {
            min = Math.min(min, j - i);
            freq[s.charAt(i++) - 'A']++;
        }
    }
    return min;
}
```

[Minimum Window Substring][minimum-window-substring]

```java
public String minWindow(String s, String t) {
    int[] freq = new int[256];
    for (char ch : t.toCharArray()) {
        freq[ch]++;
    }

    int i = 0, j = 0, k = t.length(), min = s.length();
    String window = "";
    while (j < s.length()) {
        // count of t-chars > 0
        // count of non-t-chars == 0
        if (freq[s.charAt(j++)]-- > 0) {
            // only t-chars will decrement k
            k--;
        }

        while (k == 0) {
            if (j - i <= min) {
                window = s.substring(i, j);
                min = j - i;
            }

            // count of non-t-chars < 0
            if (freq[s.charAt(i++)]++ == 0) {
                // only t-chars will increment k
                k++;
            }
        }
    }
    return window;
}
```

[Minimum Window Subsequence][minimum-window-subsequence]

```java
public String minWindow(String s1, String s2) {
    String window = "";
    int i1 = 0, i2 = 0, min = Integer.MAX_VALUE;
    while (i1 < s1.length()) {
        if (s1.charAt(i1++) == s2.charAt(i2)) {
            // pauses when s2 is fully scanned
            if (++i2 == s2.length()) {
                int j = i1;

                // finds the right most i1 in the window that satisfies
                // s1.charAt(i1) == s2.charAt(0);
                while (--i2 >= 0) {
                    while (s1.charAt(--i1) != s2.charAt(i2)) {
                    }
                }
                i2 = 0;

                if (j - i1 < min) {
                    window = s1.substring(i1, j);
                    min = j - i1;
                }

                // moves i1 forward, so we don't get inifite loop
                // e.g. "abcde" "bde" (i1 == 1)
                i1++;
            }
        }
    }
    return window;
}
```

**Count of Subarrays** 

[Number of Substrings Containing All Three Characters][number-of-substrings-containing-all-three-characters]

This problem is very similar to [Minimum Window Substring][minimum-window-substring]:

```java
public int numberOfSubstrings(String s) {
    int k = 3;
    int[] count = new int[k];
    int i = 0, j = 0, result = 0;
    while (j < s.length()) {
        if (count[s.charAt(j++) - 'a']++ == 0) {
            k--;
        }

        while (k == 0) {
            if (--count[s.charAt(i++) - 'a'] == 0) {
                k++;
            }
        }
        result += i;
    }
    return result;
}
```

## Non-monotonic Function

\\(v = f(s, m)\\) is not a monotonic function of \\(m\\), e.g. "the frequency each character in the substring is greater than or equal to k". When you anchor the start index, expanding the window will either make all chars in thw window have more than k frequency, or introduce a new char so it goes the opposite direction of the goal.

[Longest Substring with At Least K Repeating Characters][longest-substring-with-at-least-k-repeating-characters]

```java
public int longestSubstring(String s, int k) {
    int max = 0;
    // introduces a new constraint `uniqueCharsTarget` so that in sliding window function
    // we know when to move i forward (shrink window) without losing possible candidates
    for (int uniqueCharsTarget = 1; uniqueCharsTarget <= 26; uniqueCharsTarget++) {
        max = Math.max(max, slidingWindow(s, k, uniqueCharsTarget));
    }
    return max;
}

private int slidingWindow(String s, int k, int uniqueCharsTarget) {
    int[] counts = new int[26];
    // unique chars in the window
    // number of unique chars whose frequency >= k
    int uniqueChars = 0, numGeK = 0;
    int i = 0, j = 0, max = 0;
    while (j < s.length()) {
        if (counts[s.charAt(j) - 'a']++ == 0) {
            uniqueChars++;
        }
        if (counts[s.charAt(j++) - 'a'] == k) {
            numGeK++;
        }

        // adjusts the window so that uniqueChars <= uniqueCharsTarget
        while (uniqueChars > uniqueCharsTarget) {
            if (counts[s.charAt(i) - 'a']-- == k) {
                numGeK--;
            }
            if (counts[s.charAt(i++) - 'a'] == 0) {
                uniqueChars--;
            }
        }

        // windows has target number of unique chars and all of them have frequency >= k
        if (uniqueChars == uniqueCharsTarget && uniqueChars == numGeK) {
            max = Math.max(max, j - i);
        }
    }
    return max;
}
```

## Exact-value Constraints

[Minimum Operations to Reduce X to Zero][minimum-operations-to-reduce-x-to-zero]

```java
public int minOperations(int[] nums, int x) {
    int i = 0, j = 0, n = nums.length, sum = Arrays.stream(nums).sum(), min = Integer.MAX_VALUE;
    while (j < n) {
        // sum([0...i) + (j...n - 1]) == x
        sum -= nums[j++];

        while (sum < x && i < j) {
            sum += nums[i++];
        }

        if (sum == x) {
            min = Math.min(min, n - j + i);
        }
    }
    return min == Integer.MAX_VALUE ? -1 : min;
}
```

Similar to: [Maximum Size Subarray Sum Equals k][maximum-size-subarray-sum-equals-k]

# Elastic Size

[Longest Nice Subarray][longest-nice-subarray]

```java
public int longestNiceSubarray(int[] nums) {
    int mask = 0, i = 0, j = 0, max = 0;
    while (j < nums.length) {
        // with this loop, if a bit of mask is set,
        // only one element in the window has that bit set
        while ((mask & nums[j]) != 0) {
            mask ^= nums[i++];
        }

        mask |= nums[j];
        max = Math.max(max, ++j - i);
    }
    return max;
}
```

We can't use "Upper Bound Constraints, Max Length" model, otherwise if a bit of the mask is set, it's possible more than one element in the window have that bit set. Therefore, we don't know what the new mask is like after moving `nums[i++]` out of the window.

[Moving Stones Until Consecutive II][moving-stones-until-consecutive-ii]

```java
public int[] numMovesStonesII(int[] stones) {
    Arrays.sort(stones);

    // sliding window
    int n = stones.length;
    int i = 0, j = 0, min = n;
    while (j < n) {
        // moves i so that the window size is <= n and as close to n as possible
        while (stones[j] - stones[i] >= n) {
            i++;
        }

        // corner case
        // - number of stones in the window is (n - 1)
        // - window size is (n - 1)
        // e.g. [1,2,3,4,10] -> [2,3,4,6,10] -> [2,3,4,5,6]
        if (j - i + 1 == n - 1 && stones[j] - stones[i] == n - 2) {
            min = Math.min(min, 2);
        } else {
            // e.g. [1,2,4,5,10] -> [1,2,3,4,5]
            // e.g. [1,2,3,4,6] -> [2,3,4,5,6]
            // the 2nd example has two windows:
            // - the first window matches the corner case, min = 2;
            // - the second window falls into this else block, min = 1
            min = Math.min(min, n - (j - i + 1));
        }
        j++;
    }

    // moves leftmost or rightmost stone
    // e.g. moves leftmost to the next available slot
    // [1,3,5,10] -> [3,4,5,10] -> [4,5,6,10]
    //
    // max of avaible slots:
    // - left -> right: (stones[n - 1] - stones[1] + 1) - (n - 1)
    // - right -> left: (stones[n - 2] - stones[0] + 1) - (n - 1)
    int max = Math.max(stones[n - 1] - stones[1] - n + 2, stones[n - 2] - stones[0] - n + 2);

    return new int[]{min, max};
}
```

[Delivering Boxes from Storage to Ports][delivering-boxes-from-storage-to-ports]

```java
private static final int MAX_TRIPS = (int)2e5;

public int boxDelivering(int[][] boxes, int portsCount, int maxBoxes, int maxWeight) {
    int n = boxes.length;
    // dp[i]: minimum number of trips to deliver boxes[0, i)
    int[] dp = new int[n + 1];
    Arrays.fill(dp, MAX_TRIPS);
    dp[0] = 0;

    // trips needed to deliver box(i, j]
    int trips = 0, j = 0, prevJ = 0;
    for (int i = 0; i < n; i++) {
        // sliding window
        while (j < n && maxBoxes > 0 && maxWeight >= boxes[j][1]) {
            maxBoxes--;
            maxWeight -= boxes[j][1];

            // current port is different from previous port
            if (j == 0 || boxes[j][0] != boxes[j - 1][0]) {
                prevJ = j;
                trips++;
            }
            j++;
        }

        // delivers boxes[prevJ...j] ('+1')
        dp[j] = Math.min(dp[j], dp[i] + trips + 1);

        // or, don't deliver boxes[prevJ...j] to save one trip (no '+1')
        dp[prevJ] = Math.min(dp[prevJ], dp[i] + trips);

        // gets ready to move the left pointer i forward
        maxBoxes++;
        maxWeight += boxes[i][1];

        // if after moving the left pointer i forward, the port is different
        // then the trips between the new i and j needs to decrement by 1
        if (i < n - 1 && boxes[i][0] != boxes[i + 1][0]) {
            trips--;
        }
    }
    return dp[n];
}
```

# Fixed-size Window

This type of problems can sometimes be solved by prefix sum:

[Maximum Points You Can Obtain from Cards][maximum-points-you-can-obtain-from-cards]

The common steps to resolve the problems:

1. Move the element referenced by the right pointer (`j`) into the sliding window and update related variables (counters, frequency array, etc.)
2. Increment `j` (move right by one)
3. Check if the window expands to the required size. If so:
  - Move the element referenced by the left pointer (`i`) out of the sliding window and update related variables
  - Increment `i` (move right by one)
4. Repeat 1. 2. 3. until `j` is out of boundary

[Minimum Difference Between Largest and Smallest Value in Three Moves][minimum-difference-between-largest-and-smallest-value-in-three-moves]

[Maximum Number of Occurrences of a Substring][maximum-number-of-occurrences-of-a-substring]

If a substring occurs `n` times, any of its substring occurs at least `n` times. So a substring with length `minSize` will have the max occurrences.

[Minimum Swaps to Group All 1's Together][minimum-swaps-to-group-all-1s-together]

```java
public int minSwaps(int[] data) {
    int i = 0, j = 0, sum = Arrays.stream(data).sum(), count = 0, min = sum;
    while (j < data.length) {
        count += data[j++];
        if (j - i == sum) {
            min = Math.min(min, sum - count);
            count -= data[i++];
        }
    }
    return min;
}
```

[Find All Anagrams in a String][find-all-anagrams-in-a-string]

```java
public List<Integer> findAnagrams(String s, String p) {
    int[] count = new int[26];
    for (char ch : p.toCharArray()) {
        count[ch - 'a']++;
    }

    List<Integer> list = new ArrayList<>();
    int i = 0, j = 0, k = p.length();
    while (j < s.length()) {
        if (count[s.charAt(j++) - 'a']-- > 0) {
            k--; 
        }

        if (k == 0) {
            list.add(i);
        }

        // count of chars in p won't go below 0
        if (j - i == p.length() && count[s.charAt(i++) - 'a']++ >= 0) {
            k++;
        }
    }
    return list;
}
```

The above problem is similar to [Minimum Window Substring][minimum-window-substring], but when moving left pointer, we use `if` rather than `while`. That's because the window size is fixed to be the length of `p`, and for a particular `j` we move `i` at most once.

[Number of Equal Count Substrings][number-of-equal-count-substrings]

```java
public int equalCountSubstrings(String s, int count) {
    int result = 0;
    int unique = s.chars().mapToObj(i -> (char)i).collect(Collectors.toSet()).size();
    for (int k = 1; k <= unique; k++) {
        int windowSize = k * count;
        int[] c = new int[26];
        // count of chars in the window is k
        int equalCount = 0;
        for (int j = 0; j < s.length(); j++) {
            if (++c[s.charAt(j) - 'a'] == count) {
                equalCount++;
            }
            if (j >= windowSize && c[s.charAt(j - windowSize) - 'a']-- == count) {
                equalCount--;
            }
            if (equalCount == k) {
                result++;
            }
        }
    }
    return result;
}
```

[Maximum Sum of Distinct Subarrays With Length K][maximum-sum-of-distinct-subarrays-with-length-k]

```java
public long maximumSubarraySum(int[] nums, int k) {
    // the index of the last duplicate element in the window
    int lastDup = -1;
    // last index of an element
    Map<Integer, Integer> map = new HashMap<>();
    // sum: running sum of size-k window
    long sum = 0, max = 0;
    for (int i = 0; i < nums.length; i++) {
        sum += nums[i];
        if (i >= k) {
            sum -= nums[i - k];
        }

        int lastIndex = map.getOrDefault(nums[i], -1);
        if (i - lastIndex < k) {
            lastDup = Math.max(lastDup, lastIndex);
        }

        // attempts to update the max if the last duplicate is out of th window
        if (i - lastDup >= k) {
            max = Math.max(max, sum);
        }

        map.put(nums[i], i);
    }
    return max;
}
```

[Minimum Number of K Consecutive Bit Flips][minimum-number-of-k-consecutive-bit-flips]

```java
public int minKBitFlips(int[] nums, int k) {
    // accumulated is the number of flips contributed by the preceding window nums[i - k + 1, ..., i - 1]
    // flipping the windows starting with these indices will also flip nums[i]
    int accumulated = 0, flips = 0, n = nums.length;
    // sliding window (i - k, i]
    for (int i = 0; i < n; i++) {
        if (i >= k && nums[i - k] > 1) {
            accumulated--;
            nums[i - k] -= 2;
        }

        // needs flipping
        if (accumulated % 2 == nums[i]) {
            if (i + k > n) {
                return -1;
            }

            // a lazy way to mark nums[i] is flipped
            nums[i] += 2;
            accumulated++;
            flips++;
        }
    }
    return flips;
}
```

[Minimum Adjacent Swaps for K Consecutive Ones][minimum-adjacent-swaps-for-k-consecutive-ones]

```java
public int minMoves(int[] nums, int k) {
    if (k == 1) {
        return 0;
    }

    // indexes of ones
    List<Integer> ones = new ArrayList<>();
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] == 1) {
            ones.add(i);
        }
    }

    // prefix sum
    int m = ones.size();
    int[] p = new int[m + 1];
    for (int i = 0; i < m; i++) {
        p[i + 1] = p[i] + ones.get(i);
    }

    int min = Integer.MAX_VALUE;
    // sliding window [i...j] of length k
    for (int i = 0, j = k - 1; j < m; i++, j++) {
        // mid point
        int mid = (i + j) / 2;
        // number of elements on each side
        int radius = mid - i;

        int left = p[mid] - p[i];
        int right = p[j + 1] - p[mid + 1];

        int subtrahend = radius * (radius + 1);
        if (k % 2 == 0) {
            // e.g. [0, 2, 4, 6, 7, 9]
            // k = 6
            //
            // Step 1:
            // -> [4, 4, 4, 4, 4, 4]
            // left = 0 + 2 (+ 4)
            // right = 6 + 7 + 9
            // radius = 2
            // swap = (4 - 0) + (4 - 2) + (9 - 4) + (7 - 4) + (6 - 4)
            //    = (9 + 7 + 6) - (0 + 2 + 4)
            //
            // Step2:
            // -> [2, 3, 4, 5, 6, 7]
            // swap -= 1 + 2 + 0 + 1 + 2 + 3
            //   -= (radius + 1) * radius + (radius + 1)
            left += ones.get(mid);
            subtrahend += radius + 1;
        }
        min = Math.min(min, right - left - subtrahend);
    }
    return min;
}
```

[Minimum Number of Operations to Make Array Continuous][minimum-number-of-operations-to-make-array-continuous]

```java
public int minOperations(int[] nums) {
    Arrays.sort(nums);

    // de-dupe, m is the number of unique elements
    int n = nums.length, m = 1;
    for (int i = 1; i < n; i++) {
        if (nums[i] != nums[i - 1]) {
            nums[m++] = nums[i];
        }
    }

    // uses each num as the start of the range
    // finds the range which requires minimum operations
    int j = 0, min = n;
    for (int i = 0; i < m; i++) {
        // start = nums[i]
        // end = nums[i] + n - 1
        // range = [start, end], len = n
        // finds the first out-of-range element
        while (j < m && nums[j] <= n + nums[i] - 1) {
            j++;
        }

        // number of unique elements in the range is n - j + i
        min = Math.min(min, n - j + i);
    }
    return min;
}
```

[K Empty Slots][k-empty-slots]

```java
public int kEmptySlots(int[] bulbs, int k) {
    int n = bulbs.length;
    // days[i]: the day when bulbs[i] is turned on (1-indexed)
    int[] days =  new int[n];
    for (int i = 0; i < n; i++) {
        days[bulbs[i] - 1] = i + 1;
    }

    // sliding window
    // the goal is find a fixed window in days, whose length is (k + 2)
    // and for all indexes in between (exclusively), days[index] > endpoints
    int min = Integer.MAX_VALUE;
    int left = 0, right = k + 1;
    for (int i = 0; right < n; i++) {
        // current days[i] is valid, continue
        if (days[i] > days[left] && days[i] > days[right]) {
            continue;
        }

        // reaches the right endpoint
        // since all previous number are valid, this is a candidate minimum
        if (i == right) {
            min = Math.min(min, Math.max(days[left], days[right]));
        }

        // not valid, slides the window
        left = i;
        right = k + 1 + i;
    }
    return min == Integer.MAX_VALUE ? -1 : min;
}
```

[Minimum Number of Flips to Make the Binary String Alternating][minimum-number-of-flips-to-make-the-binary-string-alternating]

```java
public int minFlips(String s) {
    // sliding window
    // cyclic problem: s += s
    int n = s.length();
    // flips needed to become "0101..." and "1010..." respectively
    int flips0 = 0, flips1 = 0;
    int flips = n;

    for (int i = 0; i < 2 * n; i++) {
        // the expected char at i-th index of "0101..."
        char c = (char)('0' + i % 2);

        if (c != s.charAt(i % n)) {
            flips0++;
        } else {
            flips1++;
        }

        // i is the end of the window
        if (i >= n) {
            // i % n is outside of the window
            // decrements if it was flipped before
            c = (char)('0' + (i % n) % 2);

            if (c != s.charAt(i % n)) {
                flips0--;
            } else {
                flips1--;
            }

            flips = Math.min(flips, Math.min(flips0, flips1));
        }
    }
    return flips;
}
```

[Substring with Concatenation of All Words][substring-with-concatenation-of-all-words]

```java
private int k = 0, len = 0;
private Map<String, Integer> map = new HashMap<>();
private List<Integer> list = new ArrayList<>();

public List<Integer> findSubstring(String s, String[] words) {
    k = words.length;
    len = words[0].length();

    for (String word : words) {
        map.put(word, map.getOrDefault(word, 0) + 1);
    }

    // possible window start positions are bounded by the word length
    for (int i = 0; i < len; i++) {
        slidingWindow(s, i);
    }
    return list;
}

private void slidingWindow(String s, int i) {
    Map<String, Integer> window = new HashMap<>();
    // used words from the word list
    int used = 0, n = s.length();

    for (int j = i; j <= n - len; j += len) {
        // new word that is to be added to the window
        String newWord = s.substring(j, j + len);
        if (map.containsKey(newWord)) {
            window.put(newWord, window.getOrDefault(newWord, 0) + 1);
            if (window.get(newWord) <= map.get(newWord)) {
                used++;
            } else {
                while (window.get(newWord) > map.get(newWord)) {
                    // old word that is removed from the window
                    String oldWord = s.substring(i, i += len);
                    window.put(oldWord, window.get(oldWord) - 1);
                    // the removed old word was used
                    if (window.get(oldWord) < map.get(oldWord)) {
                        used--;
                    }
                }
            }

            // all words in the word list are used
            if (used == k) {
                list.add(i);
                // moves the left pointer forward
                String oldWord = s.substring(i, i += len);
                window.put(oldWord, window.get(oldWord) - 1);
                used--;
            }
        } else {
            // resets the start of the sliding window
            window.clear();
            used = 0;
            i = j + len;
        }
    }
}
```

# Dynamic Programming

[Maximize Win From Two Segments][maximize-win-from-two-segments]

```java
public int maximizeWin(int[] prizePositions, int k) {
    int n = prizePositions.length, max = 0;
    // dp[i]: in the first i positions, the maximum number of prizes we can get from one segment
    int[] dp = new int[n + 1];
    for (int i = 0, j = 0; j < n; j++) {
        while (prizePositions[i] + k < prizePositions[j]) {
            i++;
        }
        // the prize at index j is either selected or not
        dp[j + 1] = Math.max(dp[j], j - i + 1);
        max = Math.max(max, j - i + 1 + dp[i]);
    }
    return max;
}
```

[Jump Game VII][jump-game-vii]

```java
public boolean canReach(String s, int minJump, int maxJump) {
    // prev is the number of previous positions that we can jump from
    int n = s.length(), prev = 0;
    boolean[] dp = new boolean[n];
    dp[0] = true;

    for (int i = 1; i < n; i++) {
        // checks if there's a true in sliding window dp[i - maxJump : i - minJump]
        if (i >= minJump && dp[i - minJump]) {
            prev++;
        }
        if (i > maxJump && dp[i - maxJump - 1]) {
            prev--;
        }
        dp[i] = prev > 0 && s.charAt(i) == '0';
    }
    return dp[n - 1];
}
```

# Variants

[permutation in string][permutation-in-string]

```java
public boolean checkInclusion(String s1, String s2) {
    int n1 = s1.length(), n2 = s2.length();
    if (n1 > n2) {
        return false;
    }

    int[] map1 = new int[26], map2 = new int[26];
    for (int i = 0; i < n1; i++) {
        map1[s1.charAt(i) - 'a']++;
        map2[s2.charAt(i) - 'a']++;
    }

    int count = 0;
    for (int i = 0; i < 26; i++) {
        if (map1[i] == map2[i]) {
            count++;
        }
    }

    for (int i = 0; i + n1 < n2; i++) {
        int r = s2.charAt(i + n1) - 'a', l = s2.charAt(i) - 'a';
        if (count == 26) {
            return true;
        }

        map2[r]++;
        if (map2[r] == map1[r]) {
            count++;
        } else if (map2[r] == map1[r] + 1) {
            count--;
        }

        map2[l]--;
        if (map2[l] == map1[l]) {
            count++;
        } else if (map2[l] == map1[l] - 1) {
            count--;
        }
    }
    return count == 26;
}
```


# In Batch

[Maximum White Tiles Covered by a Carpet][maximum-white-tiles-covered-by-a-carpet]

```java
public int maximumWhiteTiles(int[][] tiles, int carpetLen) {
    Arrays.sort(tiles, Comparator.comparingInt(t -> t[0]));

    // it's always optimal to place the carpet at the left of tile range
    // i, j are indices of tiles array, not the actual tile position
    int i = 0, j = 0, max = 0, cover = 0;
    while (max < carpetLen && j < tiles.length) {
        if (i == j || tiles[i][0] + carpetLen > tiles[j][1]) {
            // case 1: tiles[j] is the first tile (i == j)
            //   carpet may be longer or shorter than this tile
            //   so picks the min of the two as the covered length
            // case 2: carpet fully covers tiles[j]
            //
            // in either case, moves tiles[j] into the window
            cover += Math.min(carpetLen, tiles[j][1] - tiles[j][0] + 1);
            max = Math.max(max, cover);
            j++;
        } else {
            // partial of tiles[j] is covered by the carpet
            int partial = Math.max(0, tiles[i][0] + carpetLen - tiles[j][0]);
            max = Math.max(max, cover + partial);
            // moves tile[i] out of the window
            cover -= tiles[i][1] - tiles[i][0] + 1;
            i++;
        }
    }
    return max;
}
```

# Dynamic Constraints

In the following example, the constraints are "dynamic" - a series of fixed windows.

[Count Zero Request Servers][count-zero-request-servers]

```java
public int[] countServers(int n, int[][] logs, int x, int[] queries) {
    Arrays.sort(logs, Comparator.comparingInt(l -> l[1]));

    int l = logs.length, m = queries.length;
    Integer[] indices = new Integer[m];
    for (int i = 0; i < m; i++) {
        indices[i] = i;
    }
    Arrays.sort(indices, Comparator.comparingInt(i -> queries[i]));

    int[] arr = new int[m];
    int i = 0, j = 0, k = 0;
    Map<Integer, Integer> freqs = new HashMap<>();
    while (k < m) {
        while (j < l && logs[j][1] <= queries[indices[k]]) {
            freqs.put(logs[j][0], freqs.getOrDefault(logs[j][0], 0) + 1);
            j++;
        }
        while (i < l && logs[i][1] < queries[indices[k]] - x) {
            freqs.put(logs[i][0], freqs.get(logs[i][0]) - 1);
            freqs.remove(logs[i][0], 0);
            i++;
        }
        arr[indices[k++]] = n - freqs.size();
    }
    return arr;
}
```

[count-complete-subarrays-in-an-array]: https://leetcode.com/problems/count-complete-subarrays-in-an-array/
[count-number-of-nice-subarrays]: https://leetcode.com/problems/count-number-of-nice-subarrays/
[count-subarrays-with-score-less-than-k]: https://leetcode.com/problems/count-subarrays-with-score-less-than-k/
[count-the-number-of-good-subarrays]: https://leetcode.com/problems/count-the-number-of-good-subarrays/
[count-vowel-substrings-of-a-string]: https://leetcode.com/problems/count-vowel-substrings-of-a-string/
[count-zero-request-servers]: https://leetcode.com/problems/count-zero-request-servers/
[delivering-boxes-from-storage-to-ports]: https://leetcode.com/problems/delivering-boxes-from-storage-to-ports/
[find-all-anagrams-in-a-string]: https://leetcode.com/problems/find-all-anagrams-in-a-string/
[find-k-th-smallest-pair-distance]: https://leetcode.com/problems/find-k-th-smallest-pair-distance/
[frequency-of-the-most-frequent-element]: https://leetcode.com/problems/frequency-of-the-most-frequent-element/
[fruit-into-baskets]: https://leetcode.com/problems/fruit-into-baskets/
[get-equal-substrings-within-budget]: https://leetcode.com/problems/get-equal-substrings-within-budget/
[jump-game-vii]: https://leetcode.com/problems/jump-game-vii/
[k-empty-slots]: https://leetcode.com/problems/k-empty-slots/
[kth-smallest-subarray-sum]: https://leetcode.com/problems/kth-smallest-subarray-sum/
[longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit]: https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/
[longest-nice-subarray]: https://leetcode.com/problems/longest-nice-subarray/
[longest-repeating-character-replacement]: https://leetcode.com/problems/longest-repeating-character-replacement/
[longest-substring-with-at-least-k-repeating-characters]: https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/
[longest-substring-with-at-most-k-distinct-characters]: https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/
[max-consecutive-ones-iii]: https://leetcode.com/problems/max-consecutive-ones-iii/
[maximize-win-from-two-segments]: https://leetcode.com/problems/maximize-win-from-two-segments/
[maximum-number-of-occurrences-of-a-substring]: https://leetcode.com/problems/maximum-number-of-occurrences-of-a-substring/
[maximum-points-you-can-obtain-from-cards]: https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/
[maximum-size-subarray-sum-equals-k]: https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/
[maximum-sum-of-distinct-subarrays-with-length-k]: https://leetcode.com/problems/maximum-sum-of-distinct-subarrays-with-length-k/
[maximum-white-tiles-covered-by-a-carpet]: https://leetcode.com/problems/maximum-white-tiles-covered-by-a-carpet/
[minimum-adjacent-swaps-for-k-consecutive-ones]: https://leetcode.com/problems/minimum-adjacent-swaps-for-k-consecutive-ones/
[minimum-difference-between-largest-and-smallest-value-in-three-moves]: https://leetcode.com/problems/minimum-difference-between-largest-and-smallest-value-in-three-moves/
[minimum-number-of-flips-to-make-the-binary-string-alternating]: https://leetcode.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/
[minimum-number-of-k-consecutive-bit-flips]: https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/
[minimum-number-of-operations-to-make-array-continuous]: https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/
[minimum-operations-to-reduce-x-to-zero]: https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/
[minimum-size-subarray-sum]: https://leetcode.com/problems/minimum-size-subarray-sum/
[minimum-swaps-to-group-all-1s-together]: https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together/
[minimum-window-subsequence]: https://leetcode.com/problems/minimum-window-subsequence/
[minimum-window-substring]: https://leetcode.com/problems/minimum-window-substring/
[moving-stones-until-consecutive-ii]: https://leetcode.com/problems/moving-stones-until-consecutive-ii/
[number-of-equal-count-substrings]: https://leetcode.com/problems/number-of-equal-count-substrings/
[number-of-substrings-containing-all-three-characters]: https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/
[permutation-in-string]: https://leetcode.com/problems/permutation-in-string/
[replace-the-substring-for-balanced-string]: https://leetcode.com/problems/replace-the-substring-for-balanced-string/
[subarray-product-less-than-k]: https://leetcode.com/problems/subarray-product-less-than-k/submissions/
[subarray-sum-equals-k]: https://leetcode.com/problems/subarray-sum-equals-k/
[subarrays-with-k-different-integers]: https://leetcode.com/problems/subarrays-with-k-different-integers/
[substring-with-concatenation-of-all-words]: https://leetcode.com/problems/substring-with-concatenation-of-all-words/
