---
layout: post
title:  "Sliding Window"
tags: array
---
# Elastic-size Window

## Upper Bound Constraints

Upper bound constraints include e.g. "**at most** k elements".

### Max Length 

The common steps to resolve the problems:

1. Move the element referenced by the right pointer (`j`) into the sliding window and update related variables (counters, frequency array, etc.)
2. Increment `j` (move to right by one)
3. Check if the constraint is satisfied. If not:
  - Move the element referenced by the left pointer (`i`) out of the sliding window and update related variables
  - Increment `i` (move to right by one)
4. Repeat 1. 2. 3. until `j` is out of boundary, the final answer is `j - i`

[Max Consecutive Ones III][max-consecutive-ones-iii]

{% highlight java %}
public int longestOnes(int[] nums, int k) {
    int i = 0, j = 0;
    // the sliding window never shrinks
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
    // its span memorizes the max range so far
    return j - i;
}
{% endhighlight %}

[Fruit Into Baskets][fruit-into-baskets]

{% highlight java %}
public int totalFruit(int[] tree) {
    int[] type = new int[tree.length];
    int k = 2;
    int i = 0, j = 0;
    while (j < tree.length) {
        if (type[tree[j++]]++ == 0) {
            k--;
        }

        if (k < 0 && --type[tree[i++]] == 0) {
            k++;
        }
    }
    return j - i;
}
{% endhighlight %}

[Get Equal Substrings Within Budget][get-equal-substrings-within-budget]

{% highlight java %}
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
{% endhighlight %}

[Longest Repeating Character Replacement][longest-repeating-character-replacement]

{% highlight java %}
public int characterReplacement(String s, int k) {
    int[] count = new int[26];
    int i = 0, j = 0, max = 0;
    while (j < s.length()) {
        // grows the window when the count of the new char exceeds the historical max count
        max = Math.max(max, ++count[s.charAt(j++) - 'A']);

        // count of other chars == j - i - max
        if (j - i - max > k) {
            count[s.charAt(i++) - 'A']--;
        }
    }
    return j - i;
}
{% endhighlight %}

[Frequency of the Most Frequent Element][frequency-of-the-most-frequent-element]

{% highlight java %}
public int maxFrequency(int[] nums, int k) {
    Arrays.sort(nums);

    int i = 0, j = 0;
    long sum = k;    
    while (j < nums.length) {
        sum += nums[j++];

        // constraint: sum >= max * length
        if (sum < (long)nums[j - 1] * (j - i)) {
            sum -= nums[i++];
        }
    }
    return j - i;
}
{% endhighlight %}

[Longest Substring with At Most K Distinct Characters][longest-substring-with-at-most-k-distinct-characters]

[Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit][longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit]

{% highlight java %}
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
{% endhighlight %}

### Count of Subarrays

The common steps to resolve the problems:

1. Move the element referenced by the right pointer (`j`) into the sliding window and update related variables (counters, frequency array, etc.)
2. Increment `j` (move to right by one)
3. Do the following in a loop until the constraint is satisfied:
  - Move the element referenced by the left pointer (`i`) out of the sliding window and update related variables
  - Increment `i` (move to right by one)
4. Add `j - i` to the final answer
5. Repeat 1. 2. 3. 4. until `j` is out of boundary

[Subarrays with K Different Integers][subarrays-with-k-different-integers]

{% highlight java %}
public int subarraysWithKDistinct(int[] nums, int k) {
    return atMost(nums, k) - atMost(nums, k - 1);
}

private int atMost(int[] nums, int k) {
    int n = nums.length;
    int[] count = new int[n + 1];
    int i = 0, j = 0, result = 0;
    while (j < n) {
        if (count[nums[j++]]++ == 0) {
            k--;
        }

        while (k < 0) {
            if (--count[nums[i++]] == 0) {
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
{% endhighlight %}

[Count Number of Nice Subarrays][count-number-of-nice-subarrays]

{% highlight java %}
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
{% endhighlight %}

If we apply `nums[i] -> nums[i] % 2`, the problem becomes [Subarray Sum Equals K][subarray-sum-equals-k]

[Count Subarrays With Score Less Than K][count-subarrays-with-score-less-than-k]

{% highlight java %}
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
{% endhighlight %}

[Count Vowel Substrings of a String][count-vowel-substrings-of-a-string]

{% highlight java %}
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
{% endhighlight %}

[Subarray Product Less Than K][subarray-product-less-than-k]

{% highlight java %}
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
{% endhighlight %}

[Kth Smallest Subarray Sum][kth-smallest-subarray-sum]

{% highlight java %}
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
{% endhighlight %}

[Find K-th Smallest Pair Distance][find-k-th-smallest-pair-distance]

{% highlight java %}
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
{% endhighlight %}

## Lower Bound Constraints

Lower bound constraints include e.g. "sum is **greater than or equal to** target".

The common steps to resolve the problems:

1. Move the element referenced by the right pointer (`j`) into the sliding window and update related variables (counters, frequency array, etc.)
2. Increment `j` (move to right by one)
3. Do the following in a loop until the constraint is not satisfied (each iteration represents a valid subarray):
  - Move the element referenced by the left pointer (`i`) out of the sliding window and update related variables
  - Increment `i` (move to right by one)
4. Repeat 1. 2. 3. until `j` is out of boundary

**Min Length**

[Minimum Size Subarray Sum][minimum-size-subarray-sum]

{% highlight java %}
public int minSubArrayLen(int s, int[] nums) {
    int i = 0, j = 0, min = Integer.MAX_VALUE;
    while (j < nums.length) {
        s -= nums[j++];

        while (s <= 0) {
            min = Math.min(min, j - i);
            s += nums[i++];
        }
    }

    return min == Integer.MAX_VALUE ? 0 : min;
}
{% endhighlight %}

[Replace the Substring for Balanced String][replace-the-substring-for-balanced-string]

{% highlight java %}
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
{% endhighlight %}

[Minimum Window Substring][minimum-window-substring]

{% highlight java %}
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
{% endhighlight %}

[Minimum Window Subsequence][minimum-window-subsequence]

{% highlight java %}
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
{% endhighlight %}

**Count of Subarrays** 

[Number of Substrings Containing All Three Characters][number-of-substrings-containing-all-three-characters]

This problem is very similar to [Minimum Window Substring][minimum-window-substring]:

{% highlight java %}
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
{% endhighlight %}

## Exact-value Constraints

[Minimum Operations to Reduce X to Zero][minimum-operations-to-reduce-x-to-zero]

{% highlight java %}
public int minOperations(int[] nums, int x) {
    int i = 0, j = 0, n = nums.length, sum = Arrays.stream(nums).sum(), min = Integer.MAX_VALUE;
    while (j < nums.length) {
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
{% endhighlight %}

Similar to: [Maximum Size Subarray Sum Equals k][maximum-size-subarray-sum-equals-k]

# Elastic Size

[Moving Stones Until Consecutive II][moving-stones-until-consecutive-ii]

{% highlight java %}
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
{% endhighlight %}

[Delivering Boxes from Storage to Ports][delivering-boxes-from-storage-to-ports]

{% highlight java %}
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
{% endhighlight %}

# Fixed-size Window

This type of problems can sometimes be solved by prefix sum:

[Maximum Points You Can Obtain from Cards][maximum-points-you-can-obtain-from-cards]

The common steps to resolve the problems:

1. Move the element referenced by the right pointer (`j`) into the sliding window and update related variables (counters, frequency array, etc.)
2. Increment `j` (move to right by one)
3. Check if the window expands to the required size. If so:
  - Move the element referenced by the left pointer (`i`) out of the sliding window and update related variables
  - Increment `i` (move to right by one)
4. Repeat 1. 2. 3. until `j` is out of boundary

[Minimum Difference Between Largest and Smallest Value in Three Moves][minimum-difference-between-largest-and-smallest-value-in-three-moves]

[Maximum Number of Occurrences of a Substring][maximum-number-of-occurrences-of-a-substring]

If a substring occurs `n` times, any of its substring occurs at least `n` times. So a substring with length `minSize` will have the max occurrences.

[Minimum Swaps to Group All 1's Together][minimum-swaps-to-group-all-1s-together]

{% highlight java %}
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
{% endhighlight %}

[Find All Anagrams in a String][find-all-anagrams-in-a-string]

{% highlight java %}
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
{% endhighlight %}

The above problem is similar to [Minimum Window Substring][minimum-window-substring], but when moving left pointer, we use `if` rather than `while`. That's because the window size is fixed to be the length of `p`, and for a particular `j` we move `i` at most once.

[Number of Equal Count Substrings][number-of-equal-count-substrings]

{% highlight java %}
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
{% endhighlight %}

[Minimum Number of K Consecutive Bit Flips][minimum-number-of-k-consecutive-bit-flips]

{% highlight java %}
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
{% endhighlight %}

[Minimum Adjacent Swaps for K Consecutive Ones][minimum-adjacent-swaps-for-k-consecutive-ones]

{% highlight java %}
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
{% endhighlight %}

[Minimum Number of Operations to Make Array Continuous][minimum-number-of-operations-to-make-array-continuous]

{% highlight java %}
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
{% endhighlight %}

[K Empty Slots][k-empty-slots]

{% highlight java %}
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
{% endhighlight %}

[Minimum Number of Flips to Make the Binary String Alternating][minimum-number-of-flips-to-make-the-binary-string-alternating]

{% highlight java %}
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
{% endhighlight %}

# Dynamic Programming

[Jump Game VII][jump-game-vii]

{% highlight java %}
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
{% endhighlight %}

# Variants

[permutation in string][permutation-in-string]

{% highlight java %}
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
{% endhighlight %}


# In Batch

[Maximum White Tiles Covered by a Carpet][maximum-white-tiles-covered-by-a-carpet]

{% highlight java %}
public int maximumWhiteTiles(int[][] tiles, int carpetLen) {
    Arrays.sort(tiles, Comparator.comparingInt(t -> t[0]));

    // it's always optimal to place the carpet at the left of tile range
    // i, j are indices of tiles array, not the actual tile position
    int i = 0, j = 0, max = 0, cover = 0;
    while (max < carpetLen && j < tiles.length) {
        if (i == j || tiles[i][0] + carpetLen > tiles[j][1]) {
            // carpet fully covers tiles[j]
            // moves tiles[j] into the window
            cover += Math.min(carpetLen, tiles[j][1] - tiles[j][0] + 1);
            max = Math.max(max, cover);
            j++;
        } else {
            // partial of tiles[j] is covered by the carpet
            int partial = Math.max(0, tiles[i][0] + carpetLen - tiles[j][0]);
            max = Math.max(max, cover + partial);
            // moves tile[j] out of the window
            cover -= tiles[i][1] - tiles[i][0] + 1;
            i++;
        }
    }
    return max;
}
{% endhighlight %}

[count-number-of-nice-subarrays]: https://leetcode.com/problems/count-number-of-nice-subarrays/
[count-subarrays-with-score-less-than-k]: https://leetcode.com/problems/count-subarrays-with-score-less-than-k/
[count-vowel-substrings-of-a-string]: https://leetcode.com/problems/count-vowel-substrings-of-a-string/
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
[longest-repeating-character-replacement]: https://leetcode.com/problems/longest-repeating-character-replacement/
[longest-substring-with-at-most-k-distinct-characters]: https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/
[max-consecutive-ones-iii]: https://leetcode.com/problems/max-consecutive-ones-iii/
[maximum-number-of-occurrences-of-a-substring]: https://leetcode.com/problems/maximum-number-of-occurrences-of-a-substring/
[maximum-points-you-can-obtain-from-cards]: https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/
[maximum-size-subarray-sum-equals-k]: https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/
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
