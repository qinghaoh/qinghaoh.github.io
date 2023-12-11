---
title:  "Monotonic Data Structure"
category: algorithm
tags: [stack, queue, map]
---

The following table shows the process of deriving monotonically (strictly) increasing stack from the `[2,5,1,3,6,4]` by iterating from left to right:

|index|element|stack|
|-|-|-|
|0|2|`[2]`|
|1|5|`[2,5]`|
|2|1|`[1]`|
|3|3|`[1,3]`|
|4|6|`[1,3,6]`|
|5|4|`[1,3,4]`|

Here are some _key properties_ of monotonic data structures:

1. The current element is always at the top.
1. The minimum element so far is always at the bottom.
1. The stack at each index can be equivalently derived in this way: iterate _reversely_ from the current index, and push the element into the stack if it's less than the stack top; when this iteration is completed, reverse the entire stack.

# Monoqueue

## Sliding Window Min/Max

Similar to that in [sliding window](../sliding-window/#monotonically-decreasing-function), the constraint function \\(h(m)\\) is monotonically decreasing. Monoqueues can be used to find the min/max of a window constrained by a monotonically decreasing function.

[Sliding Window Maximum][sliding-window-maximum]

```c++
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    int n = nums.size();
    vector<int> res;
    deque<int> dq;
    for (int i = 0; i < n; i++) {
        // Removes the out-of-window number
        if (!dq.empty() && dq.front() == i - k) {
            dq.pop_front();
        }

        // Monotonically decreasing from head to tail
        while (!dq.empty() && nums[i] > nums[dq.back()]) {
            dq.pop_back();
        }
        dq.push_back(i);

        if (i >= k - 1) {
            res.push_back(nums[dq.front()]);
        }
    }
    return res;
}
```

From the above solution, we have _Property #4_:

At index `i`, if we truncate the stack in such a way that all elements with index `< j < i` are removed, the bottom of the remaining stack is the min element of `nums[j...i]`.

Property #2 can be viewed as a special case of Property #4, where `j = 0`.

Equivalently, we can always use _priority queues_ to solve this type of problem. In the following example, we store `[value, position]` pairs in the heap, and pops elements until the distance is within the required range:

[Max Value of Equation][max-value-of-equation]

Monoqueue:

```java
public int findMaxValueOfEquation(int[][] points, int k) {
    // yi + yj + |xi - xj|
    // = yi + yj + xj - xi
    // = (yi - xi) + (xj + yj)

    // {yi - xi , xi}
    Deque<int[]> dq = new ArrayDeque<>();

    int max = Integer.MIN_VALUE;
    for (int[] p : points) {
        while (!dq.isEmpty() && p[0] - dq.peekFirst()[1] > k) {
            dq.pollFirst();
        }
        if (!dq.isEmpty()) {
            max = Math.max(max, dq.peekFirst()[0] + p[0] + p[1]);
        }

        // monotonically decreasing (from first to last)
        while (!dq.isEmpty() && p[1] - p[0] > dq.peekLast()[0]) {
            dq.pollLast();
        }
        dq.offerLast(new int[]{p[1] - p[0], p[0]});
    }
    return max;
}
```

Priority Queue:

```java
public int findMaxValueOfEquation(int[][] points, int k) {
    // yi + yj + |xi - xj|
    // = yi + yj + xj - xi
    // = (yi - xi) + (xj + yj)

    // {yi - xi , xi}
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> -a[0]));

    int max = Integer.MIN_VALUE;
    for (int[] p : points) {
        while (!pq.isEmpty() && p[0] - pq.peek()[1] > k) {
            pq.poll();
        }
        if (!pq.isEmpty()) {
            max = Math.max(max, pq.peek()[0] + p[0] + p[1]);
        }
        pq.offer(new int[]{p[1] - p[0], p[0]});
    }
    return max;
}
```

[Maximum Number of Robots Within Budget][maximum-number-of-robots-within-budget]

See [Sliding window (monotonically decreasing function)](../sliding-window/#monotonically-decreasing-function) for the sliding window template.

```java
public int maximumRobots(int[] chargeTimes, int[] runningCosts, long budget) {
    long sum = 0;
    int i = 0, j = 0, n = chargeTimes.length;
    Deque<Integer> dq = new LinkedList<>();
    while (j < n) {
        sum += runningCosts[j];

        // Monotonically decreasing
        while (!dq.isEmpty() && chargeTimes[j] >= chargeTimes[dq.peekLast()]) {
            dq.pollLast();
        }
        dq.offerLast(j++);

        // Moves i forward
        if (chargeTimes[dq.peekFirst()] + (j - i) * sum > budget) {
            if (dq.peekFirst() == i) {
                dq.pollFirst();
            }
            sum -= runningCosts[i++];
        }
    }

    // Sliding window doesn't shrink
    return j - i;
}
```

[Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit][longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit]

```c++
int longestSubarray(vector<int>& nums, int limit) {
    deque<int> maxd, mind;
    int i = 0, j = 0;
    while (j < nums.size()) {
        while (!maxd.empty() && maxd.back() < nums[j]) {
            maxd.pop_back();
        }
        maxd.push_back(nums[j]);

        while (!mind.empty() && mind.back() > nums[j]) {
            mind.pop_back();
        }
        mind.push_back(nums[j]);
        j++;

        if (maxd.front() - mind.front() > limit) {
            if (maxd.front() == nums[i]) {
                maxd.pop_front();
            }
            if (mind.front() == nums[i]) {
                mind.pop_front();
            }
            i++;
        }
    }
    return j - i;
}
```

This technique can be used to compute the recurrence relation more quickly in some _dynamic programming_ problems.

[Jump Game VI][jump-game-vi]

```java
public int maxResult(int[] nums, int k) {
    int n = nums.length;
    // dp[i]: max score to reach the end starting at index i
    int[] dp = new int[n];
    dp[n - 1] = nums[n - 1];

    Deque<Integer> dq = new ArrayDeque<>();
    for (int i = n - 2; i >= 0; i--) {
        // max(dp[i + 1, i + k])
        if (!dq.isEmpty() && dq.peek() == i + k + 1) {
            dq.poll();
        }

        while (!dq.isEmpty() && dp[i + 1] > dp[dq.peekLast()] {
            dq.pollLast();
        }
        dq.offer(i + 1);

        // Finds max using monoqueue
        dp[i] = nums[i] + dp[dq.peek()];
    }
    return dp[0];
}
```

In the solution above, it's worth noting the iteration is in reverse order, which is more intuitive and straightforward than the natural order.

Similar problem: [Constrained Subsequence Sum][constrained-subsequence-sum]

## Shortest Subarray With Sum >= k

In this type of problems, there is a constraint \\(f(i,j) \ge 0\\), where \\(i\\) is the current index and \\(j\\) is a smaller index. For all indices \\(\in (j, i)\\), \\(f(i,j) \lt 0\\). Monoqueues can be used to find the largest \\(j\\) of each \\(i\\) which satisfies the constraint.

[Shortest Subarray with Sum at Least K][shortest-subarray-with-sum-at-least-k]

```java
public int shortestSubarray(int[] nums, int k) {
    int n = nums.length;
    // Prefix sum
    int[] p = new int[n + 1];
    for (int i = 0; i < n; i++) {
        p[i + 1] = p[i] + nums[i];
    }

    // Finds the smallest j - i so that j > i and p[j] - p[i] >= k
    // We call p[j] minuend and p[i] subtrahend
    int min = n + 1;
    // Strictly increasing monoqueue
    Deque<Integer> dq = new ArrayDeque<>();
    for (int i = 0; i < p.length; i++) {
        // Checks all possible candidates when p[i] is used as the minuend.
        // Pops the head of the deque until p[i] - p[head] < k.
        // The head is a possible candidate.
        while (!dq.isEmpty() && p[i] - p[dq.peekFirst()] >= k) {
            min = Math.min(min, i - dq.pollFirst());
        }

        // The current head is the min in the range of [current head, i],
        // so for all j in this range, p[i] - p[j] < k.
        // Therefore, there's no shorter subarray.
        while (!dq.isEmpty() && p[i] <= p[dq.peekLast()]) {
            dq.pollLast();
        }
        dq.offerLast(i);
    }

    return min <= n ? min : -1;
}
```

[Find Maximum Non-decreasing Array Length][find-maximum-non-decreasing-array-length]

```c++
int findMaximumLength(vector<int>& nums) {
    // dp[i]: max len of a non-decreasing array after applying operations for the first i elements in nums
    // dp[i] = max(dp[j]) + 1, where 0 < j < i and sum(nums[j...(i - 1)]) >= last[j]
    //   last[j] is the last element after applying operations to the first j elements in nums
    //
    // dp[i] is non-decreasing, because we can always append nums[i] to the non-decreasing array
    // converted from nums[0...(i - 1)] and it's still valid. Therefore:
    //
    // dp[i] = dp[j] + 1, where j is the highest index that satisfies:
    //   - 0 < j < i
    //   - p[i] - p[j] >= last[j] (prefix sum) => last[j] + p[j] <= p[i]
    int n = nums.size();
    vector<long long> p(n + 1);
    for (int i = 0; i < n; i++) {
        p[i + 1] = p[i] + nums[i];
    }

    vector<int> dp(n + 1), last(n + 1);
    deque<int> dq;
    for (int i = 1, j = 0; i <= n; i++) {
        // For every index k in the deque, last[k] + p[k] > p[i]
        while (!dq.empty() && last[dq.front()] + p[dq.front()] <= p[i]) {
            j = dq.front();
            dq.pop_front();
        }

        dp[i] = dp[j] + 1;
        last[i] = p[i] - p[j];

        // Monotonically increasing queue
        while (!dq.empty() && last[i] + p[i] <= last[dq.back()] + p[dq.back()]) {
            dq.pop_back();
        }
        dq.push_back(i);
    }
    return dp[n];
}
```

# + Binary Search

[Find Building Where Alice and Bob Can Meet][find-building-where-alice-and-bob-can-meet]

```c++
vector<int> leftmostBuildingQueries(vector<int>& heights, vector<vector<int>>& queries) {
    int m = queries.size();
    // Sorts queries in descending order of `b`,
    // so we process the heights from right to left.
    vector<int> indices, res(m);
    for (int i = 0; i < m; i++) {
        // a <= b
        int a = *ranges::min_element(queries[i]), b = *ranges::max_element(queries[i]);

        if (a == b || heights[a] < heights[b]) {
            res[i] = b;
        } else {
            indices.push_back(i);
        }
    }
    ranges::sort(indices, greater<int>(), [&](int i){ return *ranges::max_element(queries[i]); });

    vector<int> st;
    int j = heights.size() - 1;
    for (const int& i : indices) {
        int a = *ranges::min_element(queries[i]), b = *ranges::max_element(queries[i]);

        // Pushes [j:(b - 1):-1] to maintain a monotonic stack
        while (j >= b) {
            while (!st.empty() && heights[j] >= heights[st.back()]) {
                st.pop_back();
            }
            st.push_back(j--);
        }

        // Binary search
        auto it = upper_bound(rbegin(st), rend(st), a, [&](int i, int j){ return heights[i] < heights[j]; });
        res[i] = it == rend(st) ? -1 : *it;
    }

    return res;
}
```

# Monotonic Map

[Maximum Balanced Subsequence Sum][maximum-balanced-subsequence-sum]

```c++
long long maxBalancedSubsequenceSum(vector<int>& nums) {
    {% raw %}
    map<int, long long> mp{{INT_MIN, 0}};
    {% endraw %}
    for (int i = 0; i < nums.size(); i++) {
        // Considers positive num only
        if (nums[i] > 0) {
            auto it = mp.upper_bound(nums[i] - i);
            long long sum = nums[i] + prev(it)->second;
            mp.insert_or_assign(it, nums[i] - i, sum);
            // Monotonically increasing values
            // Because greater keys with less values are no better than current
            while (it != end(mp) && it->second <= sum) {
                mp.erase(it++);
            }
        }
    }
    return mp.size() > 1 ? rbegin(mp)->second : *max_element(begin(nums), end(nums));
}
```

[Maximum Sum Queries][maximum-sum-queries]

```java
// {y: x + y}, ascending
// candidates for the current query
// monotonic map: keys are ascending, while values are descending
private TreeMap<Integer, Integer> map = new TreeMap<>();

public int[] maximumSumQueries(int[] nums1, int[] nums2, int[][] queries) {
    int n = nums1.length, m = queries.length;
    Integer[] numIndices = new Integer[n], queryIndices = new Integer[m];
    for (int i = 0; i < n; i++) {
        numIndices[i] = i;
    }
    for (int i = 0; i < m; i++) {
        queryIndices[i] = i;
    }

    // iterates nums in descending order of x
    // so we add candidate pairs to map as x decreases, rather than remove candiate pairs
    Arrays.sort(numIndices, Comparator.comparingInt(i -> -nums1[i]));
    Arrays.sort(queryIndices, Comparator.comparingInt(i -> -queries[i][0]));

    int[] answer = new int[m];
    for (int i = 0, j = 0; i < m; i++) {
        int queryIndex = queryIndices[i];
        // "nums1[j]" >= xi
        while (j < n && nums1[numIndices[j]] >= queries[queryIndex][0]) {
            int numIndex = numIndices[j];
            update(nums2[numIndex], nums1[numIndex] + nums2[numIndex]);
            j++;
        }
        // query(yi)
        answer[queryIndices[i]] = query(queries[queryIndex][1]);
    }
    return answer;
}

private void update(int y, int xy) {
    // if the candicates map already contains key y' >= y
    // then x' + y' >= current x + y, as x is in descending order
    // there is no need to put the current pair (y, x + y)
    var e = map.ceilingEntry(y);
    if (e != null && e.getValue() >= xy) {
        return;
    }

    // maintains map values as monotically decreasing
    // just like monotic stack
    e = map.floorEntry(y);
    while (e != null && e.getValue() <= xy) {
        map.remove(e.getKey());
        e = map.floorEntry(y);
    }
    map.put(y, xy);
}

// filters entries with key >= y only
private int query(int y) {
    var e = map.ceilingEntry(y);
    return e == null ? -1 : e.getValue();
}
```

[constrained-subsequence-sum]: https://leetcode.com/problems/constrained-subsequence-sum/
[find-building-where-alice-and-bob-can-meet]: https://leetcode.com/problems/find-building-where-alice-and-bob-can-meet/
[find-maximum-non-decreasing-array-length]: https://leetcode.com/problems/find-maximum-non-decreasing-array-length/
[jump-game-vi]: https://leetcode.com/problems/jump-game-vi/
[longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit]: https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/
[max-value-of-equation]: https://leetcode.com/problems/max-value-of-equation/
[maximum-balanced-subsequence-sum]: https://leetcode.com/problems/maximum-balanced-subsequence-sum/
[maximum-number-of-robots-within-budget]: https://leetcode.com/problems/maximum-number-of-robots-within-budget/
[maximum-sum-queries]: https://leetcode.com/problems/maximum-sum-queries/
[shortest-subarray-with-sum-at-least-k]: https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/
[sliding-window-maximum]: https://leetcode.com/problems/sliding-window-maximum/
