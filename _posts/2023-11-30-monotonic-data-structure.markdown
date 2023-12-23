---
title:  "Monotonic Data Structure"
category: algorithm
tags: [stack, queue, map]
---

## Overview

The **template** of monotonically non-strictly increasing stack:

```c++
// Monotonically increasing stack (non-strict)
stack<int> st;
for (int i = 0; i < n; i++) {
    while (!st.isEmpty() && nums[i] < nums[st.top()]) {
        st.pop();
    }
    // It's also very common to push index into the stack
    st.push(nums[i]);
}
```

Variants:
* Monotonically *strictly* increasing stack: `nums[i] <= nums[st.top()]`
* Monotonically (non-strictly) decreasing stack: `nums[i] > nums[st.top()]`

Let's go over the algorithm with an example `[2,5,1,3,6,4]`. The algorithm maintains a monotonically strictly increasing stack while iterating from left to right:

|index|element|stack|
|-|-|-|
|0|2|`[2]`|
|1|5|`[2,5]`|
|2|1|`[1]`|
|3|3|`[1,3]`|
|4|6|`[1,3,6]`|
|5|4|`[1,3,4]`|

Let \\(st(i)\\) denote the stack when the iterator is at index \\(i\\), we have the following **theorem**[^1]:

Define \\(m(j) = \min_{j \le k \le i}(\text{nums}[k])\\), then \\(st(i)\\) is identical to the array \\(m(0),m(1),\cdots,m(i)\\) after de-duplication.

**Proof**

**Corollary 1** The current element `nums[i]` is always at the top.

**Corollary 2** The minimum element of `nums[0...i]` is always at the bottom.

**Corollary 3** The stack is lexicographically minimum subsequence of `nums[0...i]`.

If we truncate the stack in such a way that all elements with index `< j < i` are removed, the bottom of the remaining stack is the min element of `nums[j...i]`.

The stack at each index can be equivalently derived in this way: iterate _reversely_ from the current index, and push the element into the stack if it's less than the stack top; when this iteration is completed, reverse the entire stack.

## Monotonic Stack

### Lexicographic Order

We can get lexicographically least/largest subsequency of an array by **Corollary 3**.

[Remove K Digits][remove-k-digits]

```java
public String removeKdigits(String num, int k) {
    StringBuilder sb = new StringBuilder();
    for (char ch : num.toCharArray()) {
        // Monotonically increasing stack
        while (k > 0 && sb.length() > 0 && ch < sb.charAt(sb.length() - 1)) {
            sb.deleteCharAt(sb.length() - 1);
            k--;
        }

        // To avoid leading 0, don't push '0' if the stack is empty
        if (sb.length() > 0 || ch > '0') {
            sb.append(ch);
        }
    }

    // Uses up k digits
    while (k > 0 && sb.length() > 0) {
        sb.deleteCharAt(sb.length() - 1);
        k--;
    }
    return sb.length() == 0 ? "0" : sb.toString();
}
```

[Find the Most Competitive Subsequence][find-the-most-competitive-subsequence]

```java
public int[] mostCompetitive(int[] nums, int k) {
    int[] st = new int[k];
    // j is the stack size
    for (int i = 0, j = 0, n = nums.length; i < n; i++) {
        // (n - i) remaining numbers
        while (j > 0 && nums[i] < st[j - 1] && n - i + j > k) {
            j--;
        }
        if (j < k) {
            st[j++] = nums[i];
        }
    }
    return st;
}
```

### PLE/PGE/NLE/NGE

* Previous Less Element (PLE)
* Previous Greater Element (PGE)
* Next Less Element (NLE)
* Next Greater Element (NGE)

The requested element can be strictly or non-strictly less/greater.

The **template** below uses monotonically increasing stack to get PLE/NLE.

```java
Deque<Integer> st = new ArrayDeque<>();
int[] prev = new int[n], next = new int[n];
Arrays.fill(next, n);

for (int i = 0; i < n; i++) {
    // Monotonically increasing stack
    // Strictly less next
    while (!st.isEmpty() && nums[i] < nums[st.peek()]) {
        next[st.pop()] = i;
    }
    // Non-strictly less prev
    prev[i] = st.isEmpty() ? -1 : st.peek();
    st.push(i);
}
```

|                | Previous < | Previous <= |  Next <  |  Next <=  |
|----------------|------------|-------------|----------|-----------|
|Monotonic Stack | Increasing | Increasing  |Increasing|Increasing |
|Stack Strictness|   Strict   | Non-strict  |Non-strict|  Strict   |
|   Condition    |a[i] <= top | a[i] < top  |a[i] < top|a[i] <= top|

![NGE & PGE](/assets/img/algorithm/monotonic_stack.png){: width="400" }

**NLE**

[Final Prices With a Special Discount in a Shop][final-prices-with-a-special-discount-in-a-shop]

```c++
vector<int> finalPrices(vector<int>& prices) {
    vector<int> answer = prices;
    // Next less element
    stack<int> st;
    for (int i = 0; i < prices.size(); i++) {
        while (!st.empty() && answer[i] <= answer[st.top()]) {
            answer[st.top()] -= answer[i];
            st.pop();
        }
        st.push(i);
    }
    return answer;
}
```

Similarly, with monotonically decreasing stack, we can get PGE/NGE.

|                | Previous > | Previous >= |  Next >  |  Next >=  |
|----------------|------------|-------------|----------|-----------|
|Monotonic Stack | Decreasing | Decreasing  |Decreasing|Decreasing |
|Stack Strictness|   Strict   | Non-strict  |Non-strict|  Strict   |
|   Condition    |a[i] >= top | a[i] > top  |a[i] > top|a[i] >= top|

**NGE**

Imagine each element is a node in a graph, and we connect node `u` to its NLE node `v` with an edge. The resulting graph is an [increasing spanning forest](https://users.math.msu.edu/users/bsagan/Papers/Old/isf-pub.pdf).

![Increasing Spanning Forest](/assets/img/algorithm/monotonic_stack_nle.png)

[Online Stock Span][online-stock-span]

```java
private Deque<int[]> st;

public StockSpanner() {
    this.st = new ArrayDeque<>();
}

public int next(int price) {
    int count = 1;
    // Next greater element
    while (!st.isEmpty() && price >= st.peek()[0]) {
        count += st.pop()[1];
    }
    st.push(new int[]{price, count});
    return count;
}
```

**PGE**

[Steps to Make Array Non-decreasing][steps-to-make-array-non-decreasing]

```c++
int totalSteps(vector<int>& nums) {
    // {num, number of steps performed until num}
    stack<pair<int,int>> st;
    int mx = 0;
    for (int num : nums) {
        // Next greater element (>)
        int steps = 0;
        while (!st.empty() && num >= st.top().first) {
            // The max steps among all the popped elements
            steps = max(steps, st.top().second);
            st.pop();
        }

        // If no element is left in the stack, 
        st.push({num, steps = st.empty() ? 0 : steps + 1});
        mx = max(mx, steps);
    }
    return mx;
}
```

[Next Greater Element IV][next-greater-element-iv]

```java
public int[] secondGreaterElement(int[] nums) {
    int n = nums.length;
    int[] next = new int[n];
    Arrays.fill(next, -1);

    // stack1: elements that haven't found their first NGE
    // stack2: elements that have found their first NGE but not second NGE
    Deque<Integer> st1 = new ArrayDeque<>(), st2 = new ArrayDeque<>(), tmp = new ArrayDeque<>();
    for (int i = 0; i < n; i++) {
        // finds second NGE
        while (!st2.isEmpty() && nums[i] > nums[st2.peek()]) {
            next[st2.pop()] = nums[i];
        }

        // moves all elements whose first NGE is nums[i] to stack2
        while (!st1.isEmpty() && nums[i] > nums[st1.peek()]) {
            tmp.push(st1.pop());
        }

        while (!tmp.isEmpty()) {
            st2.push(tmp.pop());
        }

        st1.push(i);
    }
    return next;
}
```

[Minimum Cost Tree From Leaf Values][minimum-cost-tree-from-leaf-values]

```java
public int mctFromLeafValues(int[] arr) {
    Deque<Integer> st = new ArrayDeque<>();
    st.push(Integer.MAX_VALUE);

    int sum = 0;
    for (int a : arr) {
        // previous and next greater element
        while (st.peek() <= a) {
            int top = st.pop();
            sum += top * Math.min(st.peek(), a);
        }
        st.push(a);
    }

    while (st.size() > 2) {
        sum += st.pop() * st.peek();
    }

    return sum;
}
```

[Number of Visible People in a Queue][number-of-visible-people-in-a-queue]

```java
public int[] canSeePersonsCount(int[] heights) {
    int n = heights.length;
    int[] answer = new int[n];
    Deque<Integer> st = new ArrayDeque<>();
    for (int i = 0; i < n; i++) {
        // monotonically decreasing stack (strict)
        // next greater (>=) element
        while (!st.isEmpty() && heights[i] >= heights[st.peek()]) {
            // st.peek() can see i
            // st.peek() can't see after i, so st.peek() is done and popped
            answer[st.pop()]++;
        }

        if (!st.isEmpty()) {
            // previous greater (>) element
            // st.peek() can see i
            answer[st.peek()]++;
        }

        st.push(i);
    }
    return answer;
}
```

Similar: [Number of People That Can Be Seen in a Grid][number-of-people-that-can-be-seen-in-a-grid]

In this problem, elements are not necessarily distinct. Therefore, we need to make a slight change to the code:

```java
public int[] canSeePersonsCount(int[] heights) {
    int n = heights.length;
    int[] answer = new int[n];
    Deque<Integer> st = new ArrayDeque<>();
    for (int i = 0; i < n; i++) {
        // monotonically decreasing stack (strict)
        // next greater (>=) element
        boolean isEqual = false;
        while (!st.isEmpty() && heights[i] >= heights[st.peek()]) {
            if (heights[i] == heights[st.peek()]) {
                isEqual = true;
            }
            // st.peek() can see i
            // st.peek() can't see after i, so st.peek() is done and popped
            answer[st.pop()]++;
        }

        // e.g. [4,2,1,1,3]
        // we skip incrementing the answer at index 0 when i == 3
        // because it's already incremented when i == 2
        // iwo, heights[0] can't see heights[3] because of heights[2]
        if (!st.isEmpty() && isEqual) {
            // previous greater (>) element
            // st.peek() can see i
            answer[st.peek()]++;
        }

        st.push(i);
    }
    return answer;
}
```

[Maximum Width Ramp][maximum-width-ramp]

```java
public int maxWidthRamp(int[] nums) {
    // decreasing stack
    Deque<Integer> st = new ArrayDeque<>();
    int max = 0, n = nums.length;
    for (int i = 0; i < n; i++) {
        if (st.isEmpty() || nums[i] < nums[st.peek()]) {
            st.push(i);
        }
    }

    for (int i = n - 1; i > max; i--) {
        while (!st.isEmpty() && nums[i] >= nums[st.peek()]) {
            // i is the largest index with nums[i] >= stack top
            // so it's safe to pop
            max = Math.max(max, i - st.pop());
        }
    }
    return max;
}
```

The following problem is not PGE or NGE, and the solution doesn't use stacks. But we can see some similarities.

[Sum of Imbalance Numbers of All Subarrays][sum-of-imbalance-numbers-of-all-subarrays]

```java
public int sumImbalanceNumbers(int[] nums) {
    int sum = 0, n = nums.length;
    // lastIndices[i]: last index of the number `i`
    // leftBounds[i]: left bound of the number `i`.
    //   subarrays starts exclusively from leftBounds[i] up to i have non-zero imbalance numbers
    int[] lastIndices = new int[n + 2], leftBounds = new int[n];

    // for each num, considers num and num + 1 only - no need to consider num - 1
    // otherwise the final result will include duplication
    Arrays.fill(lastIndices, -1);
    for (int i = 0; i < n; i++) {
        leftBounds[i] = Math.max(lastIndices[nums[i] + 1], lastIndices[nums[i]]);
        lastIndices[nums[i]] = i;
    }

    Arrays.fill(lastIndices, n);
    for (int i = n - 1; i >= 0; i--) {
        lastIndices[nums[i]] = i;
        // again, considers num + 1 only - no need to consider num, in order to deduplicate
        sum += (i - leftBounds[i]) * (lastIndices[nums[i] + 1] - i);
    }

    // subtracts the result for max(nums[i])
    return sum - n * (n + 1) / 2;
}
```

#### Subarray Min/Max

[Sum of Subarray Minimums][sum-of-subarray-minimums]

The basic idea is to find both the previous and next less elements of each array element with two stacks. This can be simplified to one stack:

```java
private int MOD = (int)1e9 + 7;

public int sumSubarrayMins(int[] arr) {
    Deque<Integer> st = new ArrayDeque<>();
    long result = 0;
    int n = arr.length;
    // virtually appends 0 to the end of the array
    for (int i = 0; i <= n; i++) {
        while (!st.isEmpty() && (i == n || arr[i] < arr[st.peek()])) {
            // i is the next less (<) element of j
            int j = st.pop();
            // k is the previous less (<=) element of j
            int k = st.isEmpty() ? -1 : st.peek();
            result = (result + (long)arr[j] * (i - j) * (j - k)) % MOD;
        }
        st.push(i);
    }
    return (int)result;
}
```

[Subarray With Elements Greater Than Varying Threshold][subarray-with-elements-greater-than-varying-threshold]

```java
if ((i - k - 1) * nums[j] > threshold) {
    return i - k - 1;
}
```

With this perspective, the solution of the following *Histogram* problem is much easier to understand:

[Largest Rectangle in Histogram][largest-rectangle-in-histogram]

```java
public int largestRectangleArea(int[] heights) {
    // index of non-descreasing heights
    Deque<Integer> st = new ArrayDeque<>();
    st.push(-1);  // makes width concise

    int area = 0, n = heights.length;
    for (int i = 0; i <= heights.length; i++) {
        while (st.peek() >= 0 && (i == n || heights[i] < heights[st.peek()])) {
            // j = st.pop()
            // heights[i] is the next less (<) element of heights[j]
            int h = heights[st.pop()];
            // k = st.peek()
            // heights[k] is the previous less (<=) element of heights[j]
            // w = length of (k, i)
            int w = i - 1 - st.peek();
            area = Math.max(area, h * w);
        }
        st.push(i);
    }
    return area;
}
```

```
[2,1,5,6,2,3]
i = 0	[0]				area = 0
i = 1	[]		h * w = 2	area = 2
i = 1	[1]				area = 2
i = 2	[2,1]				area = 2
i = 3	[3,2,1]				area = 2
i = 4	[2,1]		h * w = 6	area = 6
i = 4	[1]		h * w = 10	area = 10
i = 4	[4,1]				area = 10
i = 5	[5,4,1]				area = 10
i = 6	[4,1]		h * w = 3	area = 10
i = 6	[1]		h * w = 8	area = 10
i = 6	[]		h * w = 6	area = 10
i = 6	[6]				area = 10
```

[Maximum of Minimum Values in All Subarrays][maximum-of-minimum-values-in-all-subarrays]

```java
public int[] findMaximums(int[] nums) {
    int n = nums.length;
    int[] ans = new int[n];

    Deque<Integer> st = new ArrayDeque<>();
    // virtually appends 0 to the end of the array
    for (int i = 0; i <= n; i++) {
        while (!st.isEmpty() && (i == n || nums[i] < nums[st.peek()])) {
            // nums[i] is the next less (<) element of nums[j]
            int j = st.pop();
            // k = st.peek()
            // nums[k] is the previous less (<=) element of nums[j]
            // the range is (k, i)
            int len = i - (st.isEmpty() ? -1 : st.peek()) - 1;
            ans[len - 1] = Math.max(ans[len - 1], nums[j]);
        }
        st.push(i);
    }

    // ans is non-increasing
    for (int i = n - 1; i > 0; i--) {
        ans[i - 1] = Math.max(ans[i - 1], ans[i]);
    }
    return ans;
}
```

## Monoqueue

### Sliding Window Min/Max

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

The above solution is a good example of the application of _Property #2_:

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

Similar problem: [Constrained Subsequence Sum][constrained-subsequence-sum].

### Shortest Subarray With Sum >= k

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

## Monotonic Map

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

## + Binary Search

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

[constrained-subsequence-sum]: https://leetcode.com/problems/constrained-subsequence-sum/
[final-prices-with-a-special-discount-in-a-shop]: https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/
[find-building-where-alice-and-bob-can-meet]: https://leetcode.com/problems/find-building-where-alice-and-bob-can-meet/
[find-maximum-non-decreasing-array-length]: https://leetcode.com/problems/find-maximum-non-decreasing-array-length/
[find-the-most-competitive-subsequence]: https://leetcode.com/problems/find-the-most-competitive-subsequence/
[jump-game-vi]: https://leetcode.com/problems/jump-game-vi/
[largest-rectangle-in-histogram]: https://leetcode.com/problems/largest-rectangle-in-histogram/
[longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit]: https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/
[max-value-of-equation]: https://leetcode.com/problems/max-value-of-equation/
[maximum-balanced-subsequence-sum]: https://leetcode.com/problems/maximum-balanced-subsequence-sum/
[maximum-number-of-robots-within-budget]: https://leetcode.com/problems/maximum-number-of-robots-within-budget/
[maximum-of-minimum-values-in-all-subarrays]: https://leetcode.com/problems/maximum-of-minimum-values-in-all-subarrays/
[maximum-sum-queries]: https://leetcode.com/problems/maximum-sum-queries/
[maximum-width-ramp]: https://leetcode.com/problems/maximum-width-ramp/
[minimum-cost-tree-from-leaf-values]: https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/
[next-greater-element-iv]: https://leetcode.com/problems/next-greater-element-iv/
[number-of-people-that-can-be-seen-in-a-grid]: https://leetcode.com/problems/number-of-people-that-can-be-seen-in-a-grid/
[number-of-visible-people-in-a-queue]: https://leetcode.com/problems/number-of-visible-people-in-a-queue/
[online-stock-span]: https://leetcode.com/problems/online-stock-span/
[remove-k-digits]: https://leetcode.com/problems/remove-k-digits/
[shortest-subarray-with-sum-at-least-k]: https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/
[sliding-window-maximum]: https://leetcode.com/problems/sliding-window-maximum/
[steps-to-make-array-non-decreasing]: https://leetcode.com/problems/steps-to-make-array-non-decreasing/
[subarray-with-elements-greater-than-varying-threshold]: https://leetcode.com/problems/subarray-with-elements-greater-than-varying-threshold/
[sum-of-imbalance-numbers-of-all-subarrays]: https://leetcode.com/problems/sum-of-imbalance-numbers-of-all-subarrays/
[sum-of-subarray-minimums]: https://leetcode.com/problems/sum-of-subarray-minimums/

[^1]: For the sake of simplicity, we only discuss monotonically strictly increasing stack in the theorem, and we suppose all the elements are unique. It's trivial to extend it to non-strict or descreasing stacks.

