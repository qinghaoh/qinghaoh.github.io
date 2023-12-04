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

Monoqueue can be used to find the min/max of a _bounded_ range.

# Upper Bound

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

# Lower Bound

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
        // Checks all possible candidates when p[i] is used as the minuend
        // Pops the head of the deque until p[i] - p[head] < k
        // The head is a possible candidate
        //
        // It's impossible to miss a candidate subtrahend for p[i]
        // due to the subtrahend not being the current deque head,
        // because the minimum element so far is always the deque head
        while (!dq.isEmpty() && p[i] - p[dq.peekFirst()] >= k) {
            min = Math.min(min, i - dq.pollFirst());
        }

        // Now we prove there's no shorter subarray:
        // For the element at index j, head < j < i, it's either in the deque or was popped.
        // If in the deque:
        //   p[j] > p[head] => p[i] - p[j] < p[i] - p[head] < k
        // Otherwise, it was greater than some in-deque element p[j']:
        //   p[j] > p[j'], p[j'] > p[head] => p[i] - p[j] < p[i] - p[j'] < p[i] - p[head] < k
        while (!dq.isEmpty() && p[i] <= p[dq.peekLast()]) {
            dq.pollLast();
        }
        dq.offerLast(i);
    }

    return min <= n ? min : -1;
}
```

From another perspective, the above solution finds the min (deque head) of a bounded range.

This technique can be used to compute the recurrence relation more quickly in some dynamic programming problems.

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

[jump-game-vi]: https://leetcode.com/problems/jump-game-vi/
[shortest-subarray-with-sum-at-least-k]: https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/
[sliding-window-maximum]: https://leetcode.com/problems/sliding-window-maximum/
