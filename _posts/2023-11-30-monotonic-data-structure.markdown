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

Here are some key observations of monotonic data structures:

1. The current element is always at the top.
1. The minimum element so far is always at the bottom.
1. The stack at each index can be equivalently derived in this way: iterate _reversely_ from the current index, and push the element into the stack if it's less than the stack top; when this iteration is completed, reverse the entire stack.

# Monoqueue

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

[shortest-subarray-with-sum-at-least-k]: https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/
