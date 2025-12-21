---
title:  "Classic Problems"
category: algorithm
---
## Non-Overlapping Segments

[Number of Sets of K Non-Overlapping Line Segments][number-of-sets-of-k-non-overlapping-line-segments]

## Reverse Pairs

Examples:

* [Count of Smaller Numbers After Self][count-of-smaller-numbers-after-self]
* [Reverse Pairs][reverse-pairs]
* [Number of Pairs Satisfying Inequality][number-of-pairs-satisfying-inequality]

Solutions:

* Merge Sort
* Fenwick Tree
* Segment Tree
* Binary Search Tree

## Swim in Rising Water

* [Swim in Rising Water][swim-in-rising-water]

Solutions:

* Dijkstra's Algorithm
* BinarySearch + BFS/DFS
* Union-Find

```java
```

## Subarray Sum Equals K

[Count Number of Nice Subarrays][count-number-of-nice-subarrays]

* Sliding window
* If we apply `nums[i] -> nums[i] % 2`, the problem becomes [Subarray Sum Equals K][subarray-sum-equals-k], and we can use hash map to store the count of each sum during iteration.

[count-of-smaller-numbers-after-self]: https://leetcode.com/problems/count-of-smaller-numbers-after-self/
[count-number-of-nice-subarrays]: https://leetcode.com/problems/count-number-of-nice-subarrays/
[number-of-pairs-satisfying-inequality]: https://leetcode.com/problems/number-of-pairs-satisfying-inequality/
[number-of-sets-of-k-non-overlapping-line-segments]: https://leetcode.com/problems/number-of-sets-of-k-non-overlapping-line-segments/
[reverse-pairs]: https://leetcode.com/problems/reverse-pairs/
[subarray-sum-equals-k]: https://leetcode.com/problems/subarray-sum-equals-k/
[swim-in-rising-water]: https://leetcode.com/problems/swim-in-rising-water/
