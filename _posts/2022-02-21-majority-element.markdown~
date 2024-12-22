---
title:  "Majority Element"
category: algorithm
---

Majority element: an element that occurs repeatedly for more than half of the elements of the input.

# Boyer-Moore Voting Algorithm

[Boyer–Moore majority vote algorithm](https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_majority_vote_algorithm): finds the majority of a sequence of elements using linear time and constant space.

[Majority Element][majority-element]

```java
public int majorityElement(int[] nums) {
    int count = 0;
    Integer candidate = null;

    for (int num : nums) {
        if (count == 0) {
            candidate = num;
        }
        count += (num == candidate) ? 1 : -1;
    }

    return candidate;
}
```

[majority-element]: https://leetcode.com/problems/majority-element/
