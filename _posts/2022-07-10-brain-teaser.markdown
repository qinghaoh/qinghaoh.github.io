---
title:  "Brain Teaser"
category: algorithm
---
[Minimum Amount of Time to Fill Cups][minimum-amount-of-time-to-fill-cups]

```java
public int fillCups(int[] amount) {
    int max = 0, sum = 0;
    for (int a : amount) {
        max = Math.max(max, a);
        sum += a;
    }
    // 1 cup of any type of water: max
    // 2 cups with different types of water: ceil(sum / 2)
    //
    // one of the two cups is from the max stack
    // the other one is from either the min or mid stack
    // we can distribute the water from the min stack optimally to the other two stacks
    // e.g. [1, 3, 5] -> [0, 3 + 1, 5]: max(A) = 5
    // e.g. [3, 4, 4] -> [0, 4 + 1, 4 + 2]: ceil(sum / 2) = 6
    return Math.max(max, (sum + 1) / 2);
}
```

[minimum-amount-of-time-to-fill-cups]: https://leetcode.com/problems/minimum-amount-of-time-to-fill-cups/
