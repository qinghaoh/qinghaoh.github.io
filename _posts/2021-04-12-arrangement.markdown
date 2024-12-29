---
title:  "Arrangement"
category: algorithm
tags: math
---
# Arrangement

[Beautiful Arrangement II][beautiful-arrangement-ii]

```java
public int[] constructArray(int n, int k) {
    int[] list = new int[n];
    // max(k) == n - 1
    for (int i = 0, left = 1, right = n; left <= right; i++) {
        list[i] = k > 1 ? (k-- % 2 == 0 ? right-- : left++) : left++;
    }
    return list;
}
```

```
n = 9, k = 8

left:    1     2     3     4     5
right:      9     8     7     6
diff:     8   7 6   5 4   3  2  1
```
```
n = 9, k = 5

left:    1     2     3  4  5  6  7
right:      9     8
diff:     8   7  6  5 1  1  1  1
```

# Dearrangement

$$
!n=(n-1)({!(n-1)}+{!(n-2)})
$$

$$
!n=n!\sum _{i=0}^{n}{\frac {(-1)^{i}}{i!}}, \quad n\geq 0
$$

[beautiful-arrangement-ii]: https://leetcode.com/problems/beautiful-arrangement-ii/
