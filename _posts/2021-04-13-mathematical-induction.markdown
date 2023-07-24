---
title:  "Mathematical Induction"
category: algorithm
tags: math
---
# Arrangement

[Global and Local Inversions][global-and-local-inversions]

If the 0 occurs at index 2 or greater, then `A[0] > A[2] = 0` is a non-local inversion. So 0 can only occur at index 0 or 1.

If `A[1] = 0`, then we must have `A[0] = 1` otherwise `A[0] > A[j] = 1` is a non-local inversion.

Otherwise, `A[0] = 0`.

In summary, the possibilities are:
* A = [0] + (ideal permutation of 1...n-1)
* A = [1, 0] + (ideal permutation of 2...n-1).

A necessary and sufficient condition is: `Math.abs(A[i] - i) <= 1`. So we check this for every i.

```java
public boolean isIdealPermutation(int[] A) {
    for (int i = 0; i < A.length; i++) {
        if (Math.abs(A[i] - i) > 1) {
            return false;
        }
    }
    return true;
}
```

[Maximum Height by Stacking Cuboids][maximum-height-by-stacking-cuboids]

Credit to @quantuminfo

If the one with longest edge not as height is on the top of the cuboid stack, we can simply rotate it so the it contriubutes more height.

If the one with longest edge is in the middle, let's say it is `A` and the 3 edges are `[A1, A3, A2] (A3 > A2 && A3 > A1, A2 is not longest but it is the height)`, the one on top of `A` is `B [B1, B2, B3] (B3 >= B2 >= B1)`:

we have `A1 >= B1 && A3 >= B2 && A2 >= B3`

then: `A3 > A2 >= B3, A2 >= B3 >= B2, A1 >= B1`

so we can rotate `A` from `[A1, A3, A2]` to `[A1, A2, A3]` without afffecting `B` but make the total height larger (increase by `A3 - A2`)

[global-and-local-inversions]: https://leetcode.com/problems/global-and-local-inversions/
[maximum-height-by-stacking-cuboids]: https://leetcode.com/problems/maximum-height-by-stacking-cuboids/
