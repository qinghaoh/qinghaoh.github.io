---
title:  "Repeated Element"
category: algorithm
tags: array
---
[N-Repeated Element in Size 2N Array][n-repeated-element-in-size-2n-array]

`k` repeated element in size `n` array:

Find the minimum size `m`, so that there exists at least one subarray with size `m`, and it is guaranteed to contain more than one repeated element.

In this example, the size is `2N / N + 1 = 3`.

```java
public int repeatedNTimes(int[] A) {
    for (int i = 0; i < A.length; i++) {
        for (int j = 1; j <= 3 && i + j < A.length; j++) {
            if (A[i] == A[i + j]) {
                return A[i];
            }
        }
    }
    return 0;
}
```

Or equivalently,

```java
public int repeatedNTimes(int[] A) {
    for (int i = 2; i < A.length; i++) {
        if (A[i] == A[i - 1] || A[i] == A[i - 2]) {
            return A[i];
        }  
    }
    return A[0];
}
```

We can of course expand this subarray window to, for example, 4.

[Guess the Majority in a Hidden Array][guess-the-majority-in-a-hidden-array]

```java
public int guessMajority(ArrayReader reader) {
    int n = reader.length();
    int groupEqualsNum3 = 1;  // initially nums[3] is in this group
    int groupNotEqualsNum3 = 0;
    int indexA = -1, indexB = -1;
    int r0123 = reader.query(0, 1, 2, 3);
    for (int i = 4; i < n; i++) {
        // divides all numbers after nums[3] into two groups
        if (reader.query(0, 1, 2, i) == r0123) {  // nums[3] == nums[i]
            groupEqualsNum3++;
            indexA = i;
        } else {  // nums[3] != nums[i]
            groupNotEqualsNum3++;
            indexB = i;
        }
    }

    // finds out which group nums[0], nums[1], nums[2] belongs to
    int r0124 = reader.query(0, 1, 2, 4);
    {% raw %}
    int[][] queries = {{1, 2, 3, 4}, {0, 2, 3, 4}, {0, 1, 3, 4}};
    {% endraw %}
    for (int i = 0; i < 3; i++) {
        // e.g. r1234 vs r0124. Both contain nums[1], nums[2] and nums[4].
        // if r1234 == r0124, then nums[0] == nums[3]
        // otherwise nums[0] != nums[3]
        if (reader.query(queries[i][0], queries[i][1], queries[i][2], queries[i][3]) == r0124) {
            groupEqualsNum3++;
            indexA = i;
        } else {
            groupNotEqualsNum3++;
            indexB = i;
        }
    }

    return groupEqualsNum3 == groupNotEqualsNum3 ? -1 : (groupEqualsNum3 > groupNotEqualsNum3 ? indexA : indexB);
}
```

## Sliding window

[Detect Pattern of Length M Repeated K or More Times][detect-pattern-of-length-m-repeated-k-or-more-times]

```java
public boolean containsPattern(int[] arr, int m, int k) {
    for (int i = 0, count = 0; i + m < arr.length; i++) {
        if (arr[i] != arr[i + m]) {
            count = 0;
        } else if (++count == (k - 1) * m) {
            return true;
        }
    }
    return false;
}
```

[detect-pattern-of-length-m-repeated-k-or-more-times]: https://leetcode.com/problems/detect-pattern-of-length-m-repeated-k-or-more-times/
[guess-the-majority-in-a-hidden-array]: https://leetcode.com/problems/guess-the-majority-in-a-hidden-array/
[n-repeated-element-in-size-2n-array]: https://leetcode.com/problems/n-repeated-element-in-size-2n-array/
