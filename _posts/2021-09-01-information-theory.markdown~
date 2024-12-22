---
title:  "Information Theory"
category: algorithm
tags: math
---
[Poor Pigs][poor-pigs]

```java
public int poorPigs(int buckets, int minutesToDie, int minutesToTest) {
    // the number of available states for a pig
    // e.g. minutesToTest / minutesToDie == 2
    // there are 3 available states for the pig:
    // - alive
    // - dead after the 1st test
    // - dead after the 2nd test
    //
    // Information Theory: each pig carries log(states) bit of information
    int states = minutesToTest / minutesToDie + 1;

    // if one pig has s states, p pigs can test s ^ p buckets
    // finds min p that s ^ p >= buckets
    return (int)Math.ceil(Math.log(buckets) / Math.log(states));
}
```

Another way of explanation (credit to @StefanPochmann):

With 2 pigs, poison killing in 15 minutes, and having 60 minutes, we can find the poison in up to 25 buckets in the following way. Arrange the buckets in a 5×5 square:

```
 1  2  3  4  5
 6  7  8  9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
```

Now use one pig to find the row (make it drink from buckets 1, 2, 3, 4, 5, wait 15 minutes, make it drink from buckets 6, 7, 8, 9, 10, wait 15 minutes, etc). Use the second pig to find the column (make it drink 1, 6, 11, 16, 21, then 2, 7, 12, 17, 22, etc).

Having 60 minutes and tests taking 15 minutes means we can run four tests. If the row pig dies in the third test, the poison is in the third row. If the column pig doesn't die at all, the poison is in the fifth column (this is why we can cover five rows/columns even though we can only run four tests).

[poor-pigs]: https://leetcode.com/problems/poor-pigs/
