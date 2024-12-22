---
title:  "Divide and Conquer"
category: algorithm
tags: divide-and-conquer
---
[Merge K Sorted Lists][merge-k-sorted-lists]

```java
public ListNode mergeKLists(ListNode[] lists) {
    int k = lists.length;
    if (k == 0) {
        return null;
    }

    int interval = 1;
    while (interval < k) {
        for (int i = 0; i + interval < k; i += interval * 2) {
            lists[i] = mergeTwoLists(lists[i], lists[i + interval]);            
        }
        interval *= 2;
    }

    return lists[0];
}

private ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode head = new ListNode(0), curr = head;
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            curr.next = l1;
            l1 = l1.next;
        } else {
            curr.next = l2;
            l2 = l2.next;
        }
        curr = curr.next;
    }
    curr.next = l1 == null ? l2 : l1;
    return head.next;
}
```

Another solution is to add all nodes to a min heap.

[Super Ugly Number][super-ugly-number]

```java
public int nthSuperUglyNumber(int n, int[] primes) {
    long[] ugly = new long[n];
    ugly[0] = 1;

    int m = primes.length;
    int[] indices = new int[m];

    for (int i = 1; i < n; i++) {
        ugly[i] = Integer.MAX_VALUE;
        for (int j = 0; j < m; j++) {
            // moves the pointer of the last "base" ugly number forward
            if (ugly[indices[j]] * primes[j] == ugly[i - 1]) {
                indices[j]++;
            }

            // finds and assigns the min value to the current ugly number
            ugly[i] = Math.min(ugly[i], ugly[indices[j]] * primes[j]);
        }
    }

    return (int)ugly[n - 1];
}
```

To find the minimum of `ugly[indices[j]] * primes[j]`, we can also use Priority Queue for less time complexity.

[Longest Nice Substring][longest-nice-substring]

```java
public String longestNiceSubstring(String s) {
    if (s.length() < 2) {
        return "";
    }

    Set<Character> set = new HashSet<>();
    for (char c: s.toCharArray()) {
        set.add(c);
    }
    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        if (set.contains(Character.toUpperCase(c)) && set.contains(Character.toLowerCase(c))) {
            continue;
        }
        String s1 = longestNiceSubstring(s.substring(0, i));
        String s2 = longestNiceSubstring(s.substring(i + 1));
        return s1.length() >= s2.length() ? s1 : s2;
    }
    return s; 
}
```

[Beautiful Array][beautiful-array]

```java
public int[] beautifulArray(int n) {
    List<Integer> list = new ArrayList<>();
    list.add(1);

    // properties:
    // if A is a beautiful array, then
    //  - A + c is a beautiful array
    //  - c * A is a beautiful array
    //  - deletion: A is still a beautiful array after deleting some elements in it
    while (list.size() < n) {
        List<Integer> tmp = new ArrayList<>();
        // divide and conquer
        // divides the numbers into two parts: odd + even, because odd + even = odd != 2 * x
        for (int i : list) {
            if (i * 2 - 1 <= n) {
                tmp.add(i * 2 - 1);
            }
        }
        for (int i : list) {
            if (i * 2 <= n) {
                tmp.add(i * 2);
            }
        }
        list = tmp;
    }
    return list.stream().mapToInt(i -> i).toArray();
}
```

```
n = 10

[1]
[1,2]
[1,3,2,4]
[1,5,3,7,2,6,4,8]
[1,9,5,3,7,2,10,6,4,8]
```

It's easy to come up with a recursive version.

[Bit reversal permutation](https://en.wikipedia.org/wiki/Bit-reversal_permutation)

Another solution is bit reverse (br) sort. It can be proven that if `i + k = j * 2`, then `br(j)` is either less than or greater than both `br(i)` and `br(k)`. Thus the algorithm is to sort the numbers based on their bit reverse. Credit to @Aging.

[beautiful-array]: https://leetcode.com/problems/beautiful-array/
[longest-nice-substring]: https://leetcode.com/problems/longest-nice-substring/
[merge-k-sorted-lists]: https://leetcode.com/problems/merge-k-sorted-lists/
[super-ugly-number]: https://leetcode.com/problems/super-ugly-number/
