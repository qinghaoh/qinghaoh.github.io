---
title:  "Peak Valley"
category: algorithm
tag: array
---
[Best Time to Buy and Sell Stock II][best-time-to-buy-and-sell-stock-ii]

```c++
int maxProfit(vector<int>& prices) {
    int profit = 0;
    for (int i = 1; i < prices.size(); i++) {
        // Buys a stock at a valley and sells it at the next peak
        profit += max(0, prices[i] - prices[i - 1]);
    }
    return profit;
}
```

Almost the same:

[Maximum Alternating Subsequence Sum][maximum-alternating-subsequence-sum]

```java
public long maxAlternatingSum(int[] nums) {
    // gets the first one for free
    long max = nums[0];
    for (int i = 1; i < nums.length; i++) {
        max += Math.max(0, nums[i] - nums[i - 1]);
    }
    return max;
}
```

[Decrease Elements To Make Array Zigzag][decrease-elements-to-make-array-zigzag]

```java
private int MAX = 1001;

public int movesToMakeZigzag(int[] nums) {       
    int[] result = new int[2];
    int left = 0, right = 0;
    for (int i = 0; i < nums.length; i++) {
        left = i > 0 ? nums[i - 1] : MAX;
        right = i < nums.length - 1 ? nums[i + 1] : MAX;

        // decreases nums[odd] or nums[even]
        result[i % 2] += Math.max(0, nums[i] - Math.min(left, right) + 1);
    }
    return Math.min(result[0], result[1]);
}
```

[Find Permutation][find-permutation]

```java
public int[] findPermutation(String s) {
    int n = s.length();
    int[] perm = new int[n + 1];
    perm[n] = n + 1;
    int i = 0;
    while (i < n) {
        // finds next 'I'
        int j = i;
        while (j < n && s.charAt(j) == 'D') {
            j++;
        }

        // s.charAt(i)
        // - 'I': j == i
        // - 'D': two pointers, reverses [i + 1, j + 1]
        for (int k = j - i; k >= 0; k--, j--) {
            perm[i++] = j + 1;
        }
    }
    return perm;
}
```

In this solution, the final result is split into groups, with `I` as delimiter. In each group, the numbers are contiguous, and the groups are in ascending order.

e.g. 
```
 I    D I    D D I    D  D D I    D
[1, ' 3,2, ' 6,5,4, ' 10,9,8,7, ' 12,11]
```

[Candy][candy]

```java
public int candy(int[] ratings) {
    // steps of continuous up and down respectively
    int up = 0, down = 0, peak = 0, count = 1;
    for (int i = 1; i < ratings.length; i++) {
        // each child gets at least one candy
        count++;

        if (ratings[i - 1] < ratings[i]) {
            peak = ++up;
            down = 0;
            count += up;
        } else if (ratings[i - 1] > ratings[i])  {
            up = 0;
            down++;
            // gives peak one more candy if down > peak
            count += down + (peak >= down ? -1 : 0);
        } else {
            peak = up = down = 0;
        }
    }
    return count;
}
```

For example, `[0, 1, 10, 9, 8, 7]`

```
i = 1, up = 1, down = 0, peak = 1, count = 3
i = 2, up = 2, down = 0, peak = 2, count = 6
i = 3, up = 0, down = 1, peak = 2, count = 7
i = 4, up = 0, down = 2, peak = 2, count = 9
i = 5, up = 0, down = 3, peak = 2, count = 13

```

# Two Pointers

[Shortest Subarray to be Removed to Make Array Sorted][shortest-subarray-to-be-removed-to-make-array-sorted]

```java
public int findLengthOfShortestSubarray(int[] arr) {
    int n = arr.length;
    // first peak and last valley
    int left = 0, right = n - 1;
    while (left + 1 < n && arr[left] <= arr[left + 1]) {
        left++;
    }
    if (left == n - 1) {
        return 0;
    }

    while (right > left && arr[right - 1] <= arr[right]) {
        right--;
    }

    int min = Math.min(n - left - 1, right);
    int i = 0, j = right;
    while (i <= left && j < n) {
        if (arr[j] >= arr[i]) {
            min = Math.min(min, j - i - 1);
            i++;
        } else {
            j++;
        }
    }
    return min;
}
```

[best-time-to-buy-and-sell-stock-ii]: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
[candy]: https://leetcode.com/problems/candy/
[decrease-elements-to-make-array-zigzag]: https://leetcode.com/problems/decrease-elements-to-make-array-zigzag/
[find-permutation]: https://leetcode.com/problems/find-permutation/
[maximum-alternating-subsequence-sum]: https://leetcode.com/problems/maximum-alternating-subsequence-sum/
[shortest-subarray-to-be-removed-to-make-array-sorted]: https://leetcode.com/problems/shortest-subarray-to-be-removed-to-make-array-sorted/
