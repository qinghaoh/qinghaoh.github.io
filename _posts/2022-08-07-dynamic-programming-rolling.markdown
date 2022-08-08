---
layout: post
title:  "Dynamic Programming (Rolling)"
usemathjax: true
---
[Check if There is a Valid Partition For The Array][check-if-there-is-a-valid-partition-for-the-array]

{% highlight java %}
public boolean validPartition(int[] nums) {
    boolean[] dp = {true, false, nums[0] == nums[1], false};
    int n = nums.length;
    for (int i = 2; i < n; i++) {
        boolean two = nums[i] == nums[i - 1];
        boolean three = (two && nums[i] == nums[i - 2]) || (nums[i] - 1 == nums[i - 1] && nums[i] - 2 == nums[i - 2]);
        // we just need to record the values in 4 cases: i - 2, i - 1, i, i + 1
        // so a DP array of size 4 is enough
        dp[(i + 1) % 4] = (two && dp[(i - 1) % 4]) || (three && dp[(i - 2) % 4]);
    }
    return dp[n % 4];
}
{% endhighlight %}

[check-if-there-is-a-valid-partition-for-the-array]: https://leetcode.com/problems/check-if-there-is-a-valid-partition-for-the-array/
