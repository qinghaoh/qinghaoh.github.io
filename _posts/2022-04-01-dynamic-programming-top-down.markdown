---
layout: post
title:  "Dynamic Programming (Top-down)"
tag: dynamic programming
---

[Maximum Value of K Coins From Piles][maximum-value-of-k-coins-from-piles]

{% highlight java %}
private Integer[][] memo;

public int maxValueOfCoins(List<List<Integer>> piles, int k) {
    this.memo = new Integer[piles.size() + 1][k + 1];
    return dfs(piles, 0, k);
}

private int dfs(List<List<Integer>> piles, int i, int k) {
    if (k == 0 || i == piles.size()) {
        return 0;
    }

    if (memo[i][k] != null) {
        return memo[i][k];
    }

    int total = 0, max = dfs(piles, i + 1, k);
    for (int j = 0; j < Math.min(piles.get(i).size(), k); j++) {
        total += piles.get(i).get(j);
        max = Math.max(max, total + dfs(piles, i + 1, k - j - 1));
    }
    return memo[i][k] = max;
}
{% endhighlight %}

[maximum-value-of-k-coins-from-piles]: https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/
