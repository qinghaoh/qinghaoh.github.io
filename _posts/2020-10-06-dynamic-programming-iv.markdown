---
layout: post
title:  "Dynamic Programming IV"
tag: dynamic programming
---
[Maximal Square][maximal-square]

{% highlight java %}
public int maximalSquare(char[][] matrix) {
    if (matrix.length == 0) {
        return 0;
    }

    int m = matrix.length, n = matrix[0].length;
    // dp[i][j]: side length of the max square whose bottom-right is at (i - 1, j - 1)
    int[][] dp = new int[m + 1][n + 1];
    int maxLen = 0;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (matrix[i - 1][j - 1] == '1') {
                dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
                maxLen = Math.max(maxLen, dp[i][j]);
            }
        }
    }
    return maxLen * maxLen;
}
{% endhighlight %}

For example:

```
0 1 1 1 0
1 1 1 1 0
0 1 1 1 1
0 1 1 1 1
0 0 1 1 1
```

dp[][]:

```
0 1 1 1 0
1 1 2 2 0
0 1 2 3 1
0 1 2 3 2
0 0 1 2 3
```

[Maximal Rectangle][maximal-rectangle]

{% highlight java %}
public int maximalRectangle(char[][] matrix) {
    if (matrix.length == 0) {
        return 0;
    }

    int m = matrix.length, n = matrix[0].length;
    int area = 0;

    // Per row:
    // height[i]: number of continuous '1' from the current row to top in the i-th column
    int[] height = new int[n];
    // left[i]: index of left bound of the rectangle with height[i]
    //   if left[i] == 0, it means either no rectangle, or the rectangle starts from index 0
    int[] left = new int[n];
    // right[i]: index of right bound of the rectangle with height[i]
    //   if right[i] == n - 1, it means either no rectangle, or the rectangle ends at index (n - 1)
    int[] right = new int[n];

    Arrays.fill(right, n - 1);

    for (int i = 0; i < m; i++) {
        // right bound of the rectangle with height[j] in the current row
        int rightBound = n - 1;
        for (int j = n - 1; j >= 0; j--) {
            if (matrix[i][j] == '1') {
                // right[] is a rolling array
                right[j] = Math.min(right[j], rightBound);
            } else {
                right[j] = n - 1;
                rightBound = j - 1;
            }
        }

        // left bound of the rectangle with height[j] in the current row
        int leftBound = 0;
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] == '1') {
                // left[] is a rolling array
                left[j] = Math.max(left[j], leftBound);
                height[j]++;
                area = Math.max(area, height[j] * (right[j] - left[j] + 1));
            } else {
                left[j] = 0;
                height[j] = 0;
                leftBound = j + 1;
            }
        }
    }

    return area;
}
{% endhighlight %}

For example:

```
[1,0,1,0,0]
[1,0,1,1,1]
[1,1,1,1,1]
[1,0,0,1,0]
```

height[]:

```
[1,0,1,0,0]
[2,0,2,1,1]
[3,1,3,2,2]
[4,0,0,3,0]
```

left[]:

```
[0,0,2,0,0]
[0,0,2,2,2]
[0,0,2,2,2]
[0,0,0,3,0]
```

right[]:

```
[0,4,2,4,4]
[0,4,2,4,4]
[0,4,2,4,4]
[0,4,4,3,4]
```

[Largest Plus Sign][largest-plus-sign]

{% highlight java %}
for (int i = 0; i < N; i++) {
    // left
    count = 0;
    for (int j = 0; j < N; j++) {
	count = banned.contains(i * N + j) ? 0 : count + 1;
	dp[i][j] = count;
    }

    // right
    count = 0;
    for (int j = N - 1; j >= 0; j--) {
	count = banned.contains(i * N + j) ? 0 : count + 1;
	dp[i][j] = Math.min(dp[i][j], count);
    }
}

for (int j = 0; j < N; j++) {
    // up
    count = 0;
    for (int i = 0; i < N; i++) {
	count = banned.contains(i * N + j) ? 0 : count + 1;
	dp[i][j] = Math.min(dp[i][j], count);
    }

    // down
    count = 0;
    for (int i = N - 1; i >= 0; i--) {
	count = banned.contains(i * N + j) ? 0 : count + 1;
	dp[i][j] = Math.min(dp[i][j], count);
	ans = Math.max(ans, dp[i][j]);
    }
}
{% endhighlight %}

[Bomb Enemy][bomb-enemy]

{% highlight java %}
public int maxKilledEnemies(char[][] grid) {
    int m = grid.length, n = m == 0 ? 0 : grid[0].length;
    int count = 0, rowhits = 0;
    int[] colhits = new int[n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // resets rowhits after a wall
            if (j == 0 || grid[i][j - 1] == 'W') {
                rowhits = 0;
                for (int k = j; k < n && grid[i][k] != 'W'; k++) {
                    rowhits += grid[i][k] == 'E' ? 1 : 0;
                }
            }

            // resets colhits[j] below a wall
            if (i == 0 || grid[i - 1][j] == 'W') {
                colhits[j] = 0;
                for (int k = i; k < m && grid[k][j] != 'W'; k++) {
                    colhits[j] += grid[k][j] == 'E' ? 1 : 0;
                }
            }

            if (grid[i][j] == '0') {
                count = Math.max(count, rowhits + colhits[j]);
            }
        }
    }
    return count;
}
{% endhighlight %}

[Longest Line of Consecutive One in Matrix][longest-line-of-consecutive-one-in-matrix]

[Out of Boundary Paths][out-of-boundary-paths]

{% highlight java %}
{% raw %}
private static final int[][] DIRECTIONS = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
{% endraw %}
private static final int MOD = (int)1e9 + 7;

public int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
    int[][] dp = new int[m][n];
    dp[startRow][startColumn] = 1;

    int count = 0;
    for (int k = 1; k <= maxMove; k++) {
        int[][] tmp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int[] d : DIRECTIONS) {
                    int r = i + d[0], c = j + d[1];
                    if (r < 0 || r == m || c < 0 || c == n) {
                        count = (count + dp[i][j]) % MOD;
                    } else {
                        tmp[r][c] = (tmp[r][c] + dp[i][j]) % MOD;
                    }
                }
            }
        }
        dp = tmp;
    }
    return count;
}
{% endhighlight %}

[bomb-enemy]: https://leetcode.com/problems/bomb-enemy/
[largest-plus-sign]: https://leetcode.com/problems/largest-plus-sign/
[longest-line-of-consecutive-one-in-matrix]: https://leetcode.com/problems/longest-line-of-consecutive-one-in-matrix/
[maximal-rectangle]: https://leetcode.com/problems/maximal-rectangle/
[maximal-square]: https://leetcode.com/problems/maximal-square/
[out-of-boundary-paths]: https://leetcode.com/problems/out-of-boundary-paths/
