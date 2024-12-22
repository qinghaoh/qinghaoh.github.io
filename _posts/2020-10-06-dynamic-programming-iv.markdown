---
title:  "Dynamic Programming (Multi-dimension)"
category: algorithm
tag: dynamic programming
---
[Minimum Path Sum][minimum-path-sum]

```java
public int minPathSum(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    int[][] dp = new int[m][n];

    dp[m - 1][n - 1] = grid[m - 1][n - 1];

    for (int j = n - 2; j >= 0; j--) {
        dp[m - 1][j] = dp[m - 1][j + 1] + grid[m - 1][j];
    }

    for (int i = m - 2; i >= 0; i--) {
        dp[i][n - 1] = dp[i + 1][n - 1] + grid[i][n - 1];
    }

    for (int i = m - 2; i >= 0; i--) {
        for (int j = n - 2; j >= 0; j--) {
            dp[i][j] = Math.min(dp[i + 1][j], dp[i][j + 1]) + grid[i][j];
        }
    }

    return dp[0][0];
}
```

Reduce to 1D:

```java
int[] dp = new int[n];

dp[n - 1] = grid[m - 1][n - 1];

// last row
for (int j = n - 2; j >= 0; j--) {
    dp[j] = dp[j + 1] + grid[m - 1][j];
}

for (int i = m - 2; i >= 0; i--) {
    // last column
    dp[n - 1] += grid[i][n - 1];

    for (int j = n - 2; j >= 0; j--) {
        dp[j] = Math.min(dp[j], dp[j + 1]) + grid[i][j];
    }
}

return dp[0];
```

[Unique Paths II][unique-paths-ii]

```java
public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int m = obstacleGrid.length, n = obstacleGrid[0].length;
    int[][] dp = new int[m][n];

    dp[0][0] = 1 - obstacleGrid[0][0];

    // if there's an obstacle in the row, the following path is blocked
    for (int i = 1; i < m; i++) {
        dp[i][0] = (obstacleGrid[i][0] == 0 && dp[i - 1][0] == 1) ? 1 : 0;
    }

    // same as above
    for (int j = 1; j < n; j++) {
        dp[0][j] = (obstacleGrid[0][j] == 0 && dp[0][j - 1] == 1) ? 1 : 0;
    }

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[i][j] == 0) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            } else {
                dp[i][j] = 0;
            }
        }
    }

    return dp[m - 1][n - 1];
}
```

Reduced to 1D:

```java
public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int n = obstacleGrid[0].length;
    int[] dp = new int[n];
    dp[0] = 1;

    for (int[] row : obstacleGrid) {
        for (int j = 0; j < n; j++) {
            if (row[j] == 1) {
                dp[j] = 0;
            } else if (j > 0) {
                dp[j] += dp[j - 1];
            }
        }
    }
    return dp[n - 1];
}
```

[Maximum Number of Points with Cost][maximum-number-of-points-with-cost]

The most straight forward formula:

```java
for (int i = 1; i < m; i++) {
    long[] next = new long[n];
    for (int j = 0; j < n; j++) {
        next[j] = points[i][j];
        long d = dp[j];

        // dp[i - 1][k] +/- k is computed repeatedly
        for (int k = 0; k < n; k++) {
            d = Math.max(d, dp[k] - Math.abs(j - k));
        }
        next[j] += d;
    }
    dp = next;
}
```

To eliminate the duplicate computations:

```java
public long maxPoints(int[][] points) {
    int m = points.length, n = points[0].length;
    long[] dp = new long[n];

    for (int j = 0; j < n; j++) {
        dp[j] = points[0][j];
    }

    for (int i = 1; i < m; i++) {
        // finds the max of dp[k] +/- k
        long[] left = new long[n], right = new long[n];

        // assumes the max is on the left side of current
        // dp[j] = max(dp[k] + k) + points[i][j] - j for all 0 <= k <= j
        left[0] = dp[0];
        for (int j = 1; j < n; j++) {
            left[j] = Math.max(left[j - 1], dp[j] + j);
        }

        // assumes the max is on the right side of current
        // dp[j] = max(dp[k] - k) + points[i][j] + j for all j <= k < n
        right[n - 1] = dp[n - 1] - n + 1;
        for (int j = n - 2; j >= 0; j--) {
            right[j] = Math.max(right[j + 1], dp[j] - j);
        }

        long[] next = new long[n];
        for (int j = 0; j < n; j++) {
            next[j] = Math.max(left[j] + points[i][j] - j, right[j] + points[i][j] + j);
        }

        dp = next;
    }

    return Arrays.stream(dp).max().getAsLong();
}
```

[Dungeon Game][dungeon-game]

```java
public int calculateMinimumHP(int[][] dungeon) {
    int m = dungeon.length, n = dungeon[0].length;

    int[][] dp = new int[m][n];
    dp[m - 1][n - 1] = Math.max(1 - dungeon[m - 1][n - 1], 1);

    // last column
    for (int i = m - 2; i >= 0; i--) {
        dp[i][n - 1] = Math.max(dp[i + 1][n - 1] - dungeon[i][n - 1], 1);
    }

    // last row
    for (int j = n - 2; j >= 0; j--) {
        dp[m - 1][j] = Math.max(dp[m - 1][j + 1] - dungeon[m - 1][j], 1);
    }

    for (int i = m - 2; i >= 0; i--) {
        for (int j = n - 2; j >= 0; j--) {
            int down = Math.max(dp[i + 1][j] - dungeon[i][j], 1);
            int right = Math.max(dp[i][j + 1] - dungeon[i][j], 1);
            dp[i][j] = Math.min(right, down);
        }
    }

    return dp[0][0];
}
```

[Triangle][triangle]

```java
public int minimumTotal(List<List<Integer>> triangle) {
    int n = triangle.size();
    // bottom level
    Integer[] dp = triangle.get(n - 1).toArray(new Integer[0]);

    // from bottom to top
    for (int i = n - 2; i >= 0; i--) {
        for (int j = 0; j <= i; j++) {
            dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
        }
    }

    return dp[0];
}
```

[Cherry Pickup II][cherry-pickup-ii]

```c++
int cherryPickup(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    vector<vector<vector<int>>> dp(m, vector<vector<int>>(n, vector<int>(n)));

    for (int r = m - 1; r >= 0; r--) {
        for (int c1 = 0; c1 < n; c1++) {
            for (int c2 = 0; c2 < n; c2++) {
                // If c1 == c2, Robot #1 picks up the cherries
                // Do not double count
                int cherries = grid[r][c1];
                if (c1 != c2) {
                    cherries += grid[r][c2];
                }

                // Finds the max from the line below
                if (r != m - 1) {
                    int mx = 0;
                    for (int j1 = c1 - 1; j1 <= c1 + 1; j1++) {
                        for (int j2 = c2 - 1; j2 <= c2 + 1; j2++) {
                            if (j1 >= 0 && j1 < n && j2 >= 0 && j2 < n) {
                                mx = max(mx, dp[r + 1][j1][j2]);
                            }
                        }
                    }
                    cherries += mx;
                }
                dp[r][c1][c2] = cherries;
            }
        }
    }
    return dp[0][0][n - 1];
}
```

[Count Submatrices With All Ones][count-submatrices-with-all-ones]

```java
// O(n ^ 3)
public int numSubmat(int[][] mat) {
    int m = mat.length, n = mat[0].length;
    // dp[j]: count of consecutive ones in the j-th column in the current row
    int[] dp = new int[n];
    int count = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (mat[i][j] == 1) {
                dp[j]++;
            } else {
                dp[j] = 0;
            }
        }

        // count the number of submatrices with base mat[i][j...k]
        for (int j = 0; j < n; j++) {
            int min = m;
            for (int k = j; k < n; k++) {
                min = Math.min(min, dp[k]);
                count += min;
            }
        }
    }
    return count;
}
```

[Maximal Square][maximal-square]

```java
public int maximalSquare(char[][] matrix) {
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
```

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

Reduced to 1D:

```java
public int maximalSquare(char[][] matrix) {
    int m = matrix.length, n = matrix[0].length;
    int[] dp = new int[n + 1];
    int maxLen = 0, prev = 0;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int tmp = dp[j];
            if (matrix[i - 1][j - 1] == '1') {
                dp[j] = Math.min(Math.min(dp[j - 1], dp[j]), prev) + 1;
                maxLen = Math.max(maxLen, dp[j]);
            } else {
                dp[j] = 0;
            }
            prev = tmp;
        }
    }
    return maxLen * maxLen;
}
```

Similar: [Count Square Submatrices with All Ones][count-square-submatrices-with-all-ones]

```java
for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        if (matrix[i][j] == 1) {
            if (i > 0 && j > 0) {
                matrix[i][j] = Math.min(Math.min(matrix[i - 1][j], matrix[i][j - 1]), matrix[i - 1][j - 1]) + 1;
            }
            count += matrix[i][j];
        }
    }
}
```

[Maximal Rectangle][maximal-rectangle]

```java
public int maximalRectangle(char[][] matrix) {
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
```

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

[Count Fertile Pyramids in a Land][count-fertile-pyramids-in-a-land]

```java
public int countPyramids(int[][] grid) {
    return helper(inverse(grid)) + helper(grid);
}

public int helper(int[][] grid) {
    int count = 0;
    for (int i = 1; i < grid.length; i++) {
        for (int j = 1; j < grid[0].length - 1; j++) {
            if (grid[i][j] > 0) {
                // apex + three child pyramids
                grid[i][j] = Math.min(Math.min(grid[i - 1][j], grid[i - 1][j - 1]), grid[i - 1][j + 1]) + 1;
                // if grid[i][j] == k
                // there are (k - 1) pyramids whose apex is (i, j)
                count += grid[i][j] - 1;
            }
        }
    }
    return count;
}

public int[][] inverse(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    int[][] g = new int[m][n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            g[i][j] = grid[m - i - 1][j];
        }
    }
    return g;
}
```

[Selling Pieces of Wood][selling-pieces-of-wood]

```java
public long sellingWood(int m, int n, int[][] prices) {
    long[][] dp = new long[m + 1][n + 1];
    for (int[] p : prices) {
        dp[p[0]][p[1]] = p[2];
    }

    for (int w = 1; w <= m; w++) {
        for (int h = 1; h <= n; h++) {
            for (int a = 1; a <= w / 2; a++) {
                dp[w][h] = Math.max(dp[w][h], dp[a][h] + dp[w - a][h]);
            }
            for (int a = 1; a <= h / 2; a++) {
                dp[w][h] = Math.max(dp[w][h], dp[w][a] + dp[w][h - a]);
            }
        }
    }
    return dp[m][n];
}
```

[Largest Plus Sign][largest-plus-sign]

```java
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
```

[Bomb Enemy][bomb-enemy]

```java
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
```

[Longest Line of Consecutive One in Matrix][longest-line-of-consecutive-one-in-matrix]

[Out of Boundary Paths][out-of-boundary-paths]

```java
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
```

[Cherry Pickup][cherry-pickup]

```java
public int cherryPickup(int[][] grid) {
    // greedy (maximizing each pass) doesn't work
    // because the second pass depends on the path choice of the first pass
    int n = grid.length, m = (n << 1) - 1;

    // dp[r1][r2]: two people pick cherry from (0, 0) to (r1, c1) and (r2, c2), respectively
    // and they don't walk on the same cell except when (r1, c1) == (r2, c2)
    int[][] dp = new int[n][n];
    dp[0][0] = grid[0][0];

    // k == r1 + c1 == r2 + c2
    for (int k = 1; k < m; k++) {
        for (int r1 = n - 1; r1 >= 0; r1--) {
            for (int r2 = n - 1; r2 >= 0; r2--) {
                int c1 = k - r1, c2 = k - r2;

                // out of boundary or thorn
                if (c1 < 0 || c1 >= n || c2 < 0 || c2 >= n || grid[r1][c1] < 0 || grid[r2][c2] < 0) {
                    dp[r1][r2] = -1;
                    continue;
                 }

                 if (r1 > 0) {
                     dp[r1][r2] = Math.max(dp[r1][r2], dp[r1 - 1][r2]);
                 }
                 if (r2 > 0) {
                     dp[r1][r2] = Math.max(dp[r1][r2], dp[r1][r2 - 1]);
                 }
                 if (r1 > 0 && r2 > 0) {
                     dp[r1][r2] = Math.max(dp[r1][r2], dp[r1 - 1][r2 - 1]);
                 }

                 if (dp[r1][r2] >= 0) {
                     // don't double count if (r1, c1) == (r2, c2)
                     dp[r1][r2] += grid[r1][c1] + (r1 == r2 ? 0 : grid[r2][c2]);
                 }
             }
         }
    }
    return Math.max(0, dp[n - 1][n - 1]);
}
```

[Maximum Strictly Increasing Cells in a Matrix][maximum-strictly-increasing-cells-in-a-matrix]

```java
public int maxIncreasingCells(int[][] mat) {
    int m = mat.length, n = mat[0].length;
    Map<Integer, List<int[]>> map = new TreeMap<>();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            map.computeIfAbsent(mat[i][j], k -> new ArrayList<int[]>()).add(new int[]{i, j});
        }
    }

    // tmp[i][j]: the max number of cells that can be visited by starting from (i, j)
    int[][] tmp = new int[m][n];
    int[] dpRows = new int[m], dpCols = new int[n];
    int max = 0;
    // interates the cells in ascending order
    for (var v : map.values()) {
        for (int[] cell : v) {
            int i = cell[0], j = cell[1];
            tmp[i][j] = Math.max(dpRows[i], dpCols[j]) + 1;
            max = Math.max(max, tmp[i][j]);
        }
        for (int[] cell : v) {
            int i = cell[0], j = cell[1];
            dpRows[i] = Math.max(dpRows[i], tmp[i][j]);
            dpCols[j] = Math.max(dpCols[j], tmp[i][j]);
        }
    }
    return max;
}
```

[bomb-enemy]: https://leetcode.com/problems/bomb-enemy/
[cherry-pickup]: https://leetcode.com/problems/cherry-pickup/
[cherry-pickup-ii]: https://leetcode.com/problems/cherry-pickup-ii/
[count-fertile-pyramids-in-a-land]: https://leetcode.com/problems/count-fertile-pyramids-in-a-land/
[count-square-submatrices-with-all-ones]: https://leetcode.com/problems/count-square-submatrices-with-all-ones/
[count-submatrices-with-all-ones]: https://leetcode.com/problems/count-submatrices-with-all-ones/
[dungeon-game]: https://leetcode.com/problems/dungeon-game/
[largest-plus-sign]: https://leetcode.com/problems/largest-plus-sign/
[longest-line-of-consecutive-one-in-matrix]: https://leetcode.com/problems/longest-line-of-consecutive-one-in-matrix/
[maximal-rectangle]: https://leetcode.com/problems/maximal-rectangle/
[maximal-square]: https://leetcode.com/problems/maximal-square/
[maximum-number-of-points-with-cost]: https://leetcode.com/problems/maximum-number-of-points-with-cost/
[maximum-strictly-increasing-cells-in-a-matrix]: https://leetcode.com/problems/maximum-strictly-increasing-cells-in-a-matrix/
[minimum-path-sum]: https://leetcode.com/problems/minimum-path-sum/
[out-of-boundary-paths]: https://leetcode.com/problems/out-of-boundary-paths/
[selling-pieces-of-wood]: https://leetcode.com/problems/selling-pieces-of-wood/
[triangle]: https://leetcode.com/problems/triangle/
[unique-paths-ii]: https://leetcode.com/problems/unique-paths-ii/
