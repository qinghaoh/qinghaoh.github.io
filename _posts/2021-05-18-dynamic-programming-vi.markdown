---
title:  "Dynamic Programming VI"
category: algorithm
tag: dynamic programming
---
[Best Team With No Conflicts][best-team-with-no-conflicts]

```java
public int bestTeamScore(int[] scores, int[] ages) {
    int n = ages.length;
    Integer[] indices = new Integer[n];
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    Arrays.sort(indices, (i, j) -> ages[i] == ages[j] ? scores[i] - scores[j] : ages[i] - ages[j]);

    // dp[i]: max score if the i-th player is chosen and all the other players are between 0 and (i - 1)
    int[] dp = new int[n];
    int max = dp[0] = scores[indices[0]];
    for (int i = 1; i < n; i++) {
       dp[i] = scores[indices[i]];
       for (int j = 0; j < i; j++) {
           // age[indices[j]] <= age[indices[i]],
           // so we can always choose the younger player
           // if s/he has a lower score
           if (scores[indices[j]] <= scores[indices[i]]) {
               dp[i] = Math.max(dp[i], scores[indices[i]] + dp[j]);
           }  
       }
       max = Math.max(dp[i], max);
    }
    return max;
}
```

Alternative representation:

```java
int[][] candidate = new int[n][2];
       
for (int i = 0; i < n; i++) {
    candidate[i][0] = ages[i];
    candidate[i][1] = scores[i];
}

Arrays.sort(candidate, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
```

[Minimum Time to Make Array Sum At Most x][minimum-time-to-make-array-sum-at-most-x]

```java
public int minimumTime(List<Integer> nums1, List<Integer> nums2, int x) {
    int n = nums1.size();
    Integer[] indices = new Integer[n];
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    // Each nums1[i] is set to 0 at most once.
    // After t seconds, if we don't do the operation, then the sum would be sum2 * t + sum1.
    // With that operation at each second, The total sum of added nums2 elements is:
    //   nums2[i_0] * (t - 1) + nums2[i_1] * (t - 2) + ... + nums2[i_t] * 0
    // So, it's optimal to follow the ascending order of nums2
    Arrays.sort(indices, Comparator.comparingInt(i -> nums2.get(i)));

    // dp[i][j]: deducted value at j-th second with numbers chosen from sorted_nums1[0 ... i]
    int[] dp = new int[n + 1];
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j > 0; j--) {
            // dp[i][j] = Math.max(dp[i - 1][j] + dp[i - 1][j - 1] + sorted_nums1[i] + sorted_nums2[i] * j)
            // sorted_nums1[i] + sorted_nums2[i] * j is the deducted value if we choose i
            dp[j] = Math.max(dp[j], dp[j - 1] + j * nums2.get(indices[i]) + nums1.get(indices[i]));
        }
    }

    int sum1 = (int)nums1.stream().mapToInt(i -> i).sum(), sum2 = (int)nums2.stream().mapToInt(i -> i).sum();
    for (int i = 0; i <= n; i++) {
        if (sum2 * i + sum1 - dp[i] <= x) {
            return i;
        }
    }
    return -1;
}
```

[Maximum Height by Stacking Cuboids][maximum-height-by-stacking-cuboids]

```java
nt n = cuboids.length, max = 0;
int[] dp = new int[n];
for (int j = 0; j < n; j++) {
    dp[j] = cuboids[j][2];
    for (int i = 0; i < j; i++) {
        if (cuboids[i][0] >= cuboids[j][0] && cuboids[i][1] >= cuboids[j][1] && cuboids[i][2] >= cuboids[j][2]) {
            dp[j] = Math.max(dp[j], dp[i] + cuboids[j][2]);
        }
    }
    max = Math.max(max, dp[j]);
}
```

[Build Array Where You Can Find The Maximum Exactly K Comparisons][build-array-where-you-can-find-the-maximum-exactly-k-comparisons]

```java
private static final int MOD = (int)1e9 + 7;

public int numOfArrays(int n, int m, int k) {
    // dp[a][b][c]: max element == b
    long[][][] dp = new long[n + 1][m + 1][k + 1];

    // all one's
    for (int b = 1; b <= m; b++) {
        dp[1][b][1] = 1;
    }

    for (int a = 1; a <= n; a++) {
        for (int b = 1; b <= m; b++) {
            for (int c = 1; c <= k; c++) {
                long sum = 0;

                // dp[a][b][c] += b * dp[a - 1][b][c]
                // appends any element from [1, b] to the end of every array
                sum = (sum + b * dp[a - 1][b][c]) % MOD;

                // dp[a][b][c] += dp[a - 1][1][c - 1] + dp[a - 1][2][c - 1] + ... + dp[a - 1][b - 1][c - 1]
                // appends the element "b" to the end of every array
                for (int j = 1; j < b; j++) {
                    sum = (sum + dp[a - 1][j][c - 1]) % MOD;
                }

                dp[a][b][c] = (dp[a][b][c] + sum) % MOD;
            }
        }
    }

    long count = 0;
    for (int b = 1; b <= m; b++) {
        count = (count + dp[n][b][k]) % MOD;
    }

    return (int)count;
}
```

Prefix sum:

```java
// dp[a][b][c]: max element == b
long[][][] dp = new long[n + 1][m + 1][k + 1];
// prefix sum
long[][][] p = new long[n + 1][m + 1][k + 1];

// all one's
for (int b = 1; b <= m; b++) {
    dp[1][b][1] = 1;
    p[1][b][1] = p[1][b - 1][1] + 1;
}

for (int a = 1; a <= n; a++) {
    for (int b = 1; b <= m; b++) {
        for (int c = 1; c <= k; c++) {
            long sum = 0;

            // dp[a][b][c] += b * dp[a - 1][b][c]
            // appends any element from [1, b] to the end of every array
            sum = (sum + b * dp[a - 1][b][c]) % MOD;

            // dp[a][b][c] += dp[a - 1][1][c - 1] + dp[a - 1][2][c - 1] + ... + dp[a - 1][b - 1][c - 1]
            // appends the element "b" to the end of every array
            sum = (sum + p[a - 1][b - 1][c - 1]) % MOD;

            dp[a][b][c] = (dp[a][b][c] + sum) % MOD;
            p[a][b][c] = (dp[a][b][c] + p[a][b - 1][c]) % MOD;
        }
    }
}
```

[Number of Ways to Rearrange Sticks With K Sticks Visible][number-of-ways-to-rearrange-sticks-with-k-sticks-visible]

```java
private static final int MOD = (int)1e9 + 7;

public int rearrangeSticks(int n, int k) {
    long[][] dp = new long[n + 1][k + 1];
    dp[0][0] = 1;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= k; j++) {
            // f(n - 1, k - 1): rightmost is the longest
            // f(n - 1, k): rightmost is not the longest, and there are (n - 1) possibilities
            dp[i][j] = (dp[i - 1][j - 1] + (i - 1) * dp[i - 1][j] % MOD) % MOD;
        }
    }
    return (int)dp[n][k];
}
```

[Number of Music Playlists][number-of-music-playlists]

```java
private static final int MOD = (int)1e9 + 7;

public int numMusicPlaylists(int n, int goal, int k) {
    long[][] dp = new long[goal + 1][n + 1];
    dp[0][0] = 1;

    for (int i = 1; i <= goal; i++) {
        for (int j = 1; j <= n; j++) {
            // the last song is new
            dp[i][j] = (dp[i - 1][j - 1] * (n - (j - 1))) % MOD;

            // the last song is old
            // the songs from (j - k) to (j - 1) cannot be chosen
            if (j > k) {
                dp[i][j] = (dp[i][j] + (dp[i - 1][j] * (j - k)) % MOD) % MOD;
            }
        }
    }
    return (int)dp[goal][n];
}
```

[Frog Jump][frog-jump]

```java
public boolean canCross(int[] stones) {
    // stone : set of jump sizes which lead to the stone
    HashMap<Integer, Set<Integer>> map = new HashMap<>();
    for (int s : stones) {
        map.put(s, new HashSet<>());
    }

    map.get(0).add(0);
    for (int s : stones) {
        for (int k : map.get(s)) {
            // finds all stones that can be reached from the current stone
            for (int step = k - 1; step <= k + 1; step++) {
                if (step > 0 && map.containsKey(s + step)) {
                    map.get(s + step).add(step);
                }
            }
        }
    }
    return !map.get(stones[stones.length - 1]).isEmpty();
}
```

[Stone Game V][stone-game-v]

```java
// O(n ^ 3)
public int stoneGameV(int[] stoneValue) {
    int n = stoneValue.length;
    int[] p = new int[n + 1];
    for (int i = 0; i < n; i++) {
        p[i + 1] = p[i] + stoneValue[i];
    }

    int[][] dp = new int[n][n];
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int max = 0;
            for (int j = i; j + 1 <= i + len - 1; j++) {
                // [i, j] and [j + 1, i + len - 1]
                int left = p[j + 1] - p[i];
                int right = p[i + len] - p[j + 1];
                if (left == right) {
                    max = Math.max(max, left + dp[i][j]);
                    max = Math.max(max, right + dp[j + 1][i + len - 1]);
                } else if (left < right) {
                    max = Math.max(max, left + dp[i][j]);
                } else {
                    max = Math.max(max, right + dp[j + 1][i + len - 1]);
                }
            }
            dp[i][i + len - 1] = max;
        }
    }
    return dp[0][n - 1];
}
```

```java
// O(n ^ 2)
public int stoneGameV(int[] stoneValue) {
    int n = stoneValue.length;
    int[][] dp = new int[n][n], max = new int[n][n];

    // i <= j
    // max[i][j]: max(dp[i][i] + sum[i...i], dp[i][i + 1] + sum[i...(i + 1)], ..., dp[i][j] + sum[i...j]), i.e. left
    // max[j][i]: max(dp[i][j] + sum[i...j], dp[i + 1][j] + sum[(i + 1)...j], ..., dp[j][j] + sum[j...j]), i.e. right
    for (int i = 0; i < n; i++) {
        max[i][i] = stoneValue[i];
    }

    for (int j = 1; j < n; j++) {
        int mid = j, sum = stoneValue[j], rightHalf = 0;
        for (int i = j - 1; i >= 0; i--) {
            // sum(stoneValue[i, j])
            sum += stoneValue[i];

            // finds the index mid in the range [i, j]
            // if stoneValue[mid] is added to right half
            // then left half < right half
            while ((rightHalf + stoneValue[mid]) * 2 <= sum) {
                rightHalf += stoneValue[mid--];
            }

            // left remains
            // - if right half == left half, stoneValue[mid] is not added to right half
            //   so left half = max[i][mid]
            // - else, left half < right half, stoneValue[mid] is added to right half
            //   - mid == i means left is stoneValue[i], so Alice gets zero
            //   - else left half = max[i][mid - 1]
            dp[i][j] = rightHalf * 2 == sum ? max[i][mid] : (mid == i ? 0 : max[i][mid - 1]);

            // right remains
            // - if right half == left half, stoneValue[mid] is not added to right half
            //   so right half = max[j][mid + 1]
            // - else, left half > right half, stoneValue[mid] is not added to right half
            //   - mid == j means right is stoneValue[j], so Alice gets zero
            //   - else right half = max[j][mid + 1]
            dp[i][j] = Math.max(dp[i][j], mid == j ? 0 : max[j][mid + 1]);

            max[i][j] = Math.max(max[i][j - 1], dp[i][j] + sum);
            max[j][i] = Math.max(max[j][i + 1], dp[i][j] + sum);
        }
    }
    return dp[0][n - 1];
}
```

```java
// matrix[i][j] can be replaced by two arrays instead
left[i][j] = max(sum[i][k] + dp[i][k])
right[i][j] = max(sum[k][j] + dp[k][j])

left[i][j] = max(left[i][j-1], sum[i][j] + dp[i][j])
right[i][j] = max(right[i+1][j], sum[i][j] + dp[i][j])

dp[i][j] = max(left[i][mid], right[mid + 1][j])
```

[Paint House III][paint-house-iii]

```java
private static final int MAX = (int)1e6 + 1;

public int minCost(int[] houses, int[][] cost, int m, int n, int target) {
    // dp[i][j][k]: min cost where we have j neighborhood in the first i houses
    //   and the i-th house is painted with the color k
    int[][][] dp = new int[m + 1][target + 1][n];

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= target; j++) {
            Arrays.fill(dp[i][j], MAX);
        }
    }

    for (int k = 0; k < n; k++) {
        dp[0][0][k] = 0;
    }

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= Math.min(target, i); j++) {
            for (int k = 0; k < n; k++) {
                // the current house is houses[i - 1]
                // skips if it's painted and the painted color is not (k + 1)
                if (houses[i - 1] != 0 && k != houses[i - 1] - 1) {
                    continue;
                }

                // compares the current house with previous house
                int sameNeighborhood = dp[i - 1][j][k];

                int diffNeighborhood = MAX;
                for (int prevK = 0; prevK < n; prevK++) {
                    if (prevK != k) {
                        diffNeighborhood = Math.min(diffNeighborhood, dp[i - 1][j - 1][prevK]);
                    }
                }

                // paints the current house only if it's not pained yet
                int paintCost = cost[i - 1][k] * (houses[i - 1] == 0 ? 1 : 0);
                dp[i][j][k] = Math.min(sameNeighborhood, diffNeighborhood) + paintCost;
            }
        }
    }

    int min = MAX;
    for (int k = 0; k < n; k++) {
        min = Math.min(min, dp[m][target][k]);
    }

    return min == MAX ? -1 : min;
}
```

[Make the XOR of All Segments Equal to Zero][make-the-xor-of-all-segments-equal-to-zero]

```java
private static final int MAX = (int)1e6 + 1;

public int minCost(int[] houses, int[][] cost, int m, int n, int target) {
    // dp[i][j][k]: min cost where we have j neighborhood in the first i houses
    //   and the i-th house is painted with the color k
    int[][][] dp = new int[m + 1][target + 1][n];

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= target; j++) {
            Arrays.fill(dp[i][j], MAX);
        }
    }

    for (int k = 0; k < n; k++) {
        dp[0][0][k] = 0;
    }

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= Math.min(target, i); j++) {
            for (int k = 0; k < n; k++) {
                // the current house is houses[i - 1]
                // skips if it's painted and the painted color is not (k + 1)
                if (houses[i - 1] != 0 && k != houses[i - 1] - 1) {
                    continue;
                }

                // compares the current house with previous house
                int sameNeighborhood = dp[i - 1][j][k];

                int diffNeighborhood = MAX;
                for (int prevK = 0; prevK < n; prevK++) {
                    if (prevK != k) {
                        diffNeighborhood = Math.min(diffNeighborhood, dp[i - 1][j - 1][prevK]);
                    }
                }

                // paints the current house only if it's not pained yet
                int paintCost = cost[i - 1][k] * (houses[i - 1] == 0 ? 1 : 0);
                dp[i][j][k] = Math.min(sameNeighborhood, diffNeighborhood) + paintCost;
            }
        }
    }

    int min = MAX;
    for (int k = 0; k < n; k++) {
        min = Math.min(min, dp[m][target][k]);
    }

    return min == MAX ? -1 : min;
}
```

[Count Increasing Quadruplets][count-increasing-quadruplets]

```java
public long countQuadruplets(int[] nums) {
    int n = nums.length;
    // dp[j]: count of all valid triplets (i, j, k) so that i < j < k and nums[i] < nums[k] < nums[j]
    long[] dp = new long[n];
    long count = 0;
    for (int l = 0; l < n; l++) {
        // count of "l"s that can possibly be "k"s for newer "l"s
        int kCandidates = 0;
        for (int j = 0; j < l; j++) {
            if (nums[l] > nums[j]) {
                // "132" -> "1324": nums[j] => "3", nums[l] => "4"
                count += dp[j];
                kCandidates++;
            } else if (nums[l] < nums[j]) {
                // ("k candidate", j, l) becomes a valid triple
                dp[j] += kCandidates;
            }
        }
    }
    return count;
}
```

Another solution is by [prefix sum](../prefix-sum).

## Fractional DP

[Minimum Skips to Arrive at Meeting On Time][minimum-skips-to-arrive-at-meeting-on-time]

```java
public int minSkips(int[] dist, int speed, int hoursBefore) {
    int n = dist.length;
    // dp[i][j]: minimum arriving time * speed when we have travelled i roads and skipped j rests
    long[][] dp = new long[n + 1][n + 1];

    for (int j = 0; j <= n; j++) {
        for (int i = 0; i < n; i++) {
            // no skip, ceil
            dp[i + 1][j] = (dp[i][j] + dist[i] + speed - 1) / speed * speed;

            // skips current rest at i-th road
            if (j > 0) {
                dp[i + 1][j] = Math.min(dp[i + 1][j], dist[i] + dp[i][j - 1]);
            }
        }

        // min skips to arrive with time <= hoursBefore
        if (dp[n][j] <= speed * hoursBefore) {
            return j;
        }
    }
    return -1;
}
```

## Map

[Tallest Billboard][tallest-billboard]

```java
public int tallestBillboard(int[] rods) {
    // dp[i]: pair (a, b) with max a and b - a == i > 0
    Map<Integer, Integer> dp = new HashMap<>(), tmp;
    dp.put(0, 0);

    for (int r : rods) {
        tmp = new HashMap<>(dp);
        for (int d : tmp.keySet()) {
            // Case 1: put r to the long side
            // ---- v ----|-- d --|--- r ---|
            // ---- v ----|
            // dp[d + r] = max(dp[d + r], v)
            dp.put(d + r, Math.max(dp.getOrDefault(r + d, 0), tmp.get(d)));

            // Case 2: put r to the short side
            // ---- v ----|-- d --|
            // ---- v ----|--- r ---|
            // dp[r - d] = max(dp[r - d], v + d)
            // or
            // ---- v ----|-- d --|
            // ---- v ----|- r -|
            // dp[d - r] = max(dp[d - r], v + r)
            //
            // in summary,
            // dp[abs(d - r)] = max(dp[abs[d - r]], v + min(d, r))
            dp.put(Math.abs(d - r), Math.max(dp.getOrDefault(Math.abs(d - r), 0), tmp.get(d) + Math.min(d, r)));
        }
    }
    return dp.get(0);
}
```

[Stickers to Spell Word][stickers-to-spell-word]

```java
private Map<String, Integer> memo = new HashMap<>();
private int[][] countMap;

public int minStickers(String[] stickers, String target) {
    int n = stickers.length;
    this.countMap = new int[n][26];

    for (int i = 0; i < n; i++) {
        for (char c : stickers[i].toCharArray()) {
            countMap[i][c - 'a']++;
        }
    }

    memo.put("", 0);
    return dfs(target);
}

private int dfs(String target) {
    if (memo.containsKey(target)) {
        return memo.get(target);
    }

    int[] t = new int[26];
    for (char c : target.toCharArray()) {
        t[c - 'a']++;
    }

    int min = Integer.MAX_VALUE;
    StringBuilder sb = new StringBuilder();
    for (int[] s : countMap) {
        // the sticker has to contain the first character of target
        if (s[target.charAt(0) - 'a'] > 0) {
            // builds the string = (target - sticker)
            // the string is sorted
            for (int i = 0; i < 26; i++) {
                sb.append(String.valueOf((char)('a' + i)).repeat(Math.max(0, t[i] - s[i])));
            }

            int tmp = dfs(sb.toString());
            if (tmp != -1) {
                min = Math.min(min, tmp + 1);
            }
        }
        sb.setLength(0);
    }

    if (min == Integer.MAX_VALUE) {
        min = -1;
    }
    memo.put(target, min);
    return min;
}
```

[Minimum Distance to Type a Word Using Two Fingers][minimum-distance-to-type-a-word-using-two-fingers]

```java
public int minimumDistance(String word) {
    // distance is the total distance we get with right finger
    int distance = 0, save = 0;
    // dp[i]: the max distance that can be saved if left finger ends at character i
    int[] dp = new int[26];
    for (int i = 0; i < word.length() - 1; i++) {
        int curr = word.charAt(i) - 'A', next = word.charAt(i + 1) - 'A';
        for (int prev = 0; prev < 26; prev++) {
            // moves right finger from curr to next
            // or moves left finger from prev to next
            dp[curr] = Math.max(dp[curr], dp[prev] + cost(curr, next) - cost(prev, next));
        }
        save = Math.max(save, dp[curr]);

        // now right finger is at next, left finger is at curr
        distance += cost(curr, next);
    }
    return distance - save;
}

private int cost(int a, int b) {
    return Math.abs(a / 6 - b / 6) + Math.abs(a % 6 - b % 6);
}
```

[First Day Where You Have Been in All the Rooms][first-day-where-you-have-been-in-all-the-rooms]

```java
private static final int MOD = (int)1e9 + 7;

public int firstDayBeenInAllRooms(int[] nextVisit) {
    int n = nextVisit.length;
    long[] dp = new long[n];
    for (int i = 1; i < n; i++) {
        // 0 -> (i - 1): dp[i - 1]
        // (i - 1) -> nextVisit[i - 1]: 1
        // nextVisit[i - 1] -> (i - 1): dp[i - 1] - dp[nextVisit[i - 1]]
        // (i - 1) -> i: 1
        dp[i] = (2 * dp[i - 1] - dp[nextVisit[i - 1]] + 2 + MOD) % MOD;
    }
    return (int)dp[n - 1];
}
```

[Choose Numbers From Two Arrays in Range][choose-numbers-from-two-arrays-in-range]

```java
private static final int MOD = (int)1e9 + 7;

public int countSubranges(int[] nums1, int[] nums2) {
    // dp[i]: number of ways to sum to i
    Map<Integer, Integer> dp = new HashMap<>(), dp2;

    int count = 0;
    for (int i = 0; i < nums1.length; i++) {
        dp2 = new HashMap<>();
        dp2.put(nums1[i], 1);
        // negates nums2 elements
        // the goal is to find the number of different ranges that sum to 0
        dp2.put(-nums2[i], dp2.getOrDefault(-nums2[i], 0) + 1);

        for (var e : dp.entrySet()) {
            int k = e.getKey(), v = e.getValue();
            // picks nums1[i]
            dp2.put(k + nums1[i], (dp2.getOrDefault(k + nums1[i], 0) + v) % MOD);
            // picks -nums2[i]
            dp2.put(k - nums2[i], (dp2.getOrDefault(k - nums2[i], 0) + v) % MOD);
        }

        count = (count + dp2.getOrDefault(0, 0)) % MOD;
        dp = dp2;
    }
    return count;
}
```

[Minimum Total Space Wasted With K Resizing Operations][minimum-total-space-wasted-with-k-resizing-operations]

```java
public int minSpaceWastedKResizing(int[] nums, int k) {
    int n = nums.length, max = 0, sum = 0;
    int[][] dp = new int[n][k + 1];
    for (int i = n - 1; i >= 0; i--) {
        max = Math.max(max, nums[i]);
        sum += nums[i];
        dp[i][0] = max * (n - i) - sum;
    }

    for (int m = 1; m <= k; m++) {
        for (int i = n - 1; i >= 0; i--) {
            dp[i][m] = dp[i][m - 1];
            max = sum = 0;

            // resizes at i
            // finds the wasted space in [i, j)
            for (int j = i + 1; j < n; j++) {
                max = Math.max(max, nums[j - 1]);
                sum += nums[j - 1];
                dp[i][m] = Math.min(dp[i][m], dp[j][m - 1] + max * (j - i) - sum);
            }
        }
    }
    return dp[0][k];
}
```

[Minimum White Tiles After Covering With Carpets][minimum-white-tiles-after-covering-with-carpets]

```java
private static final int MAX_LENGTH = 1000;

public int minimumWhiteTiles(String floor, int numCarpets, int carpetLen) {
    int n = floor.length();
    // dp[i][j]: covers the first i tiles with j carpets
    int[][] dp = new int[n + 1][numCarpets + 1];
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= numCarpets; j++) {
            int skip = dp[i - 1][j] + floor.charAt(i - 1) - '0';
            int cover = j > 0 ? dp[Math.max(i - carpetLen, 0)][j - 1] : MAX_LENGTH;
            dp[i][j] = Math.min(cover, skip);
        }
    }
    return dp[n][numCarpets];
}
```

[Minimum Time to Finish the Race][minimum-time-to-finish-the-race]

```java
// f * r ^ (x - 1) >= changeTime
// if f == 1 and r == 2 (minimum)
// x >= 18
// i.e. it's better to change tire if the successiev laps >= 18
private static final int NUM_LAPS_TO_CHANGE_TIRE = 18;
// numLaps * max_time_per_lap
// = numLaps * (max_f + changeTime)
// = 2e8
private static final int MAX_TOTAL_TIME = (int)2e8;

public int minimumFinishTime(int[][] tires, int changeTime, int numLaps) {
    int n = tires.length;

    // noChange[i][j]: the total time to run j laps successively with tire i
    // j is small enough
    long[][] noChange = new long[n][NUM_LAPS_TO_CHANGE_TIRE];
    for (long[] t : noChange) {
        Arrays.fill(t, MAX_TOTAL_TIME);
    }

    for (int i = 0; i < n; i++) {
        noChange[i][1] = tires[i][0];
        // per lap
        for (int j = 2; j < NUM_LAPS_TO_CHANGE_TIRE; j++) {
            noChange[i][j] = noChange[i][j - 1] * tires[i][1];
            if (noChange[i][j] > MAX_TOTAL_TIME) {
                noChange[i][j] = MAX_TOTAL_TIME;
                break;
            }
        }

        // prefix sum
        for (int j = 2; j < NUM_LAPS_TO_CHANGE_TIRE; j++) {
            noChange[i][j] += noChange[i][j - 1];
            if (noChange[i][j] > MAX_TOTAL_TIME) {
                noChange[i][j] = MAX_TOTAL_TIME;
                break;
            }
        }
    }

    // dp[i]: the minimum time to finish i laps
    int[] dp = new int[numLaps + 1];
    Arrays.fill(dp, MAX_TOTAL_TIME);
    for (int[] tire : tires) {
        dp[1] = Math.min(dp[1], tire[0]);
    }

    for (int i = 1; i <= numLaps; i++) {
        if (i < NUM_LAPS_TO_CHANGE_TIRE) {
            // when i is small enough, the optimal solution might never change tire
            for (long[] t : noChange) {
                dp[i] = Math.min(dp[i], (int)t[i]);
            }
        }

        for (int j = i - 1; j > 0 && j >= i - NUM_LAPS_TO_CHANGE_TIRE; j--) {
            dp[i] = Math.min(dp[i], dp[j] + changeTime + dp[i - j]);
        }
    }

    return dp[numLaps];
}
```

The previous states of elements is stored in a map.

[Longest String Chain][longest-string-chain]

```java
public int longestStrChain(String[] words) {
    Arrays.sort(words, Comparator.comparingInt(s -> s.length()));

    Map<String, Integer> dp = new HashMap<>();
    int max = 0;
    for (String word : words) {
        int length = 0;
        for (int i = 0; i < word.length(); i++) {
            String predecessor = word.substring(0, i) + word.substring(i + 1);
            length = Math.max(length, dp.getOrDefault(predecessor, 0) + 1);
        }
        dp.put(word, length);
        max = Math.max(max, length);
    }
    return max;
}
```

[Make Array Strictly Increasing][make-array-strictly-increasing]

```java
public int makeArrayIncreasing(int[] arr1, int[] arr2) {
    Arrays.sort(arr2);

    // rolling dp
    // dp[i]: i is the element we choose for the current position.
    // this element can be from either arr1 or arr2.
    Map<Integer, Integer> dp = new HashMap<>();
    dp.put(-1, 0);

    for (int a1 : arr1) {
        // builds temporary dp map for i-th element of arr1
        Map<Integer, Integer> tmp = new HashMap<>();
        for (int key : dp.keySet()) {
            int val = dp.get(key);
            // option #1: no assignment for key -> a1
            if (a1 > key && (!tmp.containsKey(a1) || val < tmp.get(a1))) {
                tmp.put(a1, val);
            }

            // finds the smallest element in arr2 that's >= key
            int index = Arrays.binarySearch(arr2, key + 1);
            if (index < 0) {
                index = ~index;
            }

            // option #2: one assignment for key -> arr2[index]
            if (index < arr2.length && (!tmp.containsKey(arr2[index]) || val + 1 < tmp.get(arr2[index]))) {
                tmp.put(arr2[index], val + 1);
            }
        }
        dp = tmp;
    }

    return dp.isEmpty() ? -1 : Collections.min(dp.values());
}
```

## Reverse

[Coin Path][coin-path]

```java
public List<Integer> cheapestJump(int[] coins, int maxJump) {
    int n = coins.length;
    List<Integer> path = new ArrayList<>();
    if (coins[n - 1] < 0) {
        return path;
    }

    // dp[i]: cost from coins[i] to coins[n - 1]
    int[] dp = new int[n], next = new int[n];
    Arrays.fill(dp, Integer.MAX_VALUE);
    Arrays.fill(next, -1);

    dp[n - 1] = coins[n - 1];

    // reverse order to ensure we get the lexicographically smallest path
    for (int i = n - 2; i >= 0; i--) {
        if (coins[i] == -1) {
            continue;
        }

        for (int j = i + 1; j <= Math.min(i + maxJump, n - 1); j++) {
            // strict > guarantees lexicographical order
            if (dp[i] > dp[j] + coins[i] && dp[j] != Integer.MAX_VALUE) {
                dp[i] = dp[j] + coins[i];
                next[i] = j;
            }
        }
    }

    if (dp[0] == Integer.MAX_VALUE) {
        return path;
    }

    int index = 0;
    while (index != -1) {
        path.add(index + 1);
        index = next[index];
    }
    return path;
}
```

[Race Car][race-car]

```java
public int racecar(int target) {
    // dp[i]: the length of the shortest sequence of instructions from initial speed 1 to target i
    int[] dp = new int[target + 1];

    for (int i = 1; i <= target; i++) {
        dp[i] = Integer.MAX_VALUE;

        // j is the position of the car right before the first reverse instruction
        // j = 2 ^ a1 - 1, where a1 is the number of 'A's.
        int a1 = 1, j = 1;

        // if j < i, the reverse instruction is issued before the car reaches i
        // the car is going away from the target
        // we need to wait for the second reverse instruction
        for (; j < i; j = (1 << ++a1) - 1) {
            // j - k is the position of the car right before the second reverse instruction
            // k = 2 ^ a2 - 1
            // where a2 is the number of 'A's between j and the position of the second reverse instruction
            for (int a2 = 0, k = 0; k < j; k = (1 << ++a2) - 1) {
                // +1 are 'R's
                dp[i] = Math.min(dp[i], a1 + 1 + a2 + 1 + dp[i - (j - k)]);
            }
        }

        // if j == i, no reverse instructions
        // if j > i, only one reverse instruction
        dp[i] = Math.min(dp[i], a1 + (i == j ? 0 : 1 + dp[j - i]));
    }

    return dp[target];
}
```

## Precompute

[Number of Ways to Reach a Position After Exactly k Steps][number-of-ways-to-reach-a-position-after-exactly-k-steps]

```java
private static final int MOD = (int)1e9 + 7;
private static final int MAX_POS = 1000;

public int numberOfWays(int startPos, int endPos, int k) {
    // dp[i][j]: uses exact i steps to reach end, while distance between start (fixed to 0) and end is j
    int[][] dp = new int[MAX_POS + 1][MAX_POS + 1];
    for (int i = 1; i < dp.length; i++) {
        // one way to go distance i
        dp[i][i] = 1;
        for (int j = 0; j < i; j++) {
            dp[i][j] = (dp[i - 1][Math.abs(j - 1)] + dp[i - 1][j + 1]) % MOD;
        }
    }

    return dp[k][Math.abs(startPos - endPos)];
}
```

## Digit Dynamic Programming

[Count of Integers][count-of-integers]

```c++
    const int MOD = 1e9 + 7;
    // memo[index][isLowTight][isHighTight][sum]
    int memo[23][2][2][401];

    /**
     * @brief Counts the number of integers in [num1, num2] whose sum of digits <= `sum`.
     * @param index: Index of the current index. Index 0 stands for the most significant digit.
     * @param isLowTight: If all the selected digits so far are the lowest possible.
     * @param isHighTight: If all the selected digits so far are the highest possible.
     */
    int countStrings(int index, int sum, bool isLowTight, bool isHighTight, string num1, string num2) {
        if (sum < 0) {
            return 0;
        }

        if (index == num2.length()) {
            return 1;
        }

        if (memo[index][isLowTight][isHighTight][sum] >= 0) {
            return memo[index][isLowTight][isHighTight][sum];
        }

        int cnt = 0;
        // e.g. num1 == 234, index == 1
        //   If previous isLowTight is true, then the running number is 2**,
        //     if we want to keep the current digit low tight, then it has to be 3.
        //     any number lower than 3, e.g. 2, will yield 22* < 234
        //   otherwise, the running number is more flexible, e.g. 3**
        //     the current digit can be any number (as low as 0)
        // Same applies to isHighTight
        int low = isLowTight ? num1[index] - '0' : 0;
        int high = isHighTight ? num2[index] - '0' : 9;
        for (int d = low; d <= high; d++) {
            cnt = (cnt + countStrings(index + 1, sum - d, isLowTight && d == low, isHighTight && d == high, num1, num2) % MOD) % MOD;
        }
        return memo[index][isLowTight][isHighTight][sum] = cnt;
    }

public:
    int count(string num1, string num2, int min_sum, int max_sum) {
        memset(memo, -1, sizeof(memo));

        // Makes num1 and num2 equal length
        num1 = string(num2.length() - num1.length(), '0') + num1;

        // Initial value of `isTight` is `true`.
        int cnt1 = countStrings(0, max_sum, true, true, num1, num2);
        int cnt2 = countStrings(0, min_sum - 1, true, true, num1, num2);

        return (cnt1 - cnt2 + MOD) % MOD;
    }
```

[Number of Beautiful Integers in the Range][number-of-beautiful-integers-in-the-range]

```c++
    // memo[index][isLowTight][isHighTight][odd][even][mod][isZero]
    int memo[12][2][2][12][12][20][2];
    int k;

    int countIntegers(const string& s1, const string& s2, int index, bool isLowTight, bool isHighTight, int odd, int even, int mod, bool isZero) {
        if (index == s2.length()) {
            return !isZero && mod == 0 && odd == even;
        }

        if (memo[index][isLowTight][isHighTight][odd][even][mod][isZero] >= 0) {
            return memo[index][isLowTight][isHighTight][odd][even][mod][isZero];
        }

        int low = isLowTight ? s1[index] - '0' : 0;
        int high = isHighTight ? s2[index] - '0' : 9;
        int cnt = 0;
        for (int d = low; d <= high; d++) {
            // `isZero` means the digits at [0, index) are all zeros .
            // We don't count odds and evens for numbers with leading zeros,
            // e.g. "0032" is not a valid beautiful integer for k = 2.
            if (isZero && !d) {
                // isZero && d == 0
                //   => low == 0 && isLowTight == true
                //   => new isLowTight = (isLowTight && d == low) == true
                //
                // Since s2 has no leading zeros, the current number with isZero == true can't be high tight.
                // Therefore, new isHighTight == false.
                cnt += countIntegers(s1, s2, index + 1, true, false, odd, even, mod, true);
            } else {
                cnt += countIntegers(s1, s2, index + 1, isLowTight && d == low, isHighTight && d == high, odd + d % 2, even + 1 - d % 2, (mod * 10 + d) % k, false);
            }
        }
        return memo[index][isLowTight][isHighTight][odd][even][mod][isZero] = cnt;
    }

public:
    int numberOfBeautifulIntegers(int low, int high, int k) {
        this->k = k;

        memset(memo, -1, sizeof(memo));

        // Makes num1 and num2 equal length
        string s1 = to_string(low), s2 = to_string(high);
        s1 = string(s2.length() - s1.length(), '0') + s1;
        return countIntegers(s1, s2, 0, true, true, 0, 0, 0, true);
    }
```

{: .prompt-info }
> You can process `low` and `high` separately with two calls, eliminating the need to pad `low`.

```c++
    // memo[index][isTight][odd][even][mod][isZero]
    int memo[12][2][12][12][20][2];
    int k;

    int countIntegers(const string& s, int index = 0, bool isTight = true, int odd = 0, int even = 0, int mod = 0, bool isZero = true) {
        if (index == s.size()) {
            return !isZero && mod == 0 && odd == even;
        }

        if (memo[index][isTight][odd][even][mod][isZero] >= 0) {
            return memo[index][isTight][odd][even][mod][isZero];
        }

        int high = isTight ? s[index] - '0' : 9;
        int cnt = 0;
        for (int d = 0; d <= high; d++) {
            // `isZero` means the digits at [0, index) are all zeros .
            // We don't count odds and evens for numbers with leading zeros,
            // e.g. "0032" is not a valid beautiful integer for k = 2.
            if (isZero && !d) {
                // Since s has no leading zeros, the current number with isZero == true can't be high tight.
                // Therefore, new isHighTight == false.
                cnt += countIntegers(s, index + 1, false, odd, even, mod, true);
            } else {
                cnt += countIntegers(s, index + 1, isTight && d == high, odd + d % 2, even + 1 - d % 2, (mod * 10 + d) % k, false);
            }
        }
        return memo[index][isTight][odd][even][mod][isZero] = cnt;
    }

public:
    int numberOfBeautifulIntegers(int low, int high, int k) {
        this->k = k;

        memset(memo, -1, sizeof(memo));
        int cnt1 = countIntegers(to_string(high));

        memset(memo, -1, sizeof(memo));
        int cnt2 = countIntegers(to_string(low - 1));

        return cnt1 - cnt2;
    }
```

[Count the Number of Powerful Integers][count-the-number-of-powerful-integers]

This problem can be solved by digit DP, however, there is a more efficient and smarter solution (credits to @abhik2003):

```c++
    string s;
    int limit;

    long long countInts(const string& num) {
        // Length of prefix
        int p = num.length() - s.length();
        if (p < 0) {
            return 0;
        }

        if (p == 0) {
            return num >= s;
        }

        string suffix = num.substr(p, s.length());
        long long res = 0;
        // Iterates through nums[0...p)
        for (int i = 0; i < p; i++) {
            if (num[i] - '0' > limit) {
                // The remaining digits in the prefix (index [i, p)) can pick any number in [0, limit].
                // In this way, the built string (p + s) < num.
                return res + pow(limit + 1, p - i);
            }

            // num[i] - '0' <= limit
            // If we pick any number in [0, num[i]),
            //   the remaining digits in the prefix (index (i, p)) can pick any number in the range [0, limit].
            // If we pick num[i] (tight case),
            //   the iteration needs to go on for possibly more candidates.
            res += (num[i] - '0') * pow(limit + 1, p - i - 1);
        }

        // Code reaches here only if the prefix has been tight up to p - 1.
        // so suffix >= s is the requirement to make (p + s) <= num.
        return res + (suffix >= s);
    }

public:
    long long numberOfPowerfulInt(long long start, long long finish, int limit, string s) {
        this->limit = limit;
        this->s = s;

        return countInts(to_string(finish)) - countInts(to_string(start - 1));
    }
```

[best-team-with-no-conflicts]: https://leetcode.com/problems/best-team-with-no-conflicts/
[build-array-where-you-can-find-the-maximum-exactly-k-comparisons]: https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons/
[choose-numbers-from-two-arrays-in-range]: https://leetcode.com/problems/choose-numbers-from-two-arrays-in-range/
[coin-path]: https://leetcode.com/problems/coin-path/
[count-increasing-quadruplets]: https://leetcode.com/problems/count-increasing-quadruplets/
[count-of-integers]: https://leetcode.com/problems/count-of-integers/
[count-the-number-of-powerful-integers]: https://leetcode.com/problems/count-the-number-of-powerful-integers/
[first-day-where-you-have-been-in-all-the-rooms]: https://leetcode.com/problems/first-day-where-you-have-been-in-all-the-rooms/
[frog-jump]: https://leetcode.com/problems/frog-jump/
[longest-string-chain]: https://leetcode.com/problems/longest-string-chain/
[make-array-strictly-increasing]: https://leetcode.com/problems/make-array-strictly-increasing/
[make-the-xor-of-all-segments-equal-to-zero]: https://leetcode.com/problems/make-the-xor-of-all-segments-equal-to-zero/
[maximum-height-by-stacking-cuboids]: https://leetcode.com/problems/maximum-height-by-stacking-cuboids/
[minimum-distance-to-type-a-word-using-two-fingers]: https://leetcode.com/problems/minimum-distance-to-type-a-word-using-two-fingers/
[minimum-skips-to-arrive-at-meeting-on-time]: https://leetcode.com/problems/minimum-skips-to-arrive-at-meeting-on-time/
[minimum-time-to-finish-the-race]: https://leetcode.com/problems/minimum-time-to-finish-the-race/
[minimum-time-to-make-array-sum-at-most-x]: https://leetcode.com/problems/minimum-time-to-make-array-sum-at-most-x/
[minimum-total-space-wasted-with-k-resizing-operations]: https://leetcode.com/problems/minimum-total-space-wasted-with-k-resizing-operations/
[minimum-white-tiles-after-covering-with-carpets]: https://leetcode.com/problems/minimum-white-tiles-after-covering-with-carpets/
[number-of-beautiful-integers-in-the-range]: https://leetcode.com/problems/number-of-beautiful-integers-in-the-range/
[number-of-music-playlists]: https://leetcode.com/problems/number-of-music-playlists/
[number-of-ways-to-reach-a-position-after-exactly-k-steps]: https://leetcode.com/problems/number-of-ways-to-reach-a-position-after-exactly-k-steps/
[number-of-ways-to-rearrange-sticks-with-k-sticks-visible]: https://leetcode.com/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/
[paint-house-iii]: https://leetcode.com/problems/paint-house-iii/
[race-car]: https://leetcode.com/problems/race-car/
[stickers-to-spell-word]: https://leetcode.com/problems/stickers-to-spell-word/
[stone-game-v]: https://leetcode.com/problems/stone-game-v/
[tallest-billboard]: https://leetcode.com/problems/tallest-billboard/
