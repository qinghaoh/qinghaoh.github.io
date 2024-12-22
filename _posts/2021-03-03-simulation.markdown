---
title:  "Simulation"
category: algorithm
---
[Pour Water][pour-water]

```java
public int[] pourWater(int[] heights, int V, int K) {
    while (V > 0) {
        int i = K;
        while (i > 0 && heights[i] >= heights[i - 1]) {
            i--;
        }
        while (i < heights.length - 1 && heights[i] >= heights[i + 1]) {
            i++;
        }
        while (i > K && heights[i] == heights[i - 1]) {
            i--;
        }
        heights[i]++;
        V--;
    }
    return heights;
}
```

[Champagne Tower][champagne-tower]

```java
public double champagneTower(int poured, int query_row, int query_glass) {
    double[][] glass = new double[query_row + 2][query_row + 2];
    glass[0][0] = poured;

    for (int i = 0; i <= query_row; i++) {
        for (int j = 0; j <= i; j++) {
            if (glass[i][j] > 1) {
                double water = (glass[i][j] - 1) / 2;
                glass[i + 1][j] += water;
                glass[i + 1][j + 1] += water;
            }
        }
    }

    return Math.min(glass[query_row][query_glass], 1);
}
```

Reduced to 1D:

```java
public double champagneTower(int poured, int query_row, int query_glass) {
    double[] glass = new double[query_row + 2];
    glass[0] = poured;

    for (int i = 1; i <= query_row; i++) {
        for (int j = i; j >= 0; j--) {
            double water = Math.max((glass[j] - 1) / 2, 0);
            glass[j] = water;
            glass[j + 1] += water;
        }
    }

    return Math.min(glass[query_glass], 1);
}
```

[Dota2 Senate][dota2-senate]

```java
public String predictPartyVictory(String senate) {
    Queue<Integer> r = new LinkedList<>(), d = new LinkedList<>();
    int n = senate.length();
    for (int i = 0; i< n; i++) {
        if (senate.charAt(i) == 'R') {
            r.offer(i);
        } else {
            d.offer(i);
        }
    }

    while (!r.isEmpty() && !d.isEmpty()) {
        int ri = r.poll(), di = d.poll();
        if (ri < di) {
            r.offer(ri + n);
        } else {
            d.offer(di + n);
        }
    }
    return r.size() > d.size() ? "Radiant" : "Dire";
}
```

[Where Will the Ball Fall][where-will-the-ball-fall]

```java
public int[] findBall(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    int[] answer = new int[n];
    for (int j = 0; j < n; j++) {
        int j1 = j, j2 = 0;
        for (int i = 0; i < m; i++) {
            // next possible column
            j2 = j1 + grid[i][j1];

            // if a ball can move from j1 to j2
            // grid[i][j1] == grid[i][j2]
            if (j2 < 0 || j2 >= n || grid[i][j2] != grid[i][j1]) {
                j1 = -1;
                break;
            }

            // prepares for next row
            j1 = j2;
        }
        answer[j] = j1;
    }
    return answer;
}
```

[Number of Spaces Cleaning Robot Cleaned][number-of-spaces-cleaning-robot-cleaned]

When the robot reaches a space that it has already cleaned and is facing the same direction as before, we can stop the simulation.

[Reveal Cards In Increasing Order][reveal-cards-in-increasing-order]

```java
public int[] deckRevealedIncreasing(int[] deck) {
    Arrays.sort(deck);
    int n = deck.length;
    int[] order = new int[n];
    Queue<Integer> q = new LinkedList<>();
    for (int i = 0; i < n; i++) {
        q.offer(i);
    }

    int i = 0;
    while (i < n) {
        order[q.poll()] = deck[i++];
        q.offer(q.poll());
    }
    return order;
}
```

[Find Latest Group of Size M][find-latest-group-of-size-m]

```java
public int findLatestStep(int[] arr, int m) {
    int n = arr.length;
    if (n == m) {
        return n;
    }

    int[] length = new int[n + 2];
    int step = -1;
    for (int i = 0; i < n; i++) {
        int index = arr[i];
        int left = length[index - 1], right = length[index + 1];
        if (left == m || right == m) {
            step = i;
        }
        length[index + right] = length[index - left] = length[index] = left + right + 1;
    }
    return step;
}
```

[champagne-tower]: https://leetcode.com/problems/champagne-tower/
[dota2-senate]: https://leetcode.com/problems/dota2-senate/
[find-latest-group-of-size-m]: https://leetcode.com/problems/find-latest-group-of-size-m/
[number-of-spaces-cleaning-robot-cleaned]: https://leetcode.com/problems/number-of-spaces-cleaning-robot-cleaned/
[pour-water]: https://leetcode.com/problems/pour-water/
[reveal-cards-in-increasing-order]: https://leetcode.com/problems/reveal-cards-in-increasing-order/
[where-will-the-ball-fall]: https://leetcode.com/problems/where-will-the-ball-fall/
