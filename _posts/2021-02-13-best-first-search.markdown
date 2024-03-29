---
title:  "Best First Search"
category: algorithm
tags: graph
---
# A\* Search

[A\* search algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm) selects the path that minimizes

```f(n) = g(n) + h(n)```

where 
* `n` is the next node on the path
* `g(n)` is the cost of the path from the start node to `n`
* `h(n)` is a heuristic function that estimates the cost of the cheapest path from `n` to the goal
  * problem-specific
  * admissible: it never overestimates the cost of reaching the goal - A* is guaranteed to return a least-cost path
  * monotone/consistent: its estimate is always less than or equal to the estimated distance from any neighbouring vertex to the goal, plus the cost of reaching that neighbour - A* is guaranteed to find an optimal path without processing any node more than once and A* is equivalent to running Dijkstra's algorithm with the reduced cost `d'(x, y) = d(x, y) + h(y) − h(x)`

[Shortest Path in Binary Matrix][shortest-path-in-binary-matrix]

```java
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
{% endraw %}
private static final int MAX = 200;

public int shortestPathBinaryMatrix(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    if (grid[m - 1][n - 1] == 1) {
        return -1;
    }

    // x, y, g, f
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[3]));
    pq.offer(new int[]{0, 0, 1, 1 + Math.max(m, n)});

    Map<Integer, Integer> minDist = new HashMap<>();

    while (!pq.isEmpty()) {
        int[] node = pq.poll();
        int r = node[0], c = node[1];
        if (r < 0 || r == m || c < 0 || c == n || grid[r][c] == 1) {
            continue;
        }
        if (r == m - 1 && c == n - 1) {
            return node[2];
        }

        if (minDist.getOrDefault(r * n + c, MAX) <= node[2]) {
            continue;
        }

        minDist.put(r * n + c, node[2]);

        for (int[] d : DIRECTIONS) {
            int g = node[2] + 1;

            // h(n) is a heuristic function that 
            // estimates the cost of the cheapest path from node to the goal
            // here h(n) = diagonal distance (max norm)
            int h = Math.max(Math.abs(m - 1 - r), Math.abs(n - 1 - c));

            pq.offer(new int[]{r + d[0], c + d[1], g, g + h});
        }
    }

    return -1;
}
```

[Minimum Moves to Move a Box to Their Target Location][minimum-moves-to-move-a-box-to-their-target-location]

```java
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
{% endraw %}
public int minPushBox(char[][] grid) {
    int m = grid.length, n = grid[0].length;
    int[] start = null, box = null, target = null;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 'S') {
                start = new int[]{i, j};
            } else if (grid[i][j] == 'B') {
                box = new int[]{i, j};
            } else if (grid[i][j] == 'T') {
                target = new int[]{i, j};
            }
        }
    }

    // {h, pushes, player[0], player[1], box[0], box[1]}
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
    int[] node = new int[]{h(box[0], box[1], target), 0, start[0], start[1], box[0], box[1]};
    pq.offer(node);

    Set<String> set = new HashSet<>();
    while (!pq.isEmpty()) {
        node = pq.poll();
        // box is at the target
        if (node[4] == target[0] && node[5] == target[1]) {
            return node[1];
        }

        // set key
        String key = Arrays.stream(node, 2, node.length)
            .mapToObj(String::valueOf)
            .collect(Collectors.joining(" - "));

        if (!set.add(key)) {
            continue;
        }

        for (int[] d : DIRECTIONS) {
            // player
            int pr = node[2] + d[0], pc = node[3] + d[1];
            if (isBlocked(grid, pr, pc, m, n)) {
                continue;
            }

            int[] next = null;
            if (pr == node[4] && pc == node[5]) {
                // player is at box
                int br = node[4] + d[0], bc = node[5] + d[1];
                if (isBlocked(grid, br, bc, m, n)) {
                    continue;
                }
                next = new int[]{h(br, bc, target) + node[1] + 1, node[1] + 1, pr, pc, br, bc};
            } else {
                // box doesn't move
                next = new int[]{node[0], node[1], pr, pc, node[4], node[5]};
            }
            pq.offer(next);
        }
    }
    return -1;
}

private boolean isBlocked(char[][] grid, int i, int j, int m, int n) {
    return i < 0 || i == m || j < 0 || j == n || grid[i][j] == '#';
}

// Manhattan's distance
private int h(int box0, int box1, int[] target) {
    return Math.abs(box0 - target[0]) + Math.abs(box1 - target[1]);
}
```

[minimum-moves-to-move-a-box-to-their-target-location]: https://leetcode.com/problems/minimum-moves-to-move-a-box-to-their-target-location/
[shortest-path-in-binary-matrix]: https://leetcode.com/problems/shortest-path-in-binary-matrix/
