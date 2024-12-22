---
title:  "Shortest Path"
category: algorithm
tags: graph
mermaid: true
---
# Dijkstra

## Fundamentals

[Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)

Dijkstra's Shortest Path First algorithm (SPF algorithm) is an algorithm for finding the shortest paths between nodes in a graph. A ***single*** node as the "source" node and finds shortest paths from the source to all other nodes in the graph, producing a shortest-path tree.

The weights of the graph must be **non-negative**.

Pseuodo-code:

```
function Dijkstra(Graph, source):
    dist[source] ← 0                           // Initialization

    create vertex priority queue Q

    for each vertex v in Graph.Vertices:
        if v ≠ source
            dist[v] ← INFINITY                 // Unknown distance from source to v
            prev[v] ← UNDEFINED                // Predecessor of v

        Q.add_with_priority(v, dist[v])

    while Q is not empty:                      // The main loop
        u ← Q.extract_min()                    // Remove and return best vertex
        for each neighbor v of u:              // only v that are still in Q
            alt ← dist[u] + Graph.Edges(u, v)
            if alt < dist[v]
                dist[v] ← alt
                prev[v] ← u
                Q.decrease_priority(v, alt)

    return dist, prev
```

Note it's very important that in each iteration, we handles **only** neighbor vertices that are still in the queue.

If we are only interested in a shortest path between vertices source and target, we can terminate the search if `u == target`.

[Indexed priority queue](https://algs4.cs.princeton.edu/24pq/IndexMinPQ.java.html), Fibonacci heap or Brodal queue offer optimal implementations for the above 3 min-priority queque operations.

Time complexity: 
\\(\Theta(|V|+|E|\log{|V|})\\)

Dijkstra's algorithm has some similarities with BFS:

|       | BFS  | Dijkstra |
|-------| ------------- | ------------- |
| graph | unweighted  | weighted (non-negative) |
| queue | queue  | priority queue  |
| time complexity | O(V + E) | O(V + Elog(V)) |

A common implementation with simple priority queue is as below:

```java
/**
 * Dijkstra's algorithm.
 * @param n number of vertices. Vertices range from 0 to (n - 1)
 * @param graph graph[i][j] is the edge from vertex i to j
 * @param src source node
 */
public void dijkstra(int n, List<int[]>[] graph, int src) {
    // dist[i]: distance from src to vertex i
    // initializes all elements in the array to inf
    int[] dist = new int[n];
    Arrays.fill(dist, Integer.MAX_VALUE);

    // {vertex, dist[vertex] when enqueued}
    Queue<int[]> q = new PriorityQueue<>((a, b) -> a[1] - b[1]);

    // initializes the queue to contain source only
    int[] curr = {src, dist[src] = 0};
    q.offer(curr);

    while (!q.isEmpty()) {
        // dequeues the best vertex
        curr = q.poll();
        int u = curr[0], du = curr[1];
        
        // skips current if the vertex is visited
        // (and hence already has a shorter distance)
        if (du > dist[u]) {
            continue;
        }

        // checks all neighbors in the graph whether they are still in the queue or not
        // then a vertex can be re-enqueued and thus its time complexity is more
        // compared to traditional Dijkstra (Fibonacci queue version)
        for (int[] next : graph[u]) {
            int v = next[0], duv = next[1];
            int alt = du + duv;
            if (alt < dist[v]) {
                q.offer(new int[]{v, dist[v] = alt});
            }
        }
    }
}
```

See [Minimum Weighted Subgraph With the Required Paths][minimum-weighted-subgraph-with-the-required-paths], [Number of Ways to Arrive at Destination][number-of-ways-to-arrive-at-destination] as examples.

Time Complexity:
\\(\Theta((|V|+|E|)\log{|V|})\\)

To skip visited vertices, we can use an array or set to mark them. See [reachable-nodes-in-subdivided-graph]. 

## Examples

This section demos how to apply Dijkstra's algorithm to solve problems, along with some other steps (e.g. DFS).

[Minimum Weighted Subgraph With the Required Paths][minimum-weighted-subgraph-with-the-required-paths]

```java
public long minimumWeight(int n, int[][] edges, int src1, int src2, int dest) {
    List<long[]>[] graph = new List[n], reverse = new List[n];

    for (int i = 0; i < n; i++) {
        graph[i] = new ArrayList<>();
        reverse[i] = new ArrayList<>();
    }

    for (int[] e : edges) {
        graph[e[0]].add(new long[]{e[1], e[2]});
        reverse[e[1]].add(new long[]{e[0], e[2]});
    }

    long[] weight1 = new long[n], weight2 = new long[n], weightDest = new long[n];
    Arrays.fill(weight1, Long.MAX_VALUE);
    Arrays.fill(weight2, Long.MAX_VALUE);
    Arrays.fill(weightDest, Long.MAX_VALUE);

    // 3 Dijkstra's
    dijkstra(graph, src1, weight1);
    dijkstra(graph, src2, weight2);
    dijkstra(reverse, dest, weightDest);

    // finds min weight
    long min = Long.MAX_VALUE;
    for (int i = 0; i < n; i++) {
        if (weight1[i] < Long.MAX_VALUE &&
            weight2[i] < Long.MAX_VALUE &&
            weightDest[i] < Long.MAX_VALUE) {
            min = Math.min(min, weight1[i] + weight2[i] + weightDest[i]);
        }
    }

    return min == Long.MAX_VALUE ? -1 : min;
}

private void dijkstra(List<long[]>[] graph, int src, long[] weight) {
    Queue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> a[1]));
    long[] curr = {src, weight[src] = 0};
    pq.offer(curr);

    while (!pq.isEmpty()) {
        curr = pq.poll();
        int node = (int)curr[0];
        long w = curr[1];

        if (weight[node] < w || graph[node].isEmpty()) {
            continue;
        }

        for (var e : graph[node]) {
            int neighbor = (int)e[0];
            long neighborW = e[1];
            if (weight[neighbor] > weight[node] + neighborW) {
                pq.offer(new long[]{neighbor, weight[neighbor] = weight[node] + neighborW});
            }
        }
    }
}
```

[Number of Restricted Paths From First to Last Node][number-of-restricted-paths-from-first-to-last-node]

```java
private static final int MOD = (int)1e9 + 7;
private List<int[]>[] graph;
private int[] dist, memo;
private int n;

public int countRestrictedPaths(int n, int[][] edges) {
    this.n = n;
    this.graph = new List[n + 1];
    for (int i = 1; i <= n; i++) {
        graph[i] = new ArrayList<>();
    }

    // u : {v, weight}
    for (int[] e : edges) {
        graph[e[0]].add(new int[]{e[1], e[2]});
        graph[e[1]].add(new int[]{e[0], e[2]});
    }

    this.dist = new int[n + 1];
    Arrays.fill(dist, Integer.MAX_VALUE);

    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
    int[] curr = new int[]{n, dist[n] = 0};
    pq.offer(curr);

    while (!pq.isEmpty()) {
        curr = pq.poll();
        int node = curr[0], d = curr[1];

        if (dist[curr] < d) {
            continue;
        }

        for (int[] next : graph[curr]) {
            int neighbor = next[0], e = next[1];
            int alt = dist[curr] + e;
            if (alt < dist[neighbor]) {
                pq.offer(new int[]{neighbor, dist[neighbor] = alt});
            }
        }
    }

    this.memo = new int[n + 1];
    Arrays.fill(memo, -1);

    return dfs(1);
}

private int dfs(int node) {
    if (node == n) {
        return 1;
    }

    if (memo[node] >= 0) {
        return memo[node];
    }

    int count = 0;
    for (var e : graph[node]) {
        int neighbor = e[0];
        // on a restricted path, dist is monotonically decreasing
        // so there's no cycle
        if (dist[neighbor] < dist[node]) {
            count = (count + dfs(neighbor)) % MOD;
        }
    }

    return memo[node] = count;
}
```

[Minimum Obstacle Removal to Reach Corner][minimum-obstacle-removal-to-reach-corner]

Model the grid as a graph where cells are nodes and edges are between adjacent cells. Edges to cells with obstacles have a cost of 1 and all other edges have a cost of 0.

## Cost function

Cost function is monotonically increasing/decreasing. The traditional cost function is summation of non-negative weights. The following are cost function variations:

**Multiplication of probabilities**

[Path with Maximum Probability][path-with-maximum-probability]

```java
public double maxProbability(int n, int[][] edges, double[] succProb, int start, int end) {
    List<int[]>[] graph = new List[n];
    for (int i = 0; i < n; i++) {
        graph[i] = new ArrayList<>();
    }

    for (int i = 0; i < edges.length; i++) {
        int[] e = edges[i];
        graph[e[0]].add(new int[]{e[1], i});
        graph[e[1]].add(new int[]{e[0], i});
    }

    // Dijkstra
    double[] p = new double[n];

    // max heap
    Queue<Pair<Integer, Double>> pq = new PriorityQueue<>(Comparator.comparingDouble(a -> -a.getValue()));
    Pair<Integer, Double> curr = new Pair<>(start, p[start] = 1);
    pq.offer(curr);

    while (!pq.isEmpty()) {
        curr = pq.poll();
        int u = curr.getKey();
        double pu = curr.getValue();

        if (u == end) {
            break;
        }

        if (pu < p[u]) {
            continue;
        }

        for (int[] next : graph[u]) {
            int neighbor = next[0], index = next[1];
            double alt = p[u] * succProb[index];
            if (alt > p[neighbor]) {
                pq.offer(new Pair<>(neighbor, p[neighbor] = alt));
            }
        }
    }
    return p[end];
}
```

**Summation of absolute difference**

[Path with Minimum Effort][path-with-minimum-effort]

```java
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
{% endraw %}

public int minimumEffortPath(int[][] heights) {
    int m = heights.length, n = heights[0].length;
    int[][] dist = new int[m][n];
    for (int i = 0; i < m; i++) {
        Arrays.fill(dist[i], Integer.MAX_VALUE);
    }

    // {i, j, effort}
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));
    int[] curr = new int[]{0, 0, 0};
    pq.offer(curr);

    while (!pq.isEmpty()) {
        curr = pq.poll();
        int i = curr[0], j = curr[1], e = curr[2];
        if (i == m - 1 && j == n - 1) {
            return e;
        }

        if (e > dist[i][j]) {
            continue;
        }

        for (int[] d : DIRECTIONS) {
            int r = i + d[0], c = j + d[1];
            if (r >= 0 && r < m && c >= 0 && c < n) {
                int alt = Math.max(Math.abs(heights[r][c] - heights[i][j]), e);
                if (alt < dist[r][c]) {
                    pq.offer(new int[]{r, c, dist[r][c] = alt});
                }
            }
        }
    }

    return -1;        
}
```

**Summation of Manhattan distances**

[Campus Bikes II][campus-bikes-ii]

**Maximum of heights**

[Trapping Rain Water II][trapping-rain-water-ii]

```java
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
{% endraw %}
public int trapRainWater(int[][] heightMap) {
    int m = heightMap.length, n = heightMap[0].length;

    // virtual start representing the outside world
    int start = m * n;

    // {row * n + col, weight}
    // the path direction is from i to graph[i]
    List<int[]>[] graph = new List[m * n + 1];
    for (int i = 0; i < graph.length; i++) {
        graph[i] = new ArrayList<>();
    }

    int[] dist = new int[graph.length];
    Arrays.fill(dist, Integer.MAX_VALUE / 2);
    dist[start] = 0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // border cells
            if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                graph[start].add(new int[]{i * n + j, 0});
            }

            for (int[] d : DIRECTIONS) {
                int r = i + d[0], c = j + d[1];
                if (r >= 0 && r < m && c >= 0 && c < n) {
                    graph[i * n + j].add(new int[]{r * n + c, heightMap[i][j]});
                }
            }
        }
    }

    // for each unit cell, finds the min of all paths' weight
    // where a path's weight is defined as the highest height value along the path
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(c -> c[1]));
    int[] curr = new int[]{start, 0};
    pq.offer(curr);

    while (!pq.isEmpty()) {
        curr = pq.poll();
        int u = curr[0], du = curr[1];
        if (du > dist[u]) {
            continue;
        }

        for (int[] next : graph[u]) {
            int v = next[0], duv = next[1];
            int alt = Math.max(du, duv);
            if (alt < dist[v]) {
                pq.offer(new int[]{v, dist[v] = alt});
            }
        }
    }

    int volume = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            volume += Math.max(0, dist[i * n + j] - heightMap[i][j]);
        }
    }
    return volume;
}
```

This problem can be solved by [BFS](../bfs) as well. We can see the intrinsic connection of the two approaches.

**Additional Wait**

[Minimum Time to Visit a Cell In a Grid][minimum-time-to-visit-a-cell-in-a-grid]

```java
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
{% endraw %}
public int minimumTime(int[][] grid) {
    // it's the only condition when the bottom-right cell cannot be visited
    // because at time > 0, you can always go back and forth between previous and current cell
    // until the time is long enough to move on
    if (grid[0][1] > 1 && grid[1][0] > 1) {
        return -1;
    }

    int m = grid.length, n = grid[0].length;
    int[][] dist = new int[m][n];
    for (int[] row : dist) {
        Arrays.fill(row, Integer.MAX_VALUE);
    }

    // {row, col, time}
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));
    pq.offer(new int[]{0, 0, dist[0][0] = 0});

    while (!pq.isEmpty()) {
        int[] cell = pq.poll();
        int row = cell[0], col = cell[1], time = cell[2];

        if (row == m - 1 && col == n - 1) {
            return time;
        }

        if (time > dist[row][col]) {
            continue;
        }

        for (int[] d : DIRECTIONS) {
            int r = row + d[0], c = col + d[1];
            if (r >= 0 && r < m && c >= 0 && c < n) {
                // if the neighbor cell has value > time + 1
                // back-and-forth between previous and currnt cell to let time pass
                // case 1
                //   (grid[r][c] - time) is even
                //   e.g. neighbor = 3, time = 1
                //        then, 2 @ curr, 3 @ prev, 4 @ neighbor
                // case 2
                //   (grid[r][c] - time) is odd
                //   e.g. neighbor = 4, time = 1
                //        then, 2 @ curr, 3 @ prev, 4 @ neighbor
                int wait = ((grid[r][c] - time) % 2) ^ 1;
                int alt = Math.max(grid[r][c] + wait, time + 1);
                if (alt < dist[r][c]) {
                    pq.offer(new int[]{r, c, dist[r][c] = alt});
                }
            }
        }
    }
    return -1;
}
```

## Composite Vertex

[The Maze III][the-maze-iii]

```java
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
{% endraw %}
private static final char[] INSTRUCTIONS = {'d', 'r', 'u', 'l'};

class Point implements Comparable<Point> {
    int row, col, distance;
    String instruction;

    Point(int row, int col, int distance, String instruction) {
        this.row = row;
        this.col = col;
        this.distance = distance;
        this.instruction = instruction;
    }

    @Override
    public int compareTo(Point o) {
        return this.distance == o.distance ? this.instruction.compareTo(o.instruction) : this.distance - o.distance;
    }
}

public String findShortestWay(int[][] maze, int[] ball, int[] hole) {
    int m = maze.length, n = maze[0].length;
    boolean[][] visited = new boolean[m][n];

    Queue<Point> pq = new PriorityQueue<>();
    Point p = new Point(ball[0], ball[1], 0, "");
    pq.offer(p);

    while (!pq.isEmpty()) {
        p = pq.poll();
        if (p.row == hole[0] && p.col == hole[1]) {
            return p.instruction;
        }

        visited[p.row][p.col] = true;

        for (int i = 0; i < DIRECTIONS.length; i++) {
            int[] d = DIRECTIONS[i];
            int r = p.row + d[0], c = p.col + d[1];
            int distance = p.distance;

            // keeps rolling until hitting a wall or the hole
            while (r >= 0 && r < m && c >= 0 && c < n && maze[r][c] == 0) {
                distance++;
                if (r == hole[0] && c == hole[1]) {
                    r += d[0];
                    c += d[1];
                    break;
                }
                r += d[0];
                c += d[1];
            }
            r -= d[0];
            c -= d[1];

            if (!visited[r][c]) {
                pq.offer(new Point(r, c, distance, p.instruction + INSTRUCTIONS[i]));
            }
        }
    }
    return "impossible";
}
```

[Campus Bikes II][campus-bikes-ii]

```java
public int assignBikes(int[][] workers, int[][] bikes) {
    // {worker, bike mask, distance}
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));
    pq.offer(new int[]{0, 0, 0});

    Set<String> visited = new HashSet<>();
    while (!pq.isEmpty()) {
        int[] node = pq.poll();
        int worker = node[0], mask = node[1], d = node[2];

        if (visited.add(worker + "#" + mask)) {
            if (worker == workers.length) {
                return d;
            }

            for (int i = 0; i < bikes.length; i++) {
                // i-th bike is available
                if ((mask & (1 << i)) == 0) {
                    pq.offer(new int[]{worker + 1, mask | (1 << i), d + distance(workers[worker], bikes[i])});
                }
            }
        }

    }
    return -1;
}

private int distance(int[] p1, int[] p2) {
    return Math.abs(p1[0] - p2[0]) + Math.abs(p1[1] - p2[1]);
}
```

In this problem, each vertex is composite, i.e. a state that combines multiple variables. Specifically, vertices are constructed layer by layer from `workers[0]` to `workers[n - 1]`, and in the i-th layer, a vertex stands for a certain assignment of bikes to `workers[0...i]`. This solution is very similar to BFS - the only difference is we use a priority queue to find the min dist quickly in each layer.

## More Than One Shortest Path

If there are multiple shortest paths from `source` to `target`, we can track the count with another auxiliary array. See the following problem:

[Number of Ways to Arrive at Destination][number-of-ways-to-arrive-at-destination]

```java
private static final int MOD = (int)1e9 + 7;

public int countPaths(int n, int[][] roads) {
    List<int[]>[] graph = new List[n];
    for (int i = 0; i < n; i++) {
        graph[i] = new ArrayList<>();
    }

    for (int[] r : roads) {
        graph[r[0]].add(new int[]{r[1], r[2]});
        graph[r[1]].add(new int[]{r[0], r[2]});
    }

    // time[i]: shortest amount of time from 0 to i so far
    int[] time = new int[n], ways = new int[n];
    Arrays.fill(time, Integer.MAX_VALUE);

    // {city, time from 0 to city at the time of enqueue}
    Queue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
    int[] curr = {0, time[0] = 0};
    pq.offer(curr);

    ways[0] = 1;
    while (!pq.isEmpty()) {
        curr = pq.poll();
        int city = curr[0], currentTime = curr[1];
        if (currentTime > time[city]) {
            continue;
        }

        for (int[] next : graph[city]) {
            int neighbor = next[0], betweenTime = next[1];
            int alt = currentTime + betweenTime;
            if (time[neighbor] > alt) {
                ways[neighbor] = ways[city];
                pq.offer(new int[]{neighbor, time[neighbor] = alt});
            } else if (time[neighbor] == currentTime + betweenTime) {
                ways[neighbor] = (ways[neighbor] + ways[city]) % MOD;
            }
        }
    }
    return ways[n - 1];
}
```

## Constrained Dijkstra's

Given upper limit of weight sum, find/count all paths.

[Reachable Nodes In Subdivided Graph][reachable-nodes-in-subdivided-graph]

```java
public int reachableNodes(int[][] edges, int maxMoves, int n) {
    // graph[i][j]: count between the edge [i, j]
    int[][] graph = new int[n][n];
    for (int[] g : graph) {
        Arrays.fill(g, -1);
    }
    for (int[] e : edges) {
        graph[e[0]][e[1]] = e[2];
        graph[e[1]][e[0]] = e[2];
    }

    // {node, available moves}
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> -a[1]));
    int[] curr = {0, maxMoves};
    pq.offer(curr);

    boolean[] visited = new boolean[n];
    int count = 0;
    while (!pq.isEmpty()) {
        curr = pq.poll();
        int node = curr[0], availableMoves = curr[1];

        if (visited[node]) {
            continue;
        }

        visited[node] = true;
        count++;

        for (int neighbor = 0; neighbor < n; neighbor++) {
            if (graph[node][neighbor] >= 0) {
                if (availableMoves > graph[node][neighbor] && !visited[neighbor]) {
                    pq.offer(new int[]{neighbor, availableMoves - graph[node][neighbor] - 1});
                }

                // number of nodes that are reachable on the edge (node, neighbor]
                int reach = Math.min(availableMoves, graph[node][neighbor]);
                count += reach;

                // the remaining new nodes could be visited from the other direction
                graph[neighbor][node] -= reach;
            }
        }
    }
    return count;
}
```

Given a contraint on sum of vertex values, minimize the sum of path values. In the problem below, we view the value of each vertex as 1, so the constraint `stops` can be regarded as the sum of the vertex values.

[Cheapest Flights Within K Stops][cheapest-flights-within-k-stops]

```java
public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
    // builds graph
    int[][] graph = new int[n][n];
    for (int[] f : flights) {
        graph[f[0]][f[1]] = f[2];
    }

    int[] prices = new int[n], stops = new int[n];
    Arrays.fill(prices, Integer.MAX_VALUE);
    Arrays.fill(stops, Integer.MAX_VALUE);

    // {city, prices, stops}
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
    int[] curr = {src, prices[src] = 0, stops[src] = 0};
    pq.offer(curr);

    while (!pq.isEmpty()) {
        curr = pq.poll();
        int city = curr[0], p = curr[1], s = curr[2];

        if (city == dst) {
            return p;
        }

        for (int neighbor = 0; neighbor < n; neighbor++) {
            if (graph[city][neighbor] > 0 && s <= k) {
                int alt = p + graph[city][neighbor];
                if (alt < prices[neighbor]) {
                    pq.offer(new int[]{neighbor, prices[neighbor] = alt, stops[neighbor] = s + 1});
                } else if (s + 1 < stops[neighbor]) {
                    // although prices is not optimal at the moment, the number of stops is less than stops[neighbor]
                    // which means this path has the potential to be the optimal solution in later steps
                    //
                    // do not set stops[neighbor] = s + 1 here!
                    // stops[neighbor] is updated only if prices[neighbor] is updated
                    // it doesn't reflect the least stops of all paths so far
                    // iwo, currently it's possible that there exists a node where the stops are less than stops[node]
                    //
                    // stops[node] is a loose constraint
                    pq.offer(new int[]{neighbor, alt, s + 1});
                }
            }
        }
    }

    return prices[dst] == Integer.MAX_VALUE ? -1 : prices[dst];
}
```

For example, `n = 9, flights = [[0,1,1],[1,2,1],[2,3,1],[3,7,1],[0,4,3],[4,5,3],[5,7,3],[0,6,5],[6,7,100],[7,8,1]], src = 0, dst = 8, k = 3`.

```mermaid
graph TD
    0((0)) -- 1 --> 1((1))
    1 -- 1 --> 2((2))
    2 -- 1 --> 3((3))
    3 -- 1 --> 7((7))
    0 -- 3 --> 4((4))
    4 -- 3 --> 5((5))
    5 -- 3 --> 7
    0 -- 5 --> 6((6))
    6 -- 100 --> 7
    7 -- 1 --> 8((8))
```

If in `else if (s + 1 < stops[neighbor])`, the statement is `pq.offer(new int[]{neighbor, alt, stops[neighbor] = s + 1});`, then the path `0 -> 6 -> 7` would set `stops[7] = 2` and exclude the correct path `0 -> 4 -> 5 -> 7` (because its `stops == 3`).

Given a contraint on sum of path values, minimize the sum of vertex values.

[Minimum Cost to Reach Destination in Time][minimum-cost-to-reach-destination-in-time]

```java
public int minCost(int maxTime, int[][] edges, int[] passingFees) {
    int n = passingFees.length;
    List<int[]>[] graph = new List[n];
    for (int i = 0; i < n; i++) {
        graph[i] = new ArrayList<>();
    }
    for (int[] e : edges) {
        graph[e[0]].add(new int[]{e[1], e[2]});
        graph[e[1]].add(new int[]{e[0], e[2]});
    }

    int[] times = new int[n];
    Arrays.fill(times, maxTime + 1);

    // {city, cost, time}
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
    int[] curr = {0, passingFees[0], times[0] = 0};
    pq.offer(curr);

    while (!pq.isEmpty()) {
        curr = pq.poll();
        int city = curr[0], cost = curr[1], time = curr[2];

        if (city == n - 1) {
            return cost;
        }

        for (int[] next : graph[city]) {
            int neighbor = next[0], altTime = time + next[1], altCost = cost + passingFees[neighbor];

            if (altTime < times[neighbor]) {
                pq.offer(new int[]{neighbor, altCost, times[neighbor] = altTime});
            }
        }
    }

    return -1;
}
```

## Dijkstra's + DP

[Minimum Cost to Reach City With Discounts][minimum-cost-to-reach-city-with-discounts]

```java
public int minimumCost(int n, int[][] highways, int discounts) {
    // buils graph
    List<int[]>[] graph = new List[n];
    for (int i = 0; i < n; i++) {
        graph[i] = new ArrayList<>();
    }
    for (int[] h : highways) {
        graph[h[0]].add(new int[]{h[1], h[2]});
        graph[h[1]].add(new int[]{h[0], h[2]});
    }

    // Dijkstra + DP
    // visited[i][j]: min cost of visited city i with j discounts
    int[][] visited = new int[n][discounts + 1];
    for (int i = 0; i < n; i++) {
        Arrays.fill(visited[i], Integer.MAX_VALUE);
    }

    // {city, usedDiscount, cost}
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));
    int[] curr = {0, 0, visited[0][0] = 0};
    pq.offer(curr);

    while (!pq.isEmpty()) {
        curr = pq.poll();
        int city = curr[0], d = curr[1], cost = curr[2];

        if (visited[city][d] < cost) {
            continue;
        }

        if (city == n - 1) {
            return cost;
        }

        for (int[] next : graph[city]) {
            int neighbor = next[0], toll = next[1];
            // doesn't use discount
            int alt = cost + toll;
            if (alt < visited[neighbor][d]) {
                pq.offer(new int[]{neighbor, d, visited[neighbor][d] = alt});
            }

            // uses discount
            alt = cost + toll / 2;
            if (d < discounts && alt < visited[neighbor][d + 1]) {
                pq.offer(new int[]{neighbor, d + 1, visited[neighbor][d + 1] = alt});
            }
        }
    }
    return -1;
}
```

# Bellman-Ford Algorithm

[Bellman-Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm) is an algorithm that computes shortest paths from a single source vertex to all of the other vertices in a weighted digraph. It is capable of handling graphs in which some of the edge weights are negative numbers.

[Cheapest Flights Within K Stops][cheapest-flights-within-k-stops]

```java
public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
    int[] prices = new int[n], tmp = new int[n];
    Arrays.fill(prices, Integer.MAX_VALUE);

    prices[src] = 0;
    for(int i = 0; i <= k; i++) {
        System.arraycopy(prices, 0, tmp, 0, n);
        for (int[] f: flights) {
            int curr = f[0], next = f[1], price = f[2];
            if (tmp[curr] < Integer.MAX_VALUE) {
                prices[next] = Math.min(prices[next], tmp[curr] + price);
            }
        }
    }
    return prices[dst] == Integer.MAX_VALUE ? -1 : prices[dst];
}
```

# Floyd-Warshall Algorithm

[Floyd-Warshall algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm) is an algorithm for finding shortest paths in a directed weighted graph with positive or negative edge weights (but with no negative cycles). A single execution of the algorithm will find the lengths (summed weights) of shortest paths between ***all*** pairs of vertices.

Pseudocode:

```
let dist be a |V| × |V| array of minimum distances initialized to ∞ (infinity)
for each edge (u, v) do
    dist[u][v] ← w(u, v)  // The weight of the edge (u, v)
for each vertex v do
    dist[v][v] ← 0
for k from 1 to |V|
    for i from 1 to |V|
        for j from 1 to |V|
            if dist[i][j] > dist[i][k] + dist[k][j]
                dist[i][j] ← dist[i][k] + dist[k][j]
            end if
```

Time complexity: \\(\Theta(V^3)\\)

[Count Subtrees With Max Distance Between Cities][count-subtrees-with-max-distance-between-cities]

```java
public int[] countSubgraphsForEachDiameter(int n, int[][] edges) {
    // Floyd-Warshall
    int[][] tree = new int[n][n];
    for (int[] t : tree) {
        Arrays.fill(t, n);
    }

    for (int[] e : edges) {
        int i = e[0] - 1, j = e[1] - 1;
        tree[i][j] = tree[j][i] = 1;
    }

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                tree[i][j] = Math.min(tree[i][j], tree[i][k] + tree[k][j]);
            }
        }
    }

    int[] result = new int[n - 1];
    // bit mask
    for (int state = 1; state < (1 << n); state++) {
        int k = Integer.bitCount(state);
        // number of edges in state
        int e = 0;
        // shortest path of all pairs in state
        int d = 0;
        for (int i = 0; i < n; i++) {
            if (isBitSet(state, i)) {
                for (int j = i + 1; j < n; j++) {
                    if (isBitSet(state, j)) {
                        e += tree[i][j] == 1 ? 1 : 0;
                        d = Math.max(d, tree[i][j]);
                    }
                }
            }
        }

        // e == k - 1 means state is a subtree
        if (e == k - 1 && d > 0) {
            result[d - 1]++;
        }
    }

    return result;
}

private boolean isBitSet(int i, int b) {
    return (i & (1 << b)) != 0;
}
```

[Course Schedule IV][course-schedule-iv]

```java
// Floyd–Warshall Algorithm
public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
    boolean[][] graph = new boolean[numCourses][numCourses];
    for (int[] p : prerequisites) {
        graph[p[0]][p[1]] = true;
    }

    for (int k = 0; k < numCourses; k++) {
        for (int i = 0; i < numCourses; i++) {
            for (int j = 0; j < numCourses; j++) {
                graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
            }
        }
    }

    List<Boolean> answer = new ArrayList<>();
    for (int[] q : queries) {
        answer.add(graph[q[0]][q[1]]);
    }
    return answer;
}
```

# Paint and Expansion

[Shortest Bridge][shortest-bridge]

```java
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
{% endraw %}
public int shortestBridge(int[][] grid) {
    int n = grid.length;

    // paints one island to 2
    for (int i = 0; i < n; i++) {
        int j = 0;
        while (j < n) {
            if (paint(grid, i, j, 2)) {
                break;
            }
            j++;
        }
        if (j < n) {
            break;
        }
    }

    int color = 2;
    while (true) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == color) {
                    for (int[] d : DIRECTIONS) {
                        if (expand(grid, i + d[0], j + d[1], color)) {
                            return color - 2;
                        }
                    }
                }
            }
        }
        color++;
    }
}

private boolean paint(int[][] grid, int i, int j, int color) {
    if (i < 0 || j < 0 || i == grid.length || j == grid.length || grid[i][j] != 1) {
        return false;
    }

    grid[i][j] = color;
    for (int[] d : DIRECTIONS) {
        paint(grid, i + d[0], j + d[1], color);
    }
    return true;
}

private boolean expand(int[][] grid, int i, int j, int color) {
    if (i < 0 || j < 0 || i == grid.length || j == grid.length) {
        return false;
    }

    if (grid[i][j] == 0) {
        grid[i][j] = color + 1;
    }

    // returns true if it reaches the other island
    return grid[i][j] == 1;
}
```

[Minimum Number of Days to Disconnect Island][minimum-number-of-days-to-disconnect-island]

```java
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
{% endraw %}
private int[][] grid;
private int m, n;

public int minDays(int[][] grid) {
    this.grid = grid;
    this.m = grid.length;
    this.n = grid[0].length;

    // checks if there's only one island
    int color = 1;
    int count = countIslands(color);
    if (count != 1) {
        return 0;
    }
    color++;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // backtracking
            if (grid[i][j] == color) {
                grid[i][j] = 0;
                if (countIslands(color) != 1) {
                    return 1;
                }

                // painting bumps the color
                color++;
                grid[i][j] = color;
            }
        }
    }

    // any island can be disconnected in at most 2 days
    return 2;
}

// counts islands with color, and paints these islands to (color + 1)
private int countIslands(int color) {
    int count = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == color) {
                paint(i, j, color);
                count++;

                // if there are more than one land, returns directly
                if (count > 1) {
                    return count;
                }
            }
        }
    }
    return count;
}

// paints island of (i, j) to (color + 1)
private void paint(int i, int j, int color) {
    if (i < 0 || i == m || j < 0 || j == n || grid[i][j] == 0 || grid[i][j] == color + 1) {
        return;
    }

    // marks the cell as visited by incrementing it by 1
    grid[i][j] = color + 1;

    for (int[] d : DIRECTIONS) {
        paint(i + d[0], j + d[1], color);
    }
}
```

[Making A Large Island][making-a-large-island]

```java
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
{% endraw %}
public int largestIsland(int[][] grid) {
    // color : island size
    Map<Integer, Integer> map = new HashMap<>();
    // images the grid is surrounded by water (color 0)
    map.put(0, 0);

    // paints each island with a unique color
    int n = grid.length;
    int color = 2;  // color starts at 2
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                map.put(color, paint(grid, i, j, color));
                color++;
            }
        }
    }

    // initially, if all cells are 1, they will be painted to 2
    // and the size of island 2 will be the final result,
    // since at most one operation can be performed.
    // otherwise, we initialize max to 0
    int max = map.getOrDefault(2, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // operation
            if (grid[i][j] == 0) {
                // set of islands (including water) surrounding the current 0
                Set<Integer> set = new HashSet<>();
                for (int[] d : DIRECTIONS) {
                    int r = i + d[0], c = j + d[1];
                    if (r < 0 || r == n || c < 0 || c == n) {
                        set.add(0);
                    } else {
                        set.add(grid[r][c]);
                    }
                }

                int size = 1;
                for (int island : set) {
                    size += map.get(island);
                }
                max = Math.max(max, size);
            }
        }
    }
    return max;
}

private int paint(int[][] grid, int i, int j, int color) {
    // grid[i][j] != 1 means it's either water or another island
    if (i < 0 || j < 0 || i == grid.length || j == grid.length || grid[i][j] != 1) {
        return 0;
    }

    grid[i][j] = color;

    int size = 1;
    for (int[] d : DIRECTIONS) {
        size += paint(grid, i + d[0], j + d[1], color);
    }
    return size;
}
```

[Modify Graph Edge Weights][modify-graph-edge-weights]

```java
public int[][] modifiedGraphEdges(int n, int[][] edges, int source, int destination, int target) {
    // {node, [neighbor, weight]}
    List<int[]>[] g = new List[n];
    for (int i = 0; i < n; i++) {
        g[i] = new ArrayList<>();
    }

    List<int[]> list = new ArrayList<>();
    // (node A, node B) : weight between A and B
    // where node A < node B
    int[][] weights = new int[n][n];
    for (int[] e : edges) {
        g[e[0]].add(new int[]{e[1], e[2]});
        g[e[1]].add(new int[]{e[0], e[2]});

        int a = Math.min(e[0], e[1]), b = Math.max(e[0], e[1]);
        list.add(new int[]{a, b, weights[a][b] = e[2]});
    }

    // treats negative edges as disconnected
    // calculates the shortest distances from destionation to all nodes
    // if the shortest distance between source and destionation is alreay less than target
    // modifying the negative edges will possible make the distance even shorter, so no solution
    int[][] reverse = dijkstra(n, g, destination, true);
    if (reverse[0][source] < target) {
        return new int[0][];
    }

    // treats negative edges as weight 1
    // the shortest distance is min of assigning positive values in [1, 2e9]
    // the shortest distance must be no greater than the target
    int[][] res = dijkstra(n, g, source, false);
    if (res[0][destination] > target) {
        return new int[0][];
    }

    // when all negative edges are 1
    // constructs the shortest path from source to destionation
    List<Integer> path = new LinkedList<>();
    path.add(destination);
    while (path.get(0) != source) {
        path.add(0, res[1][path.get(0)]);
    }

    // adjusts each segment of the path if the original weight is negative
    int d = 0, m = path.size();
    for (int i = 0; i < m - 1; i++) {
        int u = path.get(i), v = path.get(i + 1);
        int a = Math.min(u, v), b = Math.max(u, v);
        if (weights[a][b] == -1) {
            // if there exists a postive weight value that makes the path total weight equal to target
            // we can skip the remaining segments on the path
            if ((weights[a][b] = Math.max(target - reverse[0][v] - d, 1)) > 1) {
                break;
            }
        }
        d += weights[a][b];
    }

    for (int[] e : list) {
        // there's already a shortest path with targeted weight
        // so the remaining negative edges can be assigned arbitrarily
        if ((e[2] = weights[e[0]][e[1]]) == -1) {
            e[2] = (int)2e9;
        }
    }
    return list.toArray(new int[0][]);
}

private int[][] dijkstra(int n, List<int[]>[] graph, int src, boolean skipNegative) {
    // dist[i]: distance from src to vertex i
    // initializes all elements in the array to inf
    int[] dist = new int[n], parents = new int[n];
    Arrays.fill(dist, Integer.MAX_VALUE);

    // {vertex, dist[vertex] when enqueued}
    Queue<int[]> q = new PriorityQueue<>((a, b) -> a[1] - b[1]);

    // initializes the queue to contain source only
    int[] curr = {src, dist[src] = 0};
    q.offer(curr);

    while (!q.isEmpty()) {
        // dequeues the best vertex
        curr = q.poll();
        int u = curr[0], du = curr[1];

        // skips current if the vertex is visited
        // (and hence already has a more optimal distance)
        if (du > dist[u]) {
            continue;
        }

        // checks all neighbors in the graph whether they are still in the queue or not
        // then a vertex can be re-enqueued and thus its time complexity is more
        // compared to traditional Dijkstra (Fibonacci queue version)
        for (int[] next : graph[u]) {
            int v = next[0], duv = next[1];
            if (duv == -1) {
                // if skip negative, negative edges are regarded as disconnected
                if (skipNegative) {
                    continue;
                }
                duv = 1;
            }

            int alt = du + duv;
            if (alt < dist[v]) {
                parents[v] = u;
                q.offer(new int[]{v, dist[v] = alt});
            }
        }
    }
    return new int[][]{dist, parents};
}
```

[campus-bikes-ii]: https://leetcode.com/problems/campus-bikes-ii/
[count-subtrees-with-max-distance-between-cities]: https://leetcode.com/problems/count-subtrees-with-max-distance-between-cities/
[course-schedule-iv]: https://leetcode.com/problems/course-schedule-iv/
[cheapest-flights-within-k-stops]: https://leetcode.com/problems/cheapest-flights-within-k-stops/
[making-a-large-island]: https://leetcode.com/problems/making-a-large-island/
[minimum-cost-to-reach-city-with-discounts]: https://leetcode.com/problems/minimum-cost-to-reach-city-with-discounts/
[minimum-cost-to-reach-destination-in-time]: https://leetcode.com/problems/minimum-cost-to-reach-destination-in-time/
[minimum-number-of-days-to-disconnect-island]: https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/
[minimum-obstacle-removal-to-reach-corner]: https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/
[minimum-time-to-visit-a-cell-in-a-grid]: https://leetcode.com/problems/minimum-time-to-visit-a-cell-in-a-grid/
[minimum-weighted-subgraph-with-the-required-paths]: https://leetcode.com/problems/minimum-weighted-subgraph-with-the-required-paths/
[modify-graph-edge-weights]: https://leetcode.com/problems/modify-graph-edge-weights/
[number-of-restricted-paths-from-first-to-last-node]: https://leetcode.com/problems/number-of-restricted-paths-from-first-to-last-node/
[number-of-ways-to-arrive-at-destination]: https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/
[path-with-maximum-minimum-value]: https://leetcode.com/problems/path-with-maximum-minimum-value/
[path-with-maximum-probability]: https://leetcode.com/problems/path-with-maximum-probability/
[path-with-minimum-effort]: https://leetcode.com/problems/path-with-minimum-effort/
[reachable-nodes-in-subdivided-graph]: https://leetcode.com/problems/reachable-nodes-in-subdivided-graph/
[shortest-bridge]: https://leetcode.com/problems/shortest-bridge/
[the-maze-iii]: https://leetcode.com/problems/the-maze-iii/
[trapping-rain-water-ii]: https://leetcode.com/problems/trapping-rain-water-ii/
