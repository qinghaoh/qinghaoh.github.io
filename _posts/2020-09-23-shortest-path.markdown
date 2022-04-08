---
layout: post
title:  "Shortest Path"
tags: graph
usemathjax: true
---
# Dijkstra

[Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)

Dijkstra's Shortest Path First algorithm (SPF algorithm) is an algorithm for finding the shortest paths between nodes in a graph. A ***single*** node as the "source" node and finds shortest paths from the source to all other nodes in the graph, producing a shortest-path tree.

```
function Dijkstra(Graph, source):

    create vertex set Q

    for each vertex v in Graph:
        dist[v] ← INFINITY
        prev[v] ← UNDEFINED
        add v to Q
    dist[source] ← 0

    while Q is not empty:
        u ← vertex in Q with min dist[u]

        remove u from Q

        for each neighbor v of u:           // only v that are still in Q
            alt ← dist[u] + length(u, v)
            if alt < dist[v]:
                dist[v] ← alt
                prev[v] ← u

    return dist[], prev[]
```

Comparison with BFS:

|       | BFS  | Dijkstra |
|-------| ------------- | ------------- |
| graph | unweighted  | weighted (non-negative) |
| queue | queue  | priority queue  |
| time complexity | O(V + E) | O(V + Elog(V)) |

[Number of Ways to Arrive at Destination][number-of-ways-to-arrive-at-destination]

{% highlight java %}
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
    time[0] = 0;
    ways[0] = 1;

    // {city, time from 0 to city at the time of enqueue}
    Queue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
    int[] node = {0, 0};
    pq.offer(node);

    while (!pq.isEmpty()) {
        node = pq.poll();
        int city = node[0], currentTime = node[1];
        if (currentTime > time[city]) {
            continue;
        }

        for (int[] next : graph[city]) {
            int neighbor = next[0], betweenTime = next[1];
            if (time[neighbor] > currentTime + betweenTime) {
                time[neighbor] = currentTime + betweenTime;
                ways[neighbor] = ways[city];
                pq.offer(new int[]{neighbor, time[neighbor]});
            } else if (time[neighbor] == currentTime + betweenTime) {
                ways[neighbor] = (ways[neighbor] + ways[city]) % MOD;
            }
        }
    }
    return ways[n - 1];
}
{% endhighlight %}

[Path with Maximum Probability][path-with-maximum-probability]

{% highlight java %}
public double maxProbability(int n, int[][] edges, double[] succProb, int start, int end) {
    // builds graph
    Map<Integer, List<int[]>> graph = new HashMap<>();  // node : (neighbor : prob index)
    for (int i = 0; i < edges.length; i++) {
        int[] edge = edges[i];
        graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(new int[]{edge[1], i});
        graph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(new int[]{edge[0], i});
    }

    // Dijkstra
    double[] p = new double[n];
    p[start] = 1;

    // max heap
    Queue<Integer> pq = new PriorityQueue<>(Comparator.comparingDouble(i -> -p[i]));
    int node = start;
    pq.offer(node);

    while (!pq.isEmpty() && node != end) {
        node = pq.poll();

        if (!graph.containsKey(node)) {
            continue;
        }

        for (int[] pair : graph.get(node)) {
            int neighbor = pair[0], index = pair[1];
            if (p[node] * succProb[index] > p[neighbor]) {
                p[neighbor] = p[node] * succProb[index];
                pq.offer(neighbor);
            }
        }
    }

    return p[end];
}
{% endhighlight %}

DFS will underflow or LTE.

[Number of Restricted Paths From First to Last Node][number-of-restricted-paths-from-first-to-last-node]

{% highlight java %}
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
    dist[n] = 0;

    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
    int[] node = new int[]{n, 0};
    pq.offer(node);

    while (!pq.isEmpty()) {
        node = pq.poll();

        // there's a node in the queue that's processed and more preferred
        // so this node can be discarded
        if (dist[node[0]] < node[1]) {
            continue;
        }

        for (var e : graph[node[0]]) {
            int neighbor = e[0];
            if (dist[node[0]] + e[1] < dist[neighbor]) {
                dist[neighbor] = dist[node[0]] + e[1];
                pq.offer(new int[]{neighbor, dist[neighbor]});
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
        // on the restricted path, dist is monotonically increasing
        // so there's no cycle
        if (dist[neighbor] < dist[node]) {
            count = (count + dfs(neighbor)) % MOD;
        }
    }

    return memo[node] = count;
}
{% endhighlight %}

[The Maze III][the-maze-iii]

{% highlight java %}
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
    pq.offer(new Point(ball[0], ball[1], 0, ""));

    while (!pq.isEmpty()) {
        Point p = pq.poll();
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
{% endhighlight %}

[Minimum Weighted Subgraph With the Required Paths][minimum-weighted-subgraph-with-the-required-paths]

{% highlight java %}
public long minimumWeight(int n, int[][] edges, int src1, int src2, int dest) {
    List<long[]>[] graph = new List[n], reverse = new List[n];

    for (int i = 0; i < n; i++) {
        graph[i] = new ArrayList<>();
        reverse[i] = new ArrayList<>();
    }

    for (int[] e : edges) {
        long w = (long) e[2];

        graph[e[0]].add(new long[]{e[1], w});
        reverse[e[1]].add(new long[]{e[0], w});
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

    pq.offer(new long[]{src, weight[src] = 0});

    while (!pq.isEmpty()) {
        long[] curr = pq.poll();
        int node = (int)curr[0];
        long w = curr[1];

        if (weight[node] < w || graph[node].isEmpty()) {
            continue;
        }

        for (var e : graph[node]) {
            int neighbor = (int)e[0];
            long neighborW = e[1];
            if (weight[neighbor] > weight[node] + neighborW) {
                weight[neighbor] = weight[node] + neighborW;
                pq.offer(new long[]{neighbor, weight[neighbor]});
            }
        }
    }
}
{% endhighlight %}

## Multiple Constraints

[Cheapest Flights Within K Stops][cheapest-flights-within-k-stops]

{% highlight java %}
public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
    // buils graph
    int[][] graph = new int[n][n];
    for (int[] f : flights) {
        graph[f[0]][f[1]] = f[2];
    }

    // Dijkstra
    // the stops of visited node
    int[] visited = new int[n];
    Arrays.fill(visited, Integer.MAX_VALUE);

    int[] node = {src, 0, 0};  // city, price, stops
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
    pq.offer(node);

    while (!pq.isEmpty()) {
        node = pq.poll();
        int city = node[0], cost = node[1], stops = node[2];
        visited[city] = stops;

        if (city == dst) {
            return cost;
        }

        if (stops <= k) {
            for (int neighbor = 0; neighbor < n; neighbor++) {
                // visited nodes should have more stops than current
                // otherwise, that visited node is a better candidate of the current one
                if (graph[city][neighbor] > 0 && visited[neighbor] > stops) {
                    pq.offer(new int[]{neighbor, cost + graph[city][neighbor], stops + 1});
                }
            }
        }
    }
    return -1;
}
{% endhighlight %}

[Minimum Cost to Reach City With Discounts][minimum-cost-to-reach-city-with-discounts]

{% highlight java %}
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

    // Dijkstra
    // visited[i][j]: min cost of visited city i with j discounts
    int[][] visited = new int[n][discounts + 1];
    for (int i = 0; i < n; i++) {
        Arrays.fill(visited[i], Integer.MAX_VALUE);
    }
    visited[0][0] = 0;

    int[] node = {0, 0, 0};  // city, cost, discount
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
    pq.offer(node);

    while (!pq.isEmpty()) {
        node = pq.poll();
        int city = node[0], cost = node[1], discount = node[2];

        if (city == n - 1) {
            return cost;
        }

        for (int[] next : graph[city]) {
            int neighbor = next[0], weight = next[1];
            // doesn't use discount
            if (cost + weight < visited[neighbor][discount]) {
                pq.offer(new int[]{neighbor, visited[neighbor][discount] = cost + weight, discount});
            }

            // uses discount
            if (discount < discounts && cost + weight / 2 < visited[neighbor][discount + 1]) {
                pq.offer(new int[]{neighbor, visited[neighbor][discount + 1] = cost + weight / 2, discount + 1});
            }
        }
    }
    return -1;
}
{% endhighlight %}

[Minimum Cost to Reach Destination in Time][minimum-cost-to-reach-destination-in-time]

{% highlight java %}
public int minCost(int maxTime, int[][] edges, int[] passingFees) {
    // {city : {neighbor, time}}
    Map<Integer, List<int[]>> graph = new HashMap<>();
    for (int[] e : edges) {
        graph.computeIfAbsent(e[0], k -> new ArrayList<>()).add(new int[]{e[1], e[2]});
        graph.computeIfAbsent(e[1], k -> new ArrayList<>()).add(new int[]{e[0], e[2]});
    }

    int n = passingFees.length;
    int[] costs = new int[n], times = new int[n];
    Arrays.fill(costs, Integer.MAX_VALUE);
    costs[0] = passingFees[0];

    Arrays.fill(times, Integer.MAX_VALUE);
    times[0] = 0;

    // {city, cost, time}
    Queue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] == b[1] ? a[2] - b[2] : a[1] - b[1]);
    pq.add(new int[]{0, costs[0], times[0]});

    while (!pq.isEmpty()) {
        int[] node = pq.poll();
        int city = node[0], cost = node[1], time = node[2];

        if (city == n - 1) {
            return cost;
        }

        for (int[] neighborNode : graph.get(city)) {
            int neighbor = neighborNode[0];
            int neighborCost = passingFees[neighbor], neighborTime = neighborNode[1];

            if (time + neighborTime <= maxTime) {
                // if cost will decrease or time will decrease
                if (cost + neighborCost < costs[neighbor] || time + neighborTime < times[neighbor]) {
                    costs[neighbor] = Math.min(costs[neighbor], cost + neighborCost);
                    pq.offer(new int[]{neighbor, cost + neighborCost, times[neighbor] = time + neighborTime});
                }
            }
        }
    }

    return costs[n - 1] == Integer.MAX_VALUE ? -1 : costs[n - 1];
}
{% endhighlight %}

## Variations

Cost function is monotonically increasing/decreasing.

[Path with Minimum Effort][path-with-minimum-effort]

{% highlight java %}
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
{% endraw %}

public int minimumEffortPath(int[][] heights) {
    int m = heights.length, n = heights[0].length;
    Set<Integer> visited = new HashSet<>();

    // i, j, effort
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));
    pq.offer(new int[]{0, 0, 0});

    while (!pq.isEmpty()) {
        int[] node = pq.poll();
        int i = node[0], j = node[1], e = node[2];
        if (i == m - 1 && j == n - 1) {
            return e;
        }

        if (visited.add(i * n + j)) {
            for (int[] d : DIRECTIONS) {
                int r = i + d[0], c = j + d[1];
                if (r >= 0 && r < m && c >= 0 && c < n && !visited.contains(r * n + c)) {
                    pq.offer(new int[]{r, c, Math.max(Math.abs(heights[r][c] - heights[i][j]), e)});
                }
            }
        }
    }
    return -1;
}
{% endhighlight %}

[Path With Maximum Minimum Value][path-with-maximum-minimum-value]

[Swim in Rising Water][swim-in-rising-water]

* Dijkstra's Algorithm
* BinarySearch + BFS/DFS
* Union-Find

[Campus Bikes II][campus-bikes-ii]

{% highlight java %}
public int assignBikes(int[][] workers, int[][] bikes) {
    // worker, bike mask, distance
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));
    pq.offer(new int[]{0, 0, 0});

    Set<String> visited = new HashSet<>();
    while (!pq.isEmpty()){
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
{% endhighlight %}

[Reachable Nodes In Subdivided Graph][reachable-nodes-in-subdivided-graph]

{% highlight java %}
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

    // {node, max moves}
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> -a[1]));
    pq.offer(new int[]{0, maxMoves});

    boolean[] visited = new boolean[n];
    int count = 0;
    while (!pq.isEmpty()) {
        int[] curr = pq.poll();
        int node = curr[0], move = curr[1];

        if (visited[node]) {
            continue;
        }

        visited[node] = true;
        count++;

        for (int neighbor = 0; neighbor < n; neighbor++) {
            if (graph[node][neighbor] >= 0) {
                if (move > graph[node][neighbor] && !visited[neighbor]) {
                    pq.offer(new int[]{neighbor, move - graph[node][neighbor] - 1});
                }

                // number of nodes that are reachable on the edge [node, neighbor]
                int reach = Math.min(move, graph[node][neighbor]);
                count += reach;

                // the remaining new nodes could be visited from the other direction
                graph[neighbor][node] -= reach;
            }
        }
    }
    return count;
}
{% endhighlight %}

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

{% highlight java %}
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
{% endhighlight %}

[Course Schedule IV][course-schedule-iv]

{% highlight java %}
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
{% endhighlight %}

# Paint and Expansion

[Shortest Bridge][shortest-bridge]

{% highlight java %}
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
{% endhighlight %}

[Minimum Number of Days to Disconnect Island][minimum-number-of-days-to-disconnect-island]

{% highlight java %}
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
{% endhighlight %}

[Making A Large Island][making-a-large-island]

{% highlight java %}
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
{% endhighlight %}

[campus-bikes-ii]: https://leetcode.com/problems/campus-bikes-ii/
[count-subtrees-with-max-distance-between-cities]: https://leetcode.com/problems/count-subtrees-with-max-distance-between-cities/
[course-schedule-iv]: https://leetcode.com/problems/course-schedule-iv/
[cheapest-flights-within-k-stops]: https://leetcode.com/problems/cheapest-flights-within-k-stops/
[making-a-large-island]: https://leetcode.com/problems/making-a-large-island/
[minimum-cost-to-reach-city-with-discounts]: https://leetcode.com/problems/minimum-cost-to-reach-city-with-discounts/
[minimum-cost-to-reach-destination-in-time]: https://leetcode.com/problems/minimum-cost-to-reach-destination-in-time/
[minimum-number-of-days-to-disconnect-island]: https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/
[minimum-weighted-subgraph-with-the-required-paths]: https://leetcode.com/problems/minimum-weighted-subgraph-with-the-required-paths/
[number-of-restricted-paths-from-first-to-last-node]: https://leetcode.com/problems/number-of-restricted-paths-from-first-to-last-node/
[number-of-ways-to-arrive-at-destination]: https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/
[path-with-maximum-minimum-value]: https://leetcode.com/problems/path-with-maximum-minimum-value/
[path-with-maximum-probability]: https://leetcode.com/problems/path-with-maximum-probability/
[path-with-minimum-effort]: https://leetcode.com/problems/path-with-minimum-effort/
[reachable-nodes-in-subdivided-graph]: https://leetcode.com/problems/reachable-nodes-in-subdivided-graph/
[shortest-bridge]: https://leetcode.com/problems/shortest-bridge/
[swim-in-rising-water]: https://leetcode.com/problems/swim-in-rising-water/
[the-maze-iii]: https://leetcode.com/problems/the-maze-iii/
