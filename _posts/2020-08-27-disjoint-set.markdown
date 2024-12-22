---
title:  "Disjoint Set"
category: algorithm
tags: [disjoint set, union find]
---

# Fundamentals

[Disjoint-set data structure](https://en.wikipedia.org/wiki/Disjoint-set_data_structure), also called union-find data structure, stores a collection of disjoint (non-overlapping) sets. Equivalently, it stores a partition of a set into disjoint subsets.

Time Complexity:

* find: `O(α(n))`
* union: `O(α(n))`

where α(n) is the extremely slow-growing [inverse Ackermann function](https://en.wikipedia.org/wiki/Ackermann_function#Inverse).

[Redundant Connection][redundant-connection]

```java
private int[] parents;

public int[] findRedundantConnection(int[][] edges) {
    parent = new int[edges.length + 1];

    for (int[] edge : edges) {
        if (!union(edge[0], edge[1])) {
            return edge;
        }
    }

    return null;
}

private int find(int u) {
    return parents[u] == 0 ? u : find(parents[u]);
}

private boolean union(int u, int v) {
    int pu = find(u), pv = find(v);

    if (pu == pv) {
        return false;
    }

    parents[pu] = pv;
    return true;
}
```

**Path compression**

```java
private int find(int u) {
    // path compression
    return parents[u] == 0 ? u : (parents[u] = find(parents[u]));
}
```

**Union by rank**

```c++
void unionSets(int u, int v) {
    int pu = find(u), pv = find(v);

    if (pu != pv) {
        if (ranks[pu] < ranks[pv]) {
            parents[pu] = pv;
        } else if (ranks[pu] > ranks[pv]) {
            parents[pv] = pu;
        } else {
            parents[pu] = pv;
            ranks[pv]++;
        }
    }
}
```

[Graph Connectivity With Threshold][graph-connectivity-with-threshold]

```java
public List<Boolean> areConnected(int n, int threshold, int[][] queries) {
    this.parents = new int[n + 1];

    for (int z = threshold + 1; z < n; z++) {
        // unions all multiples of z
        for (int k = 2; k * z <= n; k++) {
            union(z, k * z);
        }
    }

    List<Boolean> answer = new ArrayList<>();
    for (int[] q : queries) {
        answer.add(find(q[0]) == find(q[1]));
    }
    return answer;
}
```

Depending on the specific problem, `parents` elements can be initialized to `0`, `1`, `null`, etc.

[Checking Existence of Edge Length Limited Paths][checking-existence-of-edge-length-limited-paths]

Offline queries:

```java
private int[] parents;

public boolean[] distanceLimitedPathsExist(int n, int[][] edgeList, int[][] queries) {
    int m = queries.length;
    Integer[] indices = new Integer[m];
    for (int i = 0; i < m; i++) {
        indices[i] = i;
    }

    // sorts queries index by limit
    Arrays.sort(indices, Comparator.comparingInt(i -> queries[i][2]));

    // sorts edgeList by distance
    Arrays.sort(edgeList, Comparator.comparingInt(e -> e[2]));

    // union-find
    this.parents = new int[n];
    Arrays.fill(parents, -1);

    boolean[] answer = new boolean[m];
    int i = 0, j = 0;
    while (j < m) {
        int[] q = queries[indices[j]];
        // unions all nodes whose edge distance < q[2]
        // when q is updated, the existing disjoint sets remain the same
        // we just need to add new edges to the proper set
        while (i < edgeList.length && edgeList[i][2] < q[2]) {
            union(edgeList[i][0], edgeList[i][1]);
            i++;
        }

        // true if q nodes are in the same set
        if (find(q[0]) == find(q[1])) {
            answer[indices[j]] = true;
        }

        j++;
    }
    return answer;
}
```

[Redundant Connection II][redundant-connection-ii]

```java
public int[] findRedundantDirectedConnection(int[][] edges) {
    // there are two cases where the graph is invalid
    // - a node has two parents
    // - a cycle exists

    // parent of each node
    int[] parents = new int[edges.length + 1];

    // there's at most one node which has more than one parent
    // and this node has at most two parents
    // the two edges from the parents to this node are the two candidates

    // note the order of these two candidates matters
    int[] candidate1 = null, candidate2 = null;
    for (int[] e : edges) {
        if (parents[e[1]] == 0) {
            parents[e[1]] = e[0];
        } else {
            // there are two parents of e[1]
            candidate2 = new int[] {e[0], e[1]};
            candidate1 = new int[] {parents[e[1]], e[1]};

            // sets the edge of candidate2 to 0
            // so that later when we construct the graph
            // candidate2 is skipped
            e[1] = 0;
        }
    }

    // now uses the parents array as the roots for union-find
    Arrays.fill(parents, 0);

    // adds edges to the graph in order
    for (int[] e : edges) {
        if (e[1] == 0) {
            continue;
        }

        // found cycle
        if (!union(parents, e[0], e[1])) {
            // if there's no candidate edge, this edge is redundant
            // else candiate1 is redundant
            // because candidate2 is not in the graph yet

            // in other words, if there's a cycle, candidate1 has precedence over other edges
            return candidate1 == null ? e : candidate1;
        }
    }

    // there's no cycle
    // removes the candiate2
    return candidate2;
}

private int find(int[] parents, int u) {
    return parents[u] == 0 ? u : find(parents, parents[u]);
}

private boolean union(int[] parents, int u, int v) {
    int pu = find(parents, u), pv = find(parents, v);

    if (pu == pv) {
        return false;
    }

    parents[pu] = pv;
    return true;
}
```

Sometimes, the `parents` sets shall be represented as a map:

[Evaluate Division][evaluate-division]

```java
private Map<String, String> parents = new HashMap<>();
private Map<String, Double> ratios = new HashMap<>();  // parent / node

public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
    for (int i = 0; i < values.length; i++) {
        List<String> e = equations.get(i);
        union(e.get(0), e.get(1), values[i]);
    }

    int m = queries.size();
    double[] answers = new double[m];
    for (int i = 0; i < m; i++) {
        List<String> q = queries.get(i);
        answers[i] = query(q.get(0), q.get(1));
    }
    return answers;
}

private String find(String u) {
    if (!parents.containsKey(u) || parents.get(u).equals(u)) {
        ratios.put(u, 1.0);
        return u;
    }

    // path compression
    String p = parents.get(u), gp = find(p);
    parents.put(u, gp);
    ratios.put(u, ratios.get(u) * ratios.get(p));  // gp = p * ratio(p) = u * ratio(u) * ratio(p)
    return gp;
}

// u / v = value
private void union(String u, String v, double value) {
    String pu = find(u), pv = find(v);

    parents.put(pv, pu);
    // ratio = pu / pv
    //   = u * ratios(u) / (v * ratios(v))
    //   = value * ratios(u) / ratios(v)
    ratios.put(pv, value * ratios.get(u) / ratios.get(v));
}

private double query(String s1, String s2) {
    if (!ratios.containsKey(s1) || !ratios.containsKey(s2)) {
        return -1.0;
    }

    String p1 = find(s1), p2 = find(s2);
    if (!p1.equals(p2)) {
        return -1.0;
    }

    // s1 / s2 = p / ratios(s1) / (p / ratios(s2))
    //   = ratios(s2) / ratios(s1)
    return ratios.get(s2) / ratios.get(s1);
}
```

Another solution is to BFS/DFS the weighted graph.

[Checking Existence of Edge Length Limited Paths II][checking-existence-of-edge-length-limited-paths-ii]

Online queries:

```java
// snapshost[i]: for the i-th element, {snapId, val}
private List<int[]>[] snapshots;
private List<Integer> snapTime = new ArrayList<>();
private int snapId = 0;

public DistanceLimitedPathsExist(int n, int[][] edgeList) {
    this.snapshots = new List[n];
    for (int i = 0; i < n; i++) {
        snapshots[i] = new ArrayList<>();
        snapshots[i].add(new int[]{0, -1});
    }

    // sorts edgeList by distance
    Arrays.sort(edgeList, Comparator.comparingInt(e -> e[2]));

    // builds snapshots before any query
    // the dis are sorted in ascending order
    // so groups are growing
    int dis = 0;
    for (int[] e : edgeList) {
        // every time distance is increased, it's a snapshot
        if (e[2] > dis) {
            snapTime.add(dis);
            dis = e[2];
            snap();
        }
        union(e[0], e[1], snapId);
    }
    snapTime.add(dis);
    snap();
}

// id is the snapshot id
private int find(int u, int id) {
    // no path compression
    int p = get(u, id);
    return p < 0 ? u : find(p, id);
}

// id is the snapshot id
private void union(int u, int v, int id) {
    int pu = find(u, id), pv = find(v, id);
    // no path compression
    // because nodes other than u and v are not updated in this snapshot
    // otherwise there may be too many updates in one snapshot and impact performance
    if (pu != pv) {
        set(pu, pv);
    }
}

// 1146. Snapshot Array
private int get(int index, int id) {
    int pos = Collections.binarySearch(snapshots[index], new int[]{id, 0}, Comparator.comparingInt(a -> a[0]));
    if (pos < 0) {
        pos = ~pos - 1;
    }
    return snapshots[index].get(pos)[1];
}

private void set(int index, int val) {
    List<int[]> snapshot = snapshots[index];
    int size = snapshot.size();
    if (snapshot.get(size - 1)[0] == snapId) {  // overwrite
        snapshot.get(size - 1)[1] = val;
    } else {  // create
        snapshot.add(new int[]{snapId, val});
    }
}

// snapshot means potential updates on the parent of some nodes
private int snap() {
    return snapId++;
}

public boolean query(int p, int q, int limit) {
    // finds the first snapshot whose dis is strictly less than limit
    int id = Collections.binarySearch(snapTime, limit - 1);
    if (id < 0) {
        // the values of snapshot 0 is 0 <= limit - 1
        // so the insertion point can't be 0
        // ~id - 1 >= 0
        id = ~id - 1;
    }
    return find(p, id) == find(q, id);
}
```

# Number of Connected Componenets

[Number of Connected Components in an Undirected Graph][number-of-connected-components-in-an-undirected-graph]

```java
public int countComponents(int n, int[][] edges) {
    this.parent = new int[n];
    Arrays.fill(this.parent, -1);

    for (int[] e : edges) {
        if (union(e[0], e[1])) {
            n--;
        }
    }

    return n;
}
```

[Most Stones Removed with Same Row or Column][most-stones-removed-with-same-row-or-column]

```java
private Map<Integer, Integer> parent = new HashMap<>();
private int count = 0;

// counts the connected components
public int removeStones(int[][] stones) {
    for (int[] s : stones) {
        // ~ to distinguish r and c
        union(s[0], ~s[1]);
    }
    return stones.length - count;
}

private int find(int u) {
    // a new component
    if (parent.putIfAbsent(u, u) == null) {
        count++;
    }

    // if u is not root
    if (u != parent.get(u)) {
        parent.put(u, find(parent.get(u)));
    }

    return parent.get(u);
}

private void union(int u, int v) {
    int pu = find(u), pv = find(v);
    if (pu != pv) {
        parent.put(pu, pv);
        count--;
    }
}
```

[Remove Max Number of Edges to Keep Graph Fully Traversable][remove-max-number-of-edges-to-keep-graph-fully-traversable]

```java
public int maxNumEdgesToRemove(int n, int[][] edges) {
    // prioritizes Type 3
    Arrays.sort(edges, Comparator.comparingInt(e -> -e[0]));

    UnionFind alice = new UnionFind(n), bob = new UnionFind(n);

    // count of added edges
    int count = 0;
    for (int[] e : edges) {
        switch (e[0]) {
            case 1:
                if (alice.union(e[1], e[2])) {
                    count++;
                }
                break;
            case 2:
                if (bob.union(e[1], e[2])) {
                    count++;
                }
                break;
            case 3:
                // no short-circuit
                if (alice.union(e[1], e[2]) | bob.union(e[1], e[2])) {
                    count++;
                }
                break;
        }
    }

    return alice.isFullyConnected() && bob.isFullyConnected() ? edges.length - count : -1;
}

class UnionFind {
    int[] parents;
    int components;

    public UnionFind(int n) {
        this.parents = new int[n + 1];
        Arrays.fill(parents, -1);
        this.components = n;
    }

    private boolean union(int u, int v) {
        int pu = find(u), pv = find(v);

        if (pu == pv) {
            return false;
        }

        parents[pu] = pv;
        components--;

        return true;
    }

    private int find(int u) {
        // path compression
        int p = parents[u];
        if (p < 0) {
            return u;
        }
        return parents[u] = find(p);
    }

    private boolean isFullyConnected() {
        return components == 1;
    }
}
```

[Rank Transform of a Matrix][rank-transform-of-a-matrix]

```java
public int[][] matrixRankTransform(int[][] matrix) {
    int m = matrix.length, n = matrix[0].length;

    // matrix[i][j] : disjoint set
    Map<Integer, DisjointSet> disjointSets = new HashMap<>();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int v = matrix[i][j];
            disjointSets.putIfAbsent(v, new DisjointSet(m + n));

            // unions its row and col to the group
            disjointSets.get(v).union(i, j + m);
        }
    }

    // matrix[i][j] : map of groups
    // within each group, the members share the map key as the disjoint set root,
    // which means they are connected
    Map<Integer, Map<Integer, List<int[]>>> groups = new TreeMap<>();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int v = matrix[i][j];
            groups.computeIfAbsent(v, k -> new HashMap<>())
                .computeIfAbsent(disjointSets.get(v).find(i), r -> new ArrayList<>())
                .add(new int[]{i, j});
        }
    }

    int[][] answer = new int[m][n];
    int[] rowMax = new int[m], colMax = new int[n];
    for (var v : groups.values()) {
        // updates by connected cells with same value
        for (var cells : v.values()) {
            int rank = 1;
            for (int[] c : cells) {
                rank = Math.max(rank, Math.max(rowMax[c[0]], colMax[c[1]]) + 1);
            }

            for (int[] c : cells) {
                answer[c[0]][c[1]] = rank;
                rowMax[c[0]] = Math.max(rowMax[c[0]], answer[c[0]][c[1]]);
                colMax[c[1]] = Math.max(colMax[c[1]], answer[c[0]][c[1]]);
            }
        }
    }
    return answer;
}

class DisjointSet {
    int[] parent;

    public DisjointSet(int n) {
        parent = new int[n];
        Arrays.fill(parent, -1);
    }

    public int find(int u) {
        return parent[u] < 0 ? u : find(parent[u]);
    }

    public void union(int u, int v) {
        int pu = find(u), pv = find(v);
        if (pu != pv) {
            parent[pu] = pv;
        }
    }
}
```

[Regions Cut By Slashes][regions-cut-by-slashes]

```java
private int[] parent;
private int regions, n;

// splits each square into 4 triangles
private enum Triangle {
    TOP,
    RIGHT,
    BOTTOM,
    LEFT
}

public int regionsBySlashes(String[] grid) {
    this.n = grid.length;
    this.regions = n * n * 4;  // total number of triangles
    this.parent = new int[n * n * 4];

    for (int i = 0; i < regions; i++) {
        parent[i] = i;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // vertical
            if (i > 0) {
                union(indexOf(i - 1, j, Triangle.BOTTOM), indexOf(i, j, Triangle.TOP));
            }

            // horizontal
            if (j > 0) {
                union(indexOf(i, j - 1, Triangle.RIGHT), indexOf(i, j, Triangle.LEFT));
            }

            char c = grid[i].charAt(j);
            // '\\' or ' '
            if (c != '/') {
                union(indexOf(i, j, Triangle.TOP), indexOf(i, j, Triangle.RIGHT));
                union(indexOf(i, j, Triangle.BOTTOM), indexOf(i, j, Triangle.LEFT));
            }

            // '/' or ' '
            if (c != '\\') {
                union(indexOf(i, j, Triangle.TOP), indexOf(i, j, Triangle.LEFT));
                union(indexOf(i, j, Triangle.BOTTOM), indexOf(i, j, Triangle.RIGHT));
            }
        }
    }
    return regions;
}

private int find(int u) {
    return parent[u] == u ? u : find(parent[u]);
}

private void union(int u, int v) {
    int pu = find(u), pv = find(v);

    if (pu != pv) {
        parent[pu] = pv;
        regions--;
    }
}

private int indexOf(int i, int j, Triangle t) {
    return (i * n + j) * 4 + t.ordinal();
}
```

# Size of Each Set

[Minimize Malware Spread][minimize-malware-spread]

```java
private int[] parents;

public int minMalwareSpread(int[][] graph, int[] initial) {
    int n = graph.length;
    this.parents = new int[n];
    Arrays.fill(parents, -1);
}

private int find(int u) {
    return parents[u] < 0 ? u : find(parents[u]);
}

private void union(int u, int v) {
    int pu = find(u), pv = find(v);
    if (pu != pv) {
        // -parents[i] is the size of set i
        parents[pv] += parents[pu];
        parents[pu] = pv;
    }
}
```

[Minimize Malware Spread II][minimize-malware-spread-ii]

```java
public int minMalwareSpread(int[][] graph, int[] initial) {
    int n = graph.length;
    this.parents = new int[n];
    Arrays.fill(parents, -1);

    Set<Integer> initialSet = Arrays.stream(initial).boxed().collect(Collectors.toSet());
    // unions non-malware nodes
    for (int i = 0; i < n; i++) {
        if (initialSet.contains(i)) {
            continue;
        }
        for (int j = i + 1; j < n; j++) {
            if (!initialSet.contains(j) && graph[i][j] == 1) {
                union(i, j);
            }
        }
    }

    // finds the infected nodes by each initial malware
    Map<Integer, Set<Integer>> infected = new HashMap<>();

    // singleSource[p]: group p were infected by a single initial malware
    int[] numOfSources = new int[n];

    for (int i : initial) {
        for (int j = 0; j < n; j++) {
            if (!initialSet.contains(j) && graph[i][j] == 1) {
                int p = find(j);
                infected.computeIfAbsent(i, k -> new HashSet<>()).add(p);
            }
        }
        if (infected.containsKey(i)) {
            for (int p : infected.get(i)) {
                numOfSources[p]++;
            }
        }
    }

    int max = 0, index = -1;
    for (int i : initial) {
        if (infected.containsKey(i)) {
            int count = 0;
            for (var p : infected.get(i)) {
                if (numOfSources[p] == 1) {
                    count -= parents[p];
                }
            }
            if (count > max || (count == max && i < index)) {
                max = count;
                index = i;
            }
        }

    }
    return index >= 0 ? index : Arrays.stream(initial).min().getAsInt();
}
```

# Disconnected Components

[Process Restricted Friend Requests][process-restricted-friend-requests]

```java
public boolean[] friendRequests(int n, int[][] restrictions, int[][] requests) {
    this.parents = new Integer[n];

    int m = requests.length;
    boolean[] result = new boolean[m];
    for (int i = 0; i < m; i++) {
        int px = find(requests[i][0]), py = find(requests[i][1]);
        result[i] = true;
        if (px != py) {
            for (int[] r : restrictions) {
                int rx = find(r[0]), ry = find(r[1]);
                // connecting x and y is restricted
                if ((px == rx && py == ry) || (px == ry && py == rx)) {
                    result[i] = false;
                    break;
                }
            }
        }

        // unions two friends if request is valid
        if (result[i]) {
            union(px, py);
        }
    }
    return result;
}
```

[Find All People With Secret][find-all-people-with-secret]

```c++
vector<int> findAllPeople(int n, vector<vector<int>>& meetings, int firstPerson) {
    parents = vector<int>(n, -1);
    ranks = vector<int>(n);

    // Share the secret initially with firstPerson
    unionSets(0, firstPerson);

    // Sort meetings by time
    ranges::sort(meetings, {}, [&](const vector<int>& m){ return m[2]; });

    int m = meetings.size();
    // Track people involved in meetings at the current time
    unordered_set<int> people;
    for (int i = 0; i < m; i++) {
        // Share secrets during the meeting
        unionSets(meetings[i][0], meetings[i][1]);
        people.insert(meetings[i][0]);
        people.insert(meetings[i][1]);

        // Process at the end of each time frame
        if (i == m - 1 || meetings[i][2] != meetings[i + 1][2]) {
            for (int p : people) {
                // Disconnect people not knowing the secret from Person 0
                if (find(p) != find(0)) {
                    parents[p] = -1;
                }
            }
            people.clear();
        }
    }

    vector<int> v;
    for (int i = 0; i < n; i++) {
        if (find(i) == find(0)) {
            v.push_back(i);
        }
    }
    return v;
}
```

[Number of Good Paths][number-of-good-paths]

```java
private int[] parents, vals;
// maxCounts[i]: {max, count of max} in the root i
private int[][] maxCounts;

public int numberOfGoodPaths(int[] vals, int[][] edges) {
    int n = vals.length;
    this.parents = new int[n];
    this.maxCounts = new int[n][2];
    for (int i = 0; i < n; i++) {
        parents[i] = -1;
        maxCounts[i][0] = vals[i];
        maxCounts[i][1] = 1;
    }

    this.vals = vals;

    // buils the tree with nodes in ascending value order
    Arrays.sort(edges, Comparator.comparingInt(e -> Math.max(vals[e[0]], vals[e[1]])));

    int count = n;
    for (int[] e : edges) {
        count += union(e[0], e[1]);
    }
    return count;
}

private int find(int u) {
    return parents[u] < 0 ? u : (parents[u] = find(parents[u]));
}

private int union(int u, int v) {
    int pu = find(u), pv = find(v);
    if (pu == pv) {
        return 0;
    }

    parents[pu] = pv;

    int maxVal = Math.max(vals[u], vals[v]);
    int cu = maxCounts[pu][0] == maxVal ? maxCounts[pu][1] : 0;
    int cv = maxCounts[pv][0] == maxVal ? maxCounts[pv][1] : 0;

    maxCounts[pv][0] = maxVal;
    maxCounts[pv][1] = cu + cv;

    // combinatorics multiplication principle
    return cu * cv;
}
```

[checking-existence-of-edge-length-limited-paths]: https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/
[checking-existence-of-edge-length-limited-paths-ii]: https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths-ii/
[evaluate-division]: https://leetcode.com/problems/evaluate-division/
[find-all-people-with-secret]: https://leetcode.com/problems/find-all-people-with-secret/
[graph-connectivity-with-threshold]: https://leetcode.com/problems/graph-connectivity-with-threshold/
[minimize-malware-spread]: https://leetcode.com/problems/minimize-malware-spread/
[minimize-malware-spread-ii]: https://leetcode.com/problems/minimize-malware-spread-ii/
[most-stones-removed-with-same-row-or-column]: https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/
[number-of-connected-components-in-an-undirected-graph]: https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/
[number-of-good-paths]: https://leetcode.com/problems/number-of-good-paths/
[process-restricted-friend-requests]: https://leetcode.com/problems/process-restricted-friend-requests/
[rank-transform-of-a-matrix]: https://leetcode.com/problems/rank-transform-of-a-matrix/
[redundant-connection]: https://leetcode.com/problems/redundant-connection/
[redundant-connection-ii]: https://leetcode.com/problems/redundant-connection-ii/
[regions-cut-by-slashes]: https://leetcode.com/problems/regions-cut-by-slashes/
[remove-max-number-of-edges-to-keep-graph-fully-traversable]: https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/
