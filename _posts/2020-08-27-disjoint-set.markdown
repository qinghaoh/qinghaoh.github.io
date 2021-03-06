---
layout: post
title:  "Disjoint Set"
tags: union
---

Time Complexity:

* find: `O(α(n))`
* union: `O(α(n))`

where α(n) is the extremely slow-growing [inverse Ackermann function](https://en.wikipedia.org/wiki/Ackermann_function#Inverse).

[Redundante Connection][redundant-connection]

{% highlight java %}
private int[] parent;

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
    return parent[u] == 0 ? u : find(parent[u]);
}

private boolean union(int u, int v) {
    int pu = find(u), pv = find(v);

    if (pu == pv) {
        return false;
    }

    parent[pu] = pv;
    return true;
}
{% endhighlight %}

[Evaluate Division][evaluate-division]

{% highlight java %}
Map<String, String> parent = new HashMap<>();
Map<String, Double> ratio = new HashMap<>();  // parent / node

public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
    for (int i = 0; i < values.length; i++) {
        List<String> e = equations.get(i);
        union(e.get(0), e.get(1), values[i]);
    }

    double[] result = new double[queries.size()];
    for (int i = 0; i < queries.size(); i++) {
        List<String> q = queries.get(i);
        result[i] = query(q.get(0), q.get(1));
    }
    return result;
}

private String find(String u) {
    if (!parent.containsKey(u)) {
        ratio.put(u, 1.0);
        return u;
    }

    // path compression
    String p = parent.get(u);
    String gp = find(p);
    parent.put(u, gp);
    ratio.put(u, ratio.get(u) * ratio.get(p));  // gp = p * ratio(p) = u * ratio(u) * ratio(p)
    return gp;
}

// u / v = value
private void union(String u, String v, double value) {
    String pu = find(u), pv = find(v);

    parent.put(pv, pu);
    // ratio = pu / pv
    //   = u * ratio(u) / (v * ratio(v))
    //   = value * ratio(u) / ratio(v)
    ratio.put(pv, value * ratio.get(u) / ratio.get(v));
}

private double query(String s1, String s2) {
    if (!ratio.containsKey(s1) || !ratio.containsKey(s2)) {
        return -1.0;
    }

    String p1 = find(s1), p2 = find(s2);
    if (!p1.equals(p2)) {
        return -1.0;
    }

    // s1 / s2 = p / ratio(s1) / (p / ratio(s2))
    //   = ratio(s2) / ratio(s1)
    return ratio.get(s2) / ratio.get(s1);
}
{% endhighlight %}

# Number of Connected Componenets

[Number of Connected Components in an Undirected Graph][number-of-connected-components-in-an-undirected-graph]

{% highlight java %}
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
{% endhighlight %}

[Most Stones Removed with Same Row or Column][most-stones-removed-with-same-row-or-column]

{% highlight java %}
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
{% endhighlight %}

[Rank Transform of a Matrix][rank-transform-of-a-matrix]

{% highlight java %}
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
{% endhighlight %}

[Regions Cut By Slashes][regions-cut-by-slashes]

{% highlight java %}
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
    int uset = find(u), vset = find(v);

    if (uset != vset) {
        parent[uset] = vset;
        regions--;
    }
}

private int indexOf(int i, int j, Triangle t) {
    return (i * n + j) * 4 + t.ordinal();
}
{% endhighlight %}

[evaluate-division]: https://leetcode.com/problems/evaluate-division/
[most-stones-removed-with-same-row-or-column]: https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/
[number-of-connected-components-in-an-undirected-graph]: https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/
[rank-transform-of-a-matrix]: https://leetcode.com/problems/rank-transform-of-a-matrix/
[redundant-connection]: https://leetcode.com/problems/redundant-connection/
[regions-cut-by-slashes]: https://leetcode.com/problems/regions-cut-by-slashes/
