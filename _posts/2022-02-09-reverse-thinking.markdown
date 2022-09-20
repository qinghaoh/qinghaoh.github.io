---
layout: post
title:  "Reverse Thinking"
---
[Maximum Segment Sum After Removals][maximum-segment-sum-after-removals]

{% highlight java %}
private long[] parents;

public long[] maximumSegmentSum(int[] nums, int[] removeQueries) {
    // reverse union-find
    int n = nums.length;
    this.parents = new long[n];
    // max long is used to mark unvisited
    Arrays.fill(parents, Long.MAX_VALUE);

    long[] answer = new long[n];
    for (int i = n - 1; i > 0; i--) {
        int q = removeQueries[i];
        parents[q] = -nums[q];

        // unions with left interval
        if (q > 0 && parents[q - 1] != Long.MAX_VALUE) {
            union(q, q - 1);
        }

        // unions with right interval
        if (q < n - 1 && parents[q + 1] != Long.MAX_VALUE) {
            union(q, q + 1);
        }

        answer[i - 1] = Math.max(answer[i], -parents[find(q)]);
    }
    return answer;
}

private int find(int u) {
    return parents[u] < 0 ? u : (int)(parents[u] = find((int)parents[u]));
}

private void union(int u, int v) {
    int pu = find(u), pv = find(v);
    // negated sum
    parents[pv] += parents[pu];
    parents[pu] = pv;
}
{% endhighlight %}

[Execution of All Suffix Instructions Staying in a Grid][execution-of-all-suffix-instructions-staying-in-a-grid]

{% highlight java %}
// notice the direction is opposition to the instruction
// because we will process the instructions in reverse order
private static final Map<Character, int[]> DIRECTIONS = Map.of(
    'U', new int[]{1, 0},
    'L', new int[]{0, 1},
    'D', new int[]{-1, 0},
    'R', new int[]{0, -1}
);

public int[] executeInstructions(int n, int[] startPos, String s) {
    int m = s.length();
    int[] answer = new int[m];

    // steps to move off the grid from the start position
    int top = startPos[0] + 1, bottom = n - startPos[0], left = startPos[1] + 1, right = n - startPos[1];

    // (r, c) is the virtual start
    // {r, index of instruction}
    Map<Integer, Integer> rMap = new HashMap();
    rMap.put(0, m);
    // {c, index of instruction}
    Map<Integer, Integer> cMap = new HashMap();
    cMap.put(0, m);

    // virtual start position at each index when the final position is (0, 0)
    // assuming the grid has no boundary
    // updates it when processing the instructions in reverse order
    int[] pos = new int[2];
    for (int i = m - 1; i >= 0; i--) {
        int[] instr = DIRECTIONS.get(s.charAt(i));
        pos[0] += instr[0];
        pos[1] += instr[1];

        int d = m - i;
        // checks boundaries of four directions, centered at pos
        if (rMap.containsKey(pos[0] - top)) {
            d = Math.min(d, rMap.get(pos[0] - top) - i - 1);
        }

        if (rMap.containsKey(pos[0] + bottom)) {
            d = Math.min(d, rMap.get(pos[0] + bottom) - i - 1);
        }

        if (cMap.containsKey(pos[1] - left)) {
            d = Math.min(d, cMap.get(pos[1] - left) - i - 1);
        }

        if (cMap.containsKey(pos[1] + right)) {
            d = Math.min(d, cMap.get(pos[1] + right) - i - 1);
        }

        rMap.put(pos[0], i);
        cMap.put(pos[1], i);
        answer[i] = d;
    }
    return answer;
}
{% endhighlight %}

[execution-of-all-suffix-instructions-staying-in-a-grid]: https://leetcode.com/problems/execution-of-all-suffix-instructions-staying-in-a-grid/
[maximum-segment-sum-after-removals]: https://leetcode.com/problems/maximum-segment-sum-after-removals/
