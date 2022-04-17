---
layout: post
title:  "Line Sweep"
tags: math
usemathjax: true
---
# Fundamentals

Line sweep is an algorithm to solve [The Skyline Problem][the-skyline-problem]. The key idea is to keep track of every change (delta) at each position, then linear scan and process the positions in the line. This delta is very similar to the pulses in [Discrete Time](https://en.wikipedia.org/wiki/Discrete_time_and_continuous_time#Discrete_time) Signal Processing.

There are two basic forms of algorithm. The first form is to use a list to record the deltas of all position, then sort the list with regard to positions. For each position, there can be more than one list element, and we consolidate them during linear scan. For example:
 
[Average Height of Buildings in Each Segment][average-height-of-buildings-in-each-segment]

{% highlight java %}
public int[][] averageHeightOfBuildings(int[][] buildings) {
    // {point, (+/-)height}
    List<int[]> heights = new ArrayList<>();
    for (int[] b : buildings) {
        heights.add(new int[]{b[0], b[2]});
        heights.add(new int[]{b[1], -b[2]});
    }

    Collections.sort(heights, Comparator.comparingInt(a -> a[0]));

    List<int[]> list = new ArrayList<>();
    int sum = 0, count = 0;
    for (int i = 0, j = 0; i < heights.size(); i = j) {
        // updates end of prev int[]
        if (sum > 0) {
            list.get(list.size() - 1)[1] = heights.get(i)[0];
        }

        // consolidates height sum and count at the current position
        for (j = i; j < heights.size() && heights.get(i)[0] == heights.get(j)[0]; j++) {
            sum += heights.get(j)[1];
            count += heights.get(j)[1] > 0 ? 1 : -1;
        }

        // if any of the following is true, we need a new result block
        // - empty list
        // - curr != end of prev int[]
        // - average != average of prev int[]
        if (count > 0 &&
            (list.isEmpty() ||
             list.get(list.size() - 1)[1] != heights.get(i)[0] ||
             list.get(list.size() - 1)[2] != sum / count)) {
            list.add(new int[]{heights.get(i)[0], heights.get(i)[0], sum / count});
        }
    }

    int[][] street = new int[list.size()][];
    for (int i = 0; i < street.length; i++) {
        street[i] = list.get(i);
    }
    return street;
}
{% endhighlight %}

The second form is to use an array or an ordered map to store the deltas of all positions. The positions are used as keys, so the position of each entry is unique. The value of each entry is the consolidated deltas. For example:

[Average Height of Buildings in Each Segment][average-height-of-buildings-in-each-segment]

{% highlight java %}
public int[][] averageHeightOfBuildings(int[][] buildings) {
    // {height sum, count}
    Map<Integer, int[]> map = new TreeMap<>();
    for (int[] b : buildings) {
        if (map.containsKey(b[0])) {
            map.get(b[0])[0] += b[2];
            map.get(b[0])[1]++;
        } else {
            map.put(b[0], new int[]{b[2], 1});
        }
        if (map.containsKey(b[1])) {
            map.get(b[1])[0] -= b[2];
            map.get(b[1])[1]--;
        } else {
            map.put(b[1], new int[]{-b[2], -1});
        }
    }

    List<int[]> list = new ArrayList<>();
    int sum = 0, count = 0;
    for (var e : map.entrySet()) {
        int k = e.getKey();
        int[] v = e.getValue();

        // updates end of prev int[]
        if (sum > 0) {
            list.get(list.size() - 1)[1] = k;
        }

        // accumulates sum and count
        sum += v[0];
        count += v[1];

        // if any of the following is true, we need a new result block
        // - empty list
        // - curr != end of prev int[]
        // - average != average of prev int[]
        if (count > 0 &&
           (list.isEmpty() ||
            list.get(list.size() - 1)[1] != k ||
            list.get(list.size() - 1)[2] != sum / count)) {
            list.add(new int[]{k, k, sum / count});
        }
    }

    int[][] street = new int[list.size()][];
    for (int i = 0; i < list.size(); i++) {
        street[i] = list.get(i);
    }
    return street;
}
{% endhighlight %}

## List Form

[The Skyline Problem][the-skyline-problem]

{% highlight java %}
public List<List<Integer>> getSkyline(int[][] buildings) {
    List<int[]> heights = new ArrayList<>();
    for (int[] b: buildings) {
        // height at start is stored as negative
        heights.add(new int[]{b[0], -b[2]});
        // height at end is stored as positive
        heights.add(new int[]{b[1], b[2]});
    }

    Collections.sort(heights, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);

    TreeMap<Integer, Integer> map = new TreeMap<>();
    // a trick to deal with last building in a block
    map.put(0, 1);

    List<List<Integer>> list = new ArrayList<>();
    int prev = 0;
    for (int[] h: heights) {
        if (h[1] < 0) {
            // if it's start, puts/increments the height to map
            map.put(-h[1], map.getOrDefault(-h[1], 0) + 1);
        } else {
            // if it's end, removes/decrements the height from map
            map.put(h[1], map.get(h[1]) - 1);
            map.remove(h[1], 0);
        }

        // gets the max height
        int curr = map.firstKey();
        // no consecutive horizontal lines of equal height
        if (prev != curr) {
            list.add(Arrays.asList(h[0], curr));
            prev = curr;
        }
    }
    return list;
}
{% endhighlight %}

## Array/Ordered Map Form

**Array**

[Range Addition][range-addition]

{% highlight java %}
public int[] getModifiedArray(int length, int[][] updates) {
    int[] arr = new int[length];
    // finds pulses
    for (int[] u : updates) {
        arr[u[0]] += u[2];
        if (u[1] + 1 < length) {
            arr[u[1] + 1] -= u[2];
        }
    }

    // accumulates pulses
    for (int i = 1; i < length; i++) {
        arr[i] += arr[i - 1];
    }

    return arr;
}
{% endhighlight %}

[Count Positions on Street With Required Brightness][count-positions-on-street-with-required-brightness]

{% highlight java %}
public int meetRequirement(int n, int[][] lights, int[] requirement) {
    int[] brightness = new int[n + 1];
    for (int[] l : lights) {
        brightness[Math.max(0, l[0] - l[1])]++;
        brightness[Math.min(n, l[0] + l[1] + 1)]--;
    }

    int b = 0, count = 0;
    for (int i = 0; i < n; i++) {
        b += brightness[i];
        if (b >= requirement[i]) {
            count++;
        }
    }
    return count;
}
{% endhighlight %}

At each position, we sum the deltas and store them in an ordered map or priority queue.

**Ordered Map**

[Describe the Painting][describe-the-painting]

{% highlight java %}
public List<List<Long>> splitPainting(int[][] segments) { 
    Map<Integer, Long> map = new TreeMap<>();
    for (int[] s : segments) {
        map.put(s[0], map.getOrDefault(s[0], 0l) + s[2]);
        map.put(s[1], map.getOrDefault(s[1], 0l) - s[2]);
    }

    List<List<Long>> painting = new ArrayList<>();
    int prev = 0;
    long color = 0;
    for (int curr : map.keySet()) {
        // color == 0 means this segment is not painted
        if (prev > 0 && color > 0) {
            painting.add(Arrays.asList((long)prev, (long)curr, color));
        }

        color += map.get(curr);
        prev = curr;
    }
    return painting;
}
{% endhighlight %}

[Minimum Interval to Include Each Query][minimum-interval-to-include-each-query]

{% highlight java %}
public int[] minInterval(int[][] intervals, int[] queries) {
    // {size, end}
    // we can use priority queue as well
    TreeMap<Integer, Integer> map = new TreeMap<>();

    // sorts intervals
    Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));

    // sorts queries index mapping array
    int n = intervals.length, m = queries.length;
    Integer[] index = new Integer[m];
    for (int i = 0; i < m; i++) {
        index[i] = i;
    }
    Arrays.sort(index, Comparator.comparingInt(i -> queries[i]));

    int j = 0;  // index of current interval
    int[] ans = new int[m];
    // scans through queries in order
    for (int i : index) {
        // enqueues
        while (j < n && intervals[j][0] <= queries[i]) {
            map.put(intervals[j][1] - intervals[j][0] + 1, intervals[j][1]);
            j++;
        }

        // dequeues
        while (!map.isEmpty() && map.firstEntry().getValue() < queries[i]) {
            map.pollFirstEntry();
        }
        ans[i] = map.isEmpty() ? -1 : map.firstKey();
    }
    return ans;
}
{% endhighlight %}

# Arc Sweep

[Maximum Number of Darts Inside of a Circular Dartboard][maximum-number-of-darts-inside-of-a-circular-dartboard]

![Arc sweep (created by https://www.geogebra.org/)](/assets/maximum_number_of_darts_inside_of_a_circular_dartboard.png)

{% highlight java %}
public int numPoints(int[][] points, int r) {
    // at least 1 point can be included by the circle
    int max = 1;

    for (int[] p : points) {
        List<double[]> angles = new ArrayList<>();
        for (int [] q : points) {
            // for all q that are within 2r radius of p
            if (p[0] != q[0] || p[1] != q[1]) {
                double d = distance(p, q);
                if (d <= 2 * r) {
                    // angle between line p-q and the positive x axis.
                    double angle = Math.atan2(q[1] - p[1], q[0] - p[0]);

                    // half of the span between entry and exit
                    double delta = Math.acos(d / (2 * r));

                    // entry
                    angles.add(new double[]{angle - delta, 1});
                    // exit
                    angles.add(new double[]{angle + delta, -1});
                }
            }

            Collections.sort(angles, (a, b) -> a[0] == b[0] ? Double.compare(b[1], a[1]) : Double.compare(a[0], b[0]));

            // p is included
            int count = 1;
            for (var e : angles) {
                max = Math.max(max, count += e[1]);
            }
        }
    }

    return max;
}

private double distance(int[] p1, int[] p2) {
    return Math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]));
}
{% endhighlight %}

# Inclusive End

[Brightest Position on Street][brightest-position-on-street]

{% highlight java %}
public int brightestPosition(int[][] lights) {
    int n = lights.length;
    Map<Integer, Integer> map = new TreeMap<>();
    for (int i = 0; i < n; i++) {
        int start = lights[i][0] - lights[i][1];
        int end = lights[i][0] + lights[i][1] + 1;
        map.put(start, map.getOrDefault(start, 0) + 1);
        map.put(end, map.getOrDefault(end, 0) - 1);
    }

    int max = 0, index = 0, count = 0;
    for (int p : map.keySet()) {
        count += map.get(p);
        if (count > max) {
            max = count;
            index = p;
        }
    }
    return index;
}
{% endhighlight %}

[Amount of New Area Painted Each Day][amount-of-new-area-painted-each-day]

{% highlight java %}
public int[] amountPainted(int[][] paint) {
    int n = paint.length;
    int max = Arrays.stream(paint).mapToInt(p -> p[1]).max().getAsInt();

    // area[i]: end of continuous painted area starting at i
    int[] area = new int[max + 1], worklog = new int[n];
    for (int i = 0; i < n; i++) {
        int start = paint[i][0], end = paint[i][1];
        while (start < end) {
            // next position of the brush
            int jump = 0;
            // if the area is empty (area[start] == 0), jumps one step forward
            if (area[start] == 0) {
                jump = start + 1;
                worklog[i]++;
            } else {
                // jumps to the end of existing painted area
                jump = area[start];
            }

            // updates the end of the painted area starting from current `start`
            if (end > area[start]) {
                area[start] = end;
            }

            start = jump;
        }
    }
    return worklog;
}
{% endhighlight %}

[amount-of-new-area-painted-each-day]: https://leetcode.com/problems/amount-of-new-area-painted-each-day/
[average-height-of-buildings-in-each-segment]: https://leetcode.com/problems/average-height-of-buildings-in-each-segment/
[brightest-position-on-street]: https://leetcode.com/problems/brightest-position-on-street/
[count-positions-on-street-with-required-brightness]: https://leetcode.com/problems/count-positions-on-street-with-required-brightness/
[describe-the-painting]: https://leetcode.com/problems/describe-the-painting/
[maximum-number-of-darts-inside-of-a-circular-dartboard]: https://leetcode.com/problems/maximum-number-of-darts-inside-of-a-circular-dartboard/
[minimum-interval-to-include-each-query]: https://leetcode.com/problems/minimum-interval-to-include-each-query/
[range-addition]: https://leetcode.com/problems/range-addition/
[the-skyline-problem]: https://leetcode.com/problems/the-skyline-problem/
