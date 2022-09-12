---
layout: post
title:  "Segment Tree"
tags: tree
usemathjax: true
---
# Fundamentals

[Segment tree](https://en.wikipedia.org/wiki/Segment_tree)

Given a set `I` of intervals, or segments, a segment tree `T` for `I` is structured as follows:

* `T` is a binary tree
* Its leaves correspond to the elementary intervals induced by the endpoints in `I`, in an ordered way
* The internal nodes of `T` correspond to intervals that are the union of elementary intervals: the interval corresponding to node `N` is the union of the intervals corresponding to the leaves of the tree rooted at `N`

Time Complexity:
* Construct: \\(O(nlog(n))\\)
* Update: \\(O(log(n))\\)
* Query (search for all the intervals that contain a query point): \\(O(log(n) + k)\\), \\(k\\) being the number of retrieved intervals or segments

Space Complexity: \\(O(nlog(n))\\)

Efficient range query, while array modification is flexible.

The standard Segment Tree requires \\(4n\\) vertices for working on an array of size \\(n\\).

## Implementation

### Commutative

The implementation below is based on Al.Cash's blog [Efficient and easy segment trees](https://codeforces.com/blog/entry/18051). We generalize the implementation to support a commutative bi-function `f(x, y)`:

{% highlight java %}
class SegmentTree {
    private int n;
    private int[] arr;
    private BiFunction<Integer, Integer, Integer> f;

    // default all-zero array
    public SegmentTree(int n, BiFunction<Integer, Integer, Integer> f) {
        this.n = n;
        this.arr = new int[2 * n];
        this.f = f;
    }

    // initializes array with initValue
    public SegmentTree(int n, int initValue, BiFunction<Integer, Integer, Integer> f) {
        this(n, f);
        Arrays.fill(arr, 0, this.n, initValue);
    }

    public SegmentTree(int[] nums, BiFunction<Integer, Integer, Integer> f) {
        this(nums.length, f);
        System.arraycopy(nums, 0, arr, this.n, this.n);
    }

    public void build() {
        for (int i = n - 1; i > 0; i--) {
            arr[i] = f.apply(arr[i * 2], arr[i * 2 + 1]);
        }
    }

    // set nums[index] = value
    public void update(int index, int value) {
        for (arr[index += n] = value; index > 1; index /= 2) {
            // index and index ^ 1 are siblings
            arr[index / 2] = f.apply(arr[index], arr[index ^ 1]);
        }
    }

    // f.apply() on interval [start, end)
    public int query(int start, int end) {
        int res = 0;
        for (start += n, end += n; start < end; start /= 2, end /= 2) {
            if (start % 2 == 1) {
                res = f.apply(res, arr[start++]);
            }
            if (end % 2 == 1) {
                res = f.apply(res, arr[--end]);
            }
        }
        return res;
    }
}
{% endhighlight %}

Examples of commutative bi-functions:
* Sum
* Min
* Max

**Max**

[Longest Increasing Subsequence II][longest-increasing-subsequence-ii]

{% highlight java %}
public int lengthOfLIS(int[] nums, int k) {
    SegmentTree st = new SegmentTree(Arrays.stream(nums).max().getAsInt() + 1, (a, b) -> Math.max(a, b));
    st.build();

    int max = 0;
    for (int num : nums) {
        // implicit rolling dp:
        // dp[i]: LIS until the current element, and the last element of the LIS is i
        // finds the max in the range of the prev level dp
        int prev = st.query(Math.max(1, num - k), num);
        st.update(num, prev + 1);
        max = Math.max(max, prev + 1);
    }
    return max;
}
{% endhighlight %}

[Booking Concert Tickets in Groups][booking-concert-tickets-in-groups]

{% highlight java %}
{% endhighlight %}

### Non-commutative

[Longest Substring of One Repeating Character][longest-substring-of-one-repeating-character]

{% highlight java %}
public int[] longestRepeating(String s, String queryCharacters, int[] queryIndices) {
    int n = s.length();
    Node[] nodes = new Node[n];
    for (int i = 0; i < n; i++) {
        nodes[i] = new Node(s.charAt(i));
    }

    SegmentTree st = new SegmentTree(nodes, (a, b) -> combine(a, b));
    st.build();

    int k = queryIndices.length;
    int[] lengths = new int[k];
    for (int i = 0; i < k; i++) {
        st.update(queryIndices[i], new Node(queryCharacters.charAt(i)));
        lengths[i] = st.query(0, n).longest;
    }
    return lengths;
}

class Node {
    // e.g. "aabccc"
    // len = 6
    // longest = 3
    // leftChar = 'a', rightChar = 'c'
    // pLen = 2, sLen = 3
    int len = 0;
    int longest = 0;
    char leftChar = 0, rightChar = 0;
    // length of longest prefix/suffix substring of one repeating character
    int pLen = 0, sLen = 0;

    public Node() {
    }

    public Node(char ch) {
        this.leftChar = this.rightChar = ch;
        this.pLen = this.sLen = 1;
        this.longest = 1;
        this.len = 1;
    }
}

public Node combine(Node left, Node right) {
    Node node = new Node();
    node.longest = Math.max(left.longest, right.longest);
    if (left.rightChar == right.leftChar) {
        node.longest = Math.max(node.longest, left.sLen + right.pLen);
    }

    node.len = left.len + right.len;
    node.leftChar = left.leftChar;
    node.rightChar = right.rightChar;
    node.pLen = left.pLen + (left.pLen == left.len && left.leftChar == right.leftChar ? right.pLen : 0);
    node.sLen = right.sLen + (right.sLen == right.len && right.rightChar == left.rightChar ? left.sLen : 0);

    return node;
}

// non-commutative
class SegmentTree {
    private int n;
    private Node[] arr;
    private BiFunction<Node, Node, Node> f;

    // default all-zero array
    public SegmentTree(int n, BiFunction<Node, Node, Node> f) {
        this.n = n;
        this.arr = new Node[2 * n];
        this.f = f;
    }

    // initializes array with initValue
    public SegmentTree(int n, Node initValue, BiFunction<Node, Node, Node> f) {
        this(n, f);
        Arrays.fill(arr, 0, this.n, initValue);
    }

    public SegmentTree(Node[] nums, BiFunction<Node, Node, Node> f) {
        this(nums.length, f);
        System.arraycopy(nums, 0, arr, this.n, this.n);
    }

    public void build() {
        for (int i = n - 1; i > 0; i--) {
            arr[i] = f.apply(arr[i * 2], arr[i * 2 + 1]);
        }
    }

    // set nums[index] = value
    public void update(int index, Node value) {
        for (arr[index += n] = value; (index /= 2) > 0; ) {
            // ensures the correct ordering of children, knowing that left child has even index
            arr[index] = f.apply(arr[2 * index], arr [2 * index + 1]);
        }
    }

    // f.apply() on interval [start, end)
    public Node query(int start, int end) {
        Node resl = new Node(), resr = new Node();
        for (start += n, end += n; start < end; start /= 2, end /= 2) {
            // nodes corresponding to the left border are processed from left to right
            // while the right border moves from right to left
            if (start % 2 == 1) {
                resl = f.apply(resl, arr[start++]);
            }
            if (end % 2 == 1) {
                resr = f.apply(arr[--end], resr);
            }
        }
        return f.apply(resl, resr);
    }
}
{% endhighlight %}

[Rectangle Area II][rectangle-area-ii]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int rectangleArea(int[][] rectangles) {
    // x1 and x2 of all rectangles
    Set<Integer> xCoordinates = new TreeSet<>();

    // rectangle tuple list:
    //   {y, x1, x2, sign}
    // specifically,
    // - {y1, x1, x2, 1}
    // - {y2, x1, x2, -1}
    // we will sweep lines up from y = 0
    // - when y == y1, we are about to sweep the rectangle, sign > 0
    // - when y == y2, we just finished sweeping the rectangle, sign < 0
    List<int[]> rList = new ArrayList<>();
    for (int[] r : rectangles) {
        xCoordinates.add(r[0]);
        xCoordinates.add(r[2]);
        rList.add(new int[]{r[1], r[0], r[2], 1});
        rList.add(new int[]{r[3], r[0], r[2], -1});
    }

    // x coordinate : ordinality in the ordered set
    Map<Integer, Integer> xOrdinality = new HashMap<>();
    int index = 0;
    for (int x : xCoordinates) {
        xOrdinality.put(x, index++);
    }

    // sorts rList by y
    Collections.sort(rList, (a, b) -> Integer.compare(a[0], b[0]));

    // count[i]: count of rectangles covering x[i, i + 1) on this line.
    int[] count = new int[xCoordinates.size()];

    // sweeps lines up from y = 0
    long area = 0, prevLineSum = 0;
    int prevY = 0;
    for (int[] r : rList) {
        int y = r[0], x1 = r[1], x2 = r[2], sign = r[3];
        area = (area + (y - prevY) * prevLineSum) % MOD;
        prevY = y;

        // updates count of rectangles covering the current line
        for (int i = xOrdinality.get(x1); i < xOrdinality.get(x2); i++) {
            count[i] += sign;
        }

        // counts "area" of this line
        // if we use segment tree here,
        // the time complexity can be improved to O(log(n))
        prevLineSum = 0;
        index = 0;
        Iterator<Integer> itr = xCoordinates.iterator();
        int prev = itr.next();
        while (itr.hasNext()) {
            int curr = itr.next();

            // if the current x interval is covered by some rectangle
            // the interval will be part of the final area
            // adds it to the current line sum (of x intervals)
            if (count[index++] > 0) {
                prevLineSum += curr - prev;
            }

            prev = curr;
        }
    }
    return (int)area;
}
{% endhighlight %}

Segment Tree:

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int rectangleArea(int[][] rectangles) {
    // x1 and x2 of all rectangles
    List<Integer> xCoordinates = new ArrayList<>();

    // rectangle tuple list:
    //   {y, x1, x2, sign}
    // specifically,
    // - {y1, x1, x2, 1}
    // - {y2, x1, x2, -1}
    // we will sweep lines up from y = 0
    // - when y == y1, we are about to sweep the rectangle, sign > 0
    // - when y == y2, we just finished sweeping the rectangle, sign < 0
    List<int[]> rList = new ArrayList<>();
    for (int[] r : rectangles) {
        if ((r[0] < r[2]) && (r[1] < r[3])) {
            xCoordinates.add(r[0]);
            xCoordinates.add(r[2]);
            rList.add(new int[]{r[1], r[0], r[2], 1});
            rList.add(new int[]{r[3], r[0], r[2], -1});
        }
    }

    // sorts rList by y
    Collections.sort(rList, (a, b) -> Integer.compare(a[0], b[0]));

    // sorts x coordinates
    Collections.sort(xCoordinates);

    // x coordinate : ordinality in the ordered set
    Map<Integer, Integer> xOrdinality = new HashMap<>();
    for (int i = 0; i < xCoordinates.size(); i++) {
        xOrdinality.put(xCoordinates.get(i), i);
    }

    SegmentTreeNode st = new SegmentTreeNode(0, xCoordinates.size() - 1, xCoordinates);

    // sweeps lines up from y = 0
    long area = 0, prevLineSum = 0;
    int prevY = 0;
    for (int[] r : rList) {
        int y = r[0], x1 = r[1], x2 = r[2], sign = r[3];
        area = (area + (y - prevY) * prevLineSum) % MOD;
        prevY = y;

        // updates count of rectangles crossed by the current line
        prevLineSum = st.update(xOrdinality.get(x1), xOrdinality.get(x2), sign);
    }
    return (int)area;
}

class SegmentTreeNode {
    int start, end;     // [start, end]
    List<Integer> list;
    SegmentTreeNode left = null, right = null;
    int count = 0;      // count of rectangles covering the interval
    long sum = 0;       // sum of child intervals that are covered by some rectangle

    public SegmentTreeNode(int start, int end, List<Integer> list) {
        this.start = start;
        this.end = end;
        this.list = list;
    }

    private int getMid() {
        return (start + end) >>> 1;
    }

    private SegmentTreeNode getLeft() {
        return left = (left == null ? new SegmentTreeNode(start, getMid(), list) : left);
    }

    private SegmentTreeNode getRight() {
        return right = (right == null ? new SegmentTreeNode(getMid(), end, list) : right);
    }

    // Adds val to range [i, j]
    // returns sum of interval.counts
    public long update(int i, int j, int val) {
        if (i >= j) {
            return 0;
        }

        SegmentTreeNode l = getLeft(), r = getRight();
        if (start == i && end == j) {   // some rectangle covers the entire interval
            count += val;
        } else {
            // recursively updates child intervals
            l.update(i, Math.min(getMid(), j), val);
            r.update(Math.max(getMid(), i), j, val);
        }

        // if count > 0, then intervals between start and end will all be included
        // otherwise, recursively sums child intervals
        return sum = count > 0 ? list.get(end) - list.get(start) : l.sum + r.sum;
    }
}
{% endhighlight %}

[booking-concert-tickets-in-groups]: https://leetcode.com/problems/booking-concert-tickets-in-groups/
[longest-increasing-subsequence-ii]: https://leetcode.com/problems/longest-increasing-subsequence-ii/
[longest-substring-of-one-repeating-character]: https://leetcode.com/problems/longest-substring-of-one-repeating-character/
[rectangle-area-ii]: https://leetcode.com/problems/rectangle-area-ii/
