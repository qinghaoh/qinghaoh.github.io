---
layout: post
title:  "Tree Recursion"
tags: tree
---
# Fundamentals

The usual pattern of Tree DFS (or recursion) is:

{% highlight java %}
private T1 global;

public T2 processTree(TreeNode root) {
    // call dfs(root)
}

private T4 dfs(TreeNode node, T3 local) {
    // call dfs(node.left) and dfs(node.right)
} 
{% endhighlight %}

There are 3 core components:
1. Global variable (`T1 global`)
1. DFS parameter (`T3 local`)
1. DFS return value (`T4`)

[Convert BST to Greater Tree][convert-bst-to-greater-tree]

{% highlight java %}
private int sum = 0;

public TreeNode convertBST(TreeNode root) {
    if (root != null) {
        convertBST(root.right);
        sum += root.val;
        root.val = sum;
        convertBST(root.left);
    }
    return root;
}
{% endhighlight %}

[Distribute Coins in Binary Tree][distribute-coins-in-binary-tree]

{% highlight java %}
private int move = 0;

public int distributeCoins(TreeNode root) {
    dfs(root);
    return move;
}

// excess of the subtree (#coins - #nodes)
private int dfs(TreeNode node) {
    if (node == null) {
        return 0;
    }

    int left = dfs(node.left), right = dfs(node.right);
    move += Math.abs(left) + Math.abs(right);
    return left + right + node.val - 1;
}
{% endhighlight %}

[Diameter of Binary Tree][diameter-of-binary-tree]

{% highlight java %}
private int diameter = 0;

public int diameterOfBinaryTree(TreeNode root) {
    height(root);
    return diameter;
}

private int height(TreeNode node) {
    if (node == null) {
        return 0;
    }

    int left = height(node.left), right = height(node.right);
    diameter = Math.max(diameter, left + right);
    return Math.max(left, right) + 1;
}
{% endhighlight %}

Similar problem: [Diameter of N-Ary Tree][diameter-of-n-ary-tree]

[Binary Tree Maximum Path Sum][binary-tree-maximum-path-sum]

{% highlight java %}
private int sum = Integer.MIN_VALUE;

public int maxPathSum(TreeNode root) {
    dfs(root);
    return sum;
}

/**
 * Max sum of paths which start from node.
 * @param node the root of the subtree
 * @return the max sum of paths starting from node
 */
private int dfs(TreeNode node) {
    if (node == null) {
        return 0;
    }

    int left = Math.max(0, dfs(node.left)), right = Math.max(0, dfs(node.right));

    sum = Math.max(sum, node.val + left + right);
    return node.val + Math.max(left, right);
}
{% endhighlight %}

[Longest Univalue Path][longest-univalue-path]

{% highlight java %}
private int path = 0;

public int longestUnivaluePath(TreeNode root) {
    length(root);
    return path;
}

/**
 * Max length of univalue paths which start from node.
 * @param node the root of the subtree
 * @return the max length of univalue paths starting from node
 */
private int length(TreeNode node) {
    if (node == null) {
        return 0;
    }

    // max univalue paths from the child nodes
    int left = length(node.left), right = length(node.right);

    // max univalue paths from the current node
    int leftPath = 0, rightPath = 0;
    if (node.left != null && node.left.val == node.val) {
        leftPath = left + 1;
    }
    if (node.right != null && node.right.val == node.val) {
        rightPath = right + 1;
    }

    path = Math.max(path, leftPath + rightPath);
    return Math.max(leftPath, rightPath);
}
{% endhighlight %}

[Minimum Absolute Difference in BST][minimum-absolute-difference-in-bst]

{% highlight java %}
private int d = Integer.MAX_VALUE;
private TreeNode prev = null;

public int getMinimumDifference(TreeNode root) {
    if (root == null) {
        return Integer.MAX_VALUE;
    }

    getMinimumDifference(root.left);

    // inorder
    if (prev != null) {
        d = Math.min(d, root.val - prev.val);
    }
    prev = root;

    getMinimumDifference(root.right);

    return d;
}
{% endhighlight %}

[Flatten a Multilevel Doubly Linked List][flatten-a-multilevel-doubly-linked-list]

{% highlight java %}
private Node tail = null;

public Node flatten(Node head) {
    if (head == null) {
        return null;
    }

    head.prev = tail;
    tail = head;

    Node next = head.next;

    head.next = flatten(head.child);
    head.child = null;

    tail.next = flatten(next);

    return head;
}
{% endhighlight %}

[Equal Tree Partition][equal-tree-partition]

{% highlight java %}
private Set<Integer> set = new HashSet<>();

public boolean checkEqualTree(TreeNode root) {
    int sum = root.val + dfs(root.left) + dfs(root.right);
    // the root sum is not added to the set
    return sum % 2 == 0 && set.contains(sum / 2);
}

private int dfs(TreeNode node) {
    // 0 from null node is not added to the set
    // so trees like [0], [0, 1, -1] will return correct answer
    if (node == null) {
        return 0;
    }

    int sum = node.val + dfs(node.left) + dfs(node.right);
    set.add(sum);
    return sum;
}
{% endhighlight %}

[Split BST][split-bst]

{% highlight java %}
public TreeNode[] splitBST(TreeNode root, int V) {
    TreeNode[] result = new TreeNode[2];
    if (root != null) {
        if (root.val <= V) {
            result = splitBST(root.right, V);
            root.right = result[0];
            result[0] = root;
        } else {
            result = splitBST(root.left, V);
            root.left = result[1];
            result[1] = root;
        }
    }
    return result;
}
{% endhighlight %}

[Find Bottom Left Tree Value][find-bottom-left-tree-value]

{% highlight java %}
private int bottomLeft = 0;
private int depth = -1;

public int findBottomLeftValue(TreeNode root) {
    dfs(root, 0);
    return bottomLeft;
}

private void dfs(TreeNode node, int d) {
    if (depth < d) {
        bottomLeft = node.val;
        depth = d;
    }

    if (node.left != null) {
        dfs(node.left, d + 1);
    }
    if (node.right != null) {
        dfs(node.right, d + 1);
    }

    return;
}
{% endhighlight %}

Another solution is right-to-left-BFS.

[Maximum Width of Binary Tree][maximum-width-of-binary-tree]

{% highlight java %}
public int widthOfBinaryTree(TreeNode root) {
    return dfs(root, 0, 0, new ArrayList<>());
}

private int dfs(TreeNode node, int level, int index, List<Integer> starts) {
    if (node == null) {
        return 0;
    }

    // the first node visited in this level
    if (starts.size() == level) {
        starts.add(index);
    }

    int curr = index - starts.get(level) + 1;
    int left = dfs(node.left, level + 1, index * 2 + 1, starts);
    int right = dfs(node.right, level + 1, index * 2 + 2, starts);
    return Math.max(curr, Math.max(left, right));
}
{% endhighlight %}

# Return Value of DFS

[House Robber III][house-robber-iii]

{% highlight java %}
public int rob(TreeNode root) {
    int[] dp = dfs(root);
    return Math.max(dp[0], dp[1]);
}

private int[] dfs(TreeNode root) {
    // {not robbed, robbed}
    int[] dp = new int[2];
    if (root == null) {
        return dp;
    }

    int[] left = dfs(root.left), right = dfs(root.right);

    dp[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
    dp[1] = root.val + left[0] + right[0];

    return dp;
}
{% endhighlight %}

[Largest BST Subtree][largest-bst-subtree]

{% highlight java %}
public int largestBSTSubtree(TreeNode root) {
    return dfs(root)[0];
}

private int[] dfs(TreeNode node) {
    if (node == null) {
        // [0]: number of nodes of the largest BST in this subtree
        // [1]: min value of the subtree
        // [2]: max value of the subtree
        // it's a hack to assign null node in this way
        // so any node can be a valid parent of it to form a BST 
        return new int[]{0, Integer.MAX_VALUE, Integer.MIN_VALUE};
    }

    int[] left = dfs(node.left), right = dfs(node.right);
    return node.val > left[2] && node.val < right[1] ?
        // valid BST
        // [1] is min(node.val, left[1]) rather than left[1] to handle the corner case
        // when left child node is null
        // same for [2]
        new int[]{1 + left[0] + right[0], Math.min(node.val, left[1]), Math.max(node.val, right[2])} :
        // invalid BST
        // assign min and max in this way so no node can be a valid parent of it to form a BST
        new int[]{Math.max(left[0], right[0]), Integer.MIN_VALUE, Integer.MAX_VALUE};
}
{% endhighlight %}

[Longest ZigZag Path in a Binary Tree][longest-zigzag-path-in-a-binary-tree]

{% highlight java %}
public int longestZigZag(TreeNode root) {
    return dfs(root)[2];
}

private int[] dfs(TreeNode node) {
    if (node == null) {
        // max zigzag path at:
        // [0]: left child node
        // [1]: right child node
        // [2]: this node
        return new int[]{-1, -1, -1};
    }

    int[] left = dfs(node.left), right = dfs(node.right);

    // left[1], right[0] makes the path zigzag
    // leaves will have zigzag path == -1 + 1 == 0
    int path = Math.max(Math.max(left[1], right[0]) + 1, Math.max(left[2], right[2]));

    return new int[]{left[1] + 1, right[0] + 1, path};
}
{% endhighlight %}

[Number of Good Leaf Nodes Pairs][number-of-good-leaf-nodes-pairs]

{% highlight java %}
private int distance = 0;
private int pairs = 0;

public int countPairs(TreeNode root, int distance) {
    this.distance = distance;
    dfs(root);
    return pairs;
}

/**
 * Gets the leaf node count array. Here's the definition for the array:
 *   The leaf node count array contains (distance + 1) elements. The i-th element
 *   denotes the count of leaf nodes which are i away from current node's parent.
 *   The 0-th element of the array is not used.
 * @param node node the root of the subtree
 * @return left node count array
 */
private int[] dfs(TreeNode node) {
    int[] count = new int[distance + 1];
    if (node == null) {
        return count;
    }

    if (node.left == node.right) {
        count[1] = 1;
        return count;
    }

    int[] left = dfs(node.left), right = dfs(node.right);

    // prefix sum of right
    int[] p = new int[right.length];
    for (int i = 0; i < distance; i++) {
        p[i + 1] = p[i] + right[i];
    }

    for (int i = 1; i < distance; i++) {
        // p[distance - i + 1] = right[0] + right[1] + ... + right[distance - i]
        // i.e. count of all right nodes where left[i] + right[j] <= distance
        pairs += left[i] * p[distance - i + 1];
    }

    for (int i = 1; i < distance; i++) {
        count[i + 1] = left[i] + right[i];
    }

    return count;
}
{% endhighlight %}

[Second Minimum Node in a Binary Tree][second-minimum-node-in-a-binary-tree]

{% highlight java %}
private int min;
private long secondMin = Long.MAX_VALUE;

public int findSecondMinimumValue(TreeNode root) {
    // root has the minimum value
    min = root.val;
    dfs(root);
    return secondMin < Long.MAX_VALUE ? (int)secondMin : -1;
}

private void dfs(TreeNode node) {
    if (node == null) {
        return;
    }

    if (min < node.val && node.val < secondMin) {
        secondMin = node.val;
    } else if (min == node.val) {
        dfs(node.left);
        dfs(node.right);
    }
}
{% endhighlight %}

{% highlight java %}
public int findSecondMinimumValue(TreeNode root) {
    // tree leaf
    if (root.left == null) {
        return -1;
    }

    // The second minimum value on left path including root
    int left = root.left.val == root.val ? findSecondMinimumValue(root.left) : root.left.val;
    // The second minimum value on right path including root
    int right = root.right.val == root.val ? findSecondMinimumValue(root.right) : root.right.val;

    // if left == -1 && right == -1, returns -1
    // else left == -1 || right == -1, returns none -1
    // else returns the lesser of the two second minimum values
    return left == -1 || right == -1 ? Math.max(left, right) : Math.min(left, right);
}
{% endhighlight %}

# Level Order Traversal

The most intuitive way is BFS:

{% highlight java %}
List<Integer> currLevel = new ArrayList<>();
for (int i = q.size(); i > 0; i--) {
    Node node = q.poll();
    currLevel.add(node.val);
    for (Node child : node.children) {
        q.offer(child);
    }
}
{% endhighlight %}

However, it can be implemented with DFS as well:

[N-ary Tree Level Order Traversal][n-ary-tree-level-order-traversal]

{% highlight java %}
private List<List<Integer>> list;

public List<List<Integer>> levelOrder(Node root) {
    list = new ArrayList<>();
    dfs(root, 0);
    return list;
}

private void dfs(Node node, int level) {
    if (node == null) {
        return;
    }

    if (list.size() == level) {
        list.add(new ArrayList<>());
    }
    list.get(level).add(node.val);

    for (Node child : node.children) {
        dfs(child, level + 1);
    }
}
{% endhighlight %}

[Find Leaves of Binary Tree][find-leaves-of-binary-tree]

{% highlight java %}
private List<List<Integer>> leaves = new ArrayList<>();

public List<List<Integer>> findLeaves(TreeNode root) {
    dfs(root);
    return leaves;
}

private int height(TreeNode node) {
    if (node == null) {
        return -1;
    }

    int h = Math.max(height(node.left), height(node.right)) + 1;

    if (h == leaves.size()) {
        leaves.add(new ArrayList<>());
    }
    leaves.get(h).add(node.val);

    return h;
}
{% endhighlight %}

[Binary Tree Right Side View][binary-tree-right-side-view]

{% highlight java %}
private List<Integer> rightside;

public List<Integer> rightSideView(TreeNode root) {
    this.rightside = new ArrayList<>();

    dfs(root, 0);
    return rightside;
}

public void dfs(TreeNode node, int level) {
    if (node == null) {
        return;
    }

    if (rightside.size() == level) {
        rightside.add(node.val);
    } 

    dfs(node.right, level + 1);
    dfs(node.left, level + 1);
}
{% endhighlight %}

[Increasing Order Search Tree][increasing-order-search-tree]

{% highlight java %}
private TreeNode curr;  // current node of the list

public TreeNode increasingBST(TreeNode root) {
    TreeNode head = new TreeNode(0);
    curr = head;
    inorder(root);
    return head.right;
}

private void inorder(TreeNode node) {
    if (node == null) {
        return;
    }

    inorder(node.left);
    node.left = null;
    curr.right = node;
    curr = node;
    inorder(node.right);
}
{% endhighlight %}

{% highlight java %}
public TreeNode increasingBST(TreeNode root) {
    return inorder(root, null);
}

/**
 * Inorder traverses and rearranges the tree.
 * @param node current tree node
 * @param next next ancestor node in inorder traversal
 * @return head of the list after rearrangement
 */
public TreeNode inorder(TreeNode node, TreeNode next) {
    // If the current node is a left child, next will be its parent
    // else if the current node is a right child, next will be its "leftmost" parent's parent
    if (node == null) {
        return next;
    }

    TreeNode left = inorder(node.left, node);
    node.left = null;
    // If node.right == 0, it links the next ancesotr to the rearranged right list
    // otherwise it links the rearranged right list to the current node
    node.right = inorder(node.right, next);
    return left;
}
{% endhighlight %}

# Top-down Local State

[Count Good Nodes in Binary Tree][count-good-nodes-in-binary-tree]

{% highlight java %}
public int goodNodes(TreeNode root) {
    return dfs(root, root.val);
}

private int dfs(TreeNode node, int max) {
    if (node == null) {
        return 0;
    }

    int count = node.val >= max ? 1 : 0;

    max = Math.max(max, node.val);
    count += dfs(node.left, max);
    count += dfs(node.right, max);

    return count;
}
{% endhighlight %}

[Maximum Difference Between Node and Ancestor][maximum-difference-between-node-and-ancestor]

{% highlight java %}
private int diff = 0;

public int maxAncestorDiff(TreeNode root) {
    dfs(root, root.val, root.val);
    return diff;
}

private void dfs(TreeNode node, int min, int max) {
    if (node == null) {
        return;
    }

    min = Math.min(min, node.val);
    max = Math.max(max, node.val);
    diff = Math.max(diff, max - min);

    dfs(node.left, min, max);
    dfs(node.right, min, max);
}
{% endhighlight %}

[Pseudo-Palindromic Paths in a Binary Tree][pseudo-palindromic-paths-in-a-binary-tree]

{% highlight java %}
public int pseudoPalindromicPaths (TreeNode root) {
    return dfs(root, 0);
}

/**
 * Count pseudo palindromic paths by DFS.
 * In a pseudo palindromic path, each digit appears at most once.
 * @param node current node
 * @param vector an integer whose i-th bit indicates the presence of digit i
 * @return count of pseudo palinddromic paths
 */
private int dfs(TreeNode node, int vector) {
    if (node == null) {
        return 0;
    }

    vector ^= 1 << (node.val - 1);
    int count = dfs(node.left, vector) + dfs(node.right, vector);

    // leaf node
    if (node.left == node.right && (vector & (vector - 1)) == 0) {
        count++;
    }

    return count;
}
{% endhighlight %}

# Nested DFS

[Path Sum III][path-sum-iii]

{% highlight java %}
public int pathSum(TreeNode root, int sum) {
    if (root == null) {
        return 0;
    }

    return pathSumFromNode(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
}

/**
 * Counts paths with target sum which start from node.
 * @param node the root of the subtree
 * @param sum target sum
 * @return the count of paths with target sum which start from node
 */
private int pathSumFromNode(TreeNode node, int sum) {
    if (node == null) {
        return 0;
    }

    return (sum == node.val ? 1 : 0) + pathSumFromNode(node.left, sum - node.val) + pathSumFromNode(node.right, sum - node.val);
}
{% endhighlight %}

# Backtracking

[Path Sum III][path-sum-iii]

{% highlight java %}
public int pathSum(TreeNode root, int sum) {
    Map<Integer, Integer> prefixSum = new HashMap();
    prefixSum.put(0,1);
    return dfs(root, 0, sum, prefixSum);
}

public int dfs(TreeNode node, int currSum, int target, Map<Integer, Integer> prefixSum) {
    if (node == null) {
        return 0;
    }

    currSum += node.val;
    int count = prefixSum.getOrDefault(currSum - target, 0);

    // backtracking
    prefixSum.compute(currSum, (k, v) -> v == null ? 1 : v + 1);
    count += dfs(node.left, currSum, target, prefixSum) + dfs(node.right, currSum, target, prefixSum);
    prefixSum.compute(currSum, (k, v) -> v - 1);

    return count;
}
{% endhighlight %}

[All Nodes Distance K in Binary Tree][all-nodes-distance-k-in-binary-tree]

{% highlight java %}
private List<Integer> list = new ArrayList<>();
private Map<TreeNode, Integer> map = new HashMap<>();
private TreeNode target;
private int k;

public List<Integer> distanceK(TreeNode root, TreeNode target, int K) {
    this.target = target;
    this.k = K;

    distance(root);
    dfs(root, map.get(root));
    return list;
}

/**
 * Returns the distance from node to target if the node is on the path from root to target;
 * otherwise returns -1.
 * @param node a node in the tree
 * @return distance from node to target if node is on the path from root to target, otherwise -1
 */
private int distance(TreeNode node) {
    if (node == null) {
        return -1;
    }

    if (node == target) {
        map.put(node, 0);
        return 0;
    }

    int left = distance(node.left);
    if (left >= 0) {
        map.put(node, left + 1);
        return left + 1;
    }

    int right = distance(node.right);
    if (right >= 0) {
        map.put(node, right + 1);
        return right + 1;
    }
    return -1;
}

private void dfs(TreeNode node, int d) {
    if (node == null) {
        return;
    }

    if (map.containsKey(node)) {
        d = map.get(node);
    }

    if (d == k) {
        list.add(node.val);
    }

    dfs(node.left, d + 1);
    dfs(node.right, d + 1);
}
{% endhighlight %}

# Multiple DFS

[Sum of Distances in Tree][sum-of-distances-in-tree]

{% highlight java %}
private List<List<Integer>> tree = new ArrayList<>();
// count[i]: count of all nodes in the subtree i
private int[] answer, count;

public int[] sumOfDistancesInTree(int n, int[][] edges) {
    this.answer = new int[n];
    this.count = new int[n];

    for (int i = 0; i < n; i++) {
        tree.add(new ArrayList<>());
    }

    for (int[] e : edges) {
        tree.get(e[0]).add(e[1]);
        tree.get(e[1]).add(e[0]);
    }

    dfs(0, -1);
    dfs2(0, -1);

    return answer;
}

// postorder, computes count[] and answer[]
private void dfs(int node, int parent) {
    for (int child : tree.get(node)) {
        if (child == parent) {
            continue;
        }

        dfs(child, node);
        count[node] += count[child];
        // sum of distances between this node and all the other nodes in the subtree
        answer[node] += answer[child] + count[child];
    }
    count[node]++;
}

// preorder, updates answer[]
private void dfs2(int node, int parent) {
    for (int child : tree.get(node)) {
        if (child == parent) {
            continue;
        }

        // when we move node to child:
        // * count[child] nodes get 1 closer to node
        // * n - count[i] nodes get 1 further to node
        answer[child] = answer[node] - count[child] + count.length - count[child];
        dfs2(child, node);
    }
}
{% endhighlight %}

# Postorder

In postorder, we don't have to pass parent node as a parameter of dfs().

[Delete Nodes And Return Forest][delete-nodes-and-return-forest]

{% highlight java %}
private List<TreeNode> list = new ArrayList<>();
private Set<Integer> set = new HashSet<>();

public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
    for (int i : to_delete) {
        set.add(i);
    }

    if (!set.contains(root.val)) {
        list.add(root);
    }

    dfs(root);
    return list;
}

private TreeNode dfs(TreeNode node) {
    if (node == null) {
        return null;
    }

    node.left = dfs(node.left);
    node.right = dfs(node.right);

    // postorder
    if (set.contains(node.val)) {
        if (node.left != null) {
            list.add(node.left);
        }
        if (node.right != null) {
            list.add(node.right);
        }
        return null;
    }

    return node;
}
{% endhighlight %}

# Greedy

[Binary Tree Cameras][binary-tree-cameras]

{% highlight java %}
private int count = 0;

private enum Camera {
    NOT_MONITORED,  // not monitored
    HAS_CAMERA,     // has camera
    MONITORED       // monitored, no camera
};

/*Return 0 if it's a leaf.
Return 1 if it's a parent of a leaf, with a camera on this node.
Return 2 if it's coverd, without a camera on this node.*/

public int minCameraCover(TreeNode root) {
    // installs cameras on parents of all leaves
    // then removes all monitored nodes

    // installs a camera at root if it's not monitored
    return dfs(root) == Camera.NOT_MONITORED ? ++count : count;
}

public Camera dfs(TreeNode root) {
    // if there's no node, it's already monitored
    if (root == null) {
        return Camera.MONITORED;
    }

    Camera left = dfs(root.left), right = dfs(root.right);

    // if either child node is not monitored,
    // installs a camera at the current node
    if (left == Camera.NOT_MONITORED || right == Camera.NOT_MONITORED) {
        count++;
        return Camera.HAS_CAMERA;
    }

    return left == Camera.HAS_CAMERA || right == Camera.HAS_CAMERA ? Camera.MONITORED : Camera.NOT_MONITORED;
}
{% endhighlight %}

[Smallest Missing Genetic Value in Each Subtree][smallest-missing-genetic-value-in-each-subtree]

{% highlight java %}
private Map<Integer, List<Integer>> tree = new HashMap<>();
private Set<Integer> set = new HashSet<>();

public int[] smallestMissingValueSubtree(int[] parents, int[] nums) {
    int n = parents.length;
    for (int i = 0; i < n; i++) {
        tree.computeIfAbsent(parents[i], k -> new ArrayList<>()).add(i);
    }

    // only the node with genetic value 1 and its ancestors have missing values > 1
    int miss = 1;
    int[] ans = new int[n];
    Arrays.fill(ans, miss);

    // finds the node with genetic value 1
    int node = -1;
    for (int i = 0; i < n; i++) {
        if (nums[i] == 1) {
            node = i;
            break;
        }
    }
    if (node < 0) {
        return ans;
    }

    int prev = -1;
    while (node >= 0) {
        if (tree.containsKey(node)) {
            for (int child : tree.get(node)) {
                // skips previously visited child
                if (child != prev) {
                    dfs(nums, child);
                }
            }
        }

        set.add(nums[node]);
        // finds next missing genetic value
        while (set.contains(miss)) {
            miss++;
        }
        ans[node] = miss;

        prev = node;
        // goes up by one node
        node = parents[node];
    }
    return ans;
}

// adds all descendants to the set
private void dfs(int[] nums, int node) {
    set.add(nums[node]);
    if (tree.containsKey(node)) {
        for (int child : tree.get(node)) {
            dfs(nums, child);
        }
    }
}
{% endhighlight %}

[Number Of Ways To Reconstruct A Tree][number-of-ways-to-reconstruct-a-tree]

{% highlight java %}
public int checkWays(int[][] pairs) {
    // builds graph
    Map<Integer, Set<Integer>> graph = new HashMap<>();
    for (int[] p : pairs) {
        graph.computeIfAbsent(p[0], v -> new HashSet<>()).add(p[1]);
        graph.computeIfAbsent(p[1], v -> new HashSet<>()).add(p[0]);
    }

    // {node, degree}
    // the degree is of the graph node, not of the tree node
    Queue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> -a[1]));
    for (var e : graph.entrySet()) {
        pq.offer(new int[]{e.getKey(), e.getValue().size()});
    }

    // number of nodes
    int n = graph.size();
    Set<Integer> visited = new HashSet<>();
    boolean isMultiple = false;

    // selects the node with the greatest graph degree greedily
    while (!pq.isEmpty()) {
        int[] curr = pq.poll();

        // a node's parent's pairs always contain all of the node's pairs
        // so parent's degree is always >= the child node's degree
        //
        // we are processing from max degree to min in descending order
        // so the already visited neighbors are the node's ancestors
        // and the one with least degree is its parent
        int parent = 0, minDegree = n;
        for (int neighbor : graph.get(curr[0])) {
            if (visited.contains(neighbor) && graph.get(neighbor).size() < minDegree) {
                parent = neighbor;
                minDegree = graph.get(neighbor).size();
            }
        }

        visited.add(curr[0]);

        // current node is a root candidate (parent == 0)
        if (parent == 0) {
            // if the node has degree < n - 1, there's no root
            if (curr[1] != n - 1) {
                return 0;
            }
            continue;
        }

        // parent's pairs must contain the current node's neighbor
        for (int neighbor : graph.get(curr[0])) {
            if (neighbor != parent && !graph.get(parent).contains(neighbor)) {
                return 0;
            }
        }

        // parent's degree = current node's degree
        if (minDegree == curr[1]) {
            isMultiple = true;
        }
    }

    return isMultiple ? 2 : 1;
}
{% endhighlight %}

[all-nodes-distance-k-in-binary-tree]: https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/
[binary-tree-cameras]: https://leetcode.com/problems/binary-tree-cameras/
[binary-tree-maximum-path-sum]: https://leetcode.com/problems/binary-tree-maximum-path-sum/
[binary-tree-right-side-view]: https://leetcode.com/problems/binary-tree-right-side-view/
[convert-bst-to-greater-tree]: https://leetcode.com/problems/convert-bst-to-greater-tree/
[count-good-nodes-in-binary-tree]: https://leetcode.com/problems/count-good-nodes-in-binary-tree/
[delete-nodes-and-return-forest]: https://leetcode.com/problems/delete-nodes-and-return-forest/
[diameter-of-binary-tree]: https://leetcode.com/problems/diameter-of-binary-tree/
[diameter-of-n-ary-tree]: https://leetcode.com/problems/diameter-of-n-ary-tree/
[distribute-coins-in-binary-tree]: https://leetcode.com/problems/distribute-coins-in-binary-tree/
[equal-tree-partition]: https://leetcode.com/problems/equal-tree-partition/
[find-bottom-left-tree-value]: https://leetcode.com/problems/find-bottom-left-tree-value/
[find-leaves-of-binary-tree]: https://leetcode.com/problems/find-leaves-of-binary-tree/
[flatten-a-multilevel-doubly-linked-list]: https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/
[house-robber-iii]: https://leetcode.com/problems/house-robber-iii/
[increasing-order-search-tree]: https://leetcode.com/problems/increasing-order-search-tree/
[largest-bst-subtree]: https://leetcode.com/problems/largest-bst-subtree/
[longest-univalue-path]: https://leetcode.com/problems/longest-univalue-path/
[longest-zigzag-path-in-a-binary-tree]: https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/
[maximum-difference-between-node-and-ancestor]: https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/
[maximum-width-of-binary-tree]: https://leetcode.com/problems/maximum-width-of-binary-tree/
[minimum-absolute-difference-in-bst]: https://leetcode.com/problems/minimum-absolute-difference-in-bst/
[n-ary-tree-level-order-traversal]: https://leetcode.com/problems/n-ary-tree-level-order-traversal/
[number-of-good-leaf-nodes-pairs]: https://leetcode.com/problems/number-of-good-leaf-nodes-pairs/
[number-of-ways-to-reconstruct-a-tree]: https://leetcode.com/problems/number-of-ways-to-reconstruct-a-tree/
[path-sum-iii]: https://leetcode.com/problems/path-sum-iii/
[pseudo-palindromic-paths-in-a-binary-tree]: https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/
[second-minimum-node-in-a-binary-tree]: https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/
[smallest-missing-genetic-value-in-each-subtree]: https://leetcode.com/problems/smallest-missing-genetic-value-in-each-subtree/
[split-bst]: https://leetcode.com/problems/split-bst/
[sum-of-distances-in-tree]: https://leetcode.com/problems/sum-of-distances-in-tree/
