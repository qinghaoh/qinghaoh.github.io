---
title:  "Tree Recursion"
category: algorithm
tags: tree
---
# Fundamentals

The usual pattern of Tree DFS (or recursion) is:

```java
private T1 global;

public T2 processTree(TreeNode root) {
    // call dfs(root)
}

private T4 dfs(TreeNode node, T3 local) {
    // call dfs(node.left) and dfs(node.right)
} 
```

There are 3 core components:
1. Global variable (`T1 global`)
1. DFS parameter (`T3 local`)
1. DFS return value (`T4`)

# Global Variable

Global variables store the global state throughout the entire recursions. They are usually modified by each DFS call. Generally, there are two types of global variables: primitive/collection and structure.

## Primitive/Collection

A primitive global variable is usually used as a counter or state tracker (e.g., bit mask). A collection global variable is used to store related elements. 

[Convert BST to Greater Tree][convert-bst-to-greater-tree]

```java
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
```

[Distribute Coins in Binary Tree][distribute-coins-in-binary-tree]

```java
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
```

[Equal Tree Partition][equal-tree-partition]

```java
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
```

## Structure

A structure global variable is used for maintaining a data structure, like linked list. Traverse order usally plays an important role.

[Flatten Binary Tree to Linked List][flatten-binary-tree-to-linked-list]

```java
// the last node of the already formed linked list
private TreeNode last = null;

public void flatten(TreeNode root) {
    if (root == null) {
        return;
    }

    if (last != null) {
        last.left = null;
        last.right = root;
    }

    last = root;
    // memorizes the right child before the left subtree recursion,
    // which could modify the right child of root (last)
    TreeNode right = root.right;
    flatten(root.left);
    flatten(right);
}
```

[Flatten a Multilevel Doubly Linked List][flatten-a-multilevel-doubly-linked-list]

```java
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
```

[Minimum Absolute Difference in BST][minimum-absolute-difference-in-bst]

```java
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
```

[Find Bottom Left Tree Value][find-bottom-left-tree-value]

```c++
int bottomLeft = 0, depth = -1;

void dfs(TreeNode* node, int d) {
    if (depth < d) {
        bottomLeft = node->val;
        depth = d;
    }

    if (node->left != nullptr) {
        dfs(node->left, d + 1);
    }
    if (node->right != nullptr) {
        dfs(node->right, d + 1);
    }
}

public:
int findBottomLeftValue(TreeNode* root) {
    dfs(root, 0);
    return bottomLeft;
}
```

Another solution is right-to-left-BFS.

[Maximum Width of Binary Tree][maximum-width-of-binary-tree]

```java
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
```

# DFS Parameter

[Count Paths That Can Form a Palindrome in a Tree][count-paths-that-can-form-a-palindrome-in-a-tree]

```java
private List<Integer>[] tree;
private String s;

public long countPalindromePaths(List<Integer> parent, String s) {
    int n = parent.size();
    tree = new List[n];
    for (int i = 0; i < n; i++) {
        tree[i] = new ArrayList<>();
    }
    for (int i = 1; i < n; i++) {
        tree[parent.get(i)].add(i);
    }
    this.s = s;

    Map<Integer, Long> freqs = new HashMap<>();
    freqs.put(0, 1l);
    return dfs(0, freqs, 0);
}

private long dfs(int node, Map<Integer, Long> freqs, int mask) {
    long c = 0;
    if (node > 0) {
        mask ^= (1 << (s.charAt(node) - 'a'));
        for (int i = 0; i < 26; i++) {
            c += freqs.getOrDefault(mask ^ (1 << i), 0l);
        }
        long v = freqs.getOrDefault(mask, 0l);
        freqs.put(mask, v + 1);
        c += v;
    }

    for (int child : tree[node]) {
        c += dfs(child, freqs, mask);
    }
    return c;
}
```

# DFS Return Value

[House Robber III][house-robber-iii]

```java
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
```

[Maximum Score After Applying Operations on a Tree][maximum-score-after-applying-operations-on-a-tree]

```c++
// score, subtree sum
pair<long long, long long> dfs(vector<vector<int>>& tree, vector<int>& values, int node, int parent) {
    // When the subtree is rooted at leaf, the node can't be reset
    if (tree[node].size() == 1 && node > 0) {
        return {0, values[node]};
    }

    // pick: reset current node
    long long pick = values[node], sum = values[node];
    for (auto& neighbor : tree[node]) {
        if (neighbor != parent) {
            auto v = dfs(tree, values, neighbor, node);
            pick += v.first;
            sum += v.second;
        }
    }

    // sum - values[node]: keep current node
    return {max(pick, sum - values[node]), sum};
}

public:
long long maximumScoreAfterOperations(vector<vector<int>>& edges, vector<int>& values) {
    int n = values.size();
    vector<vector<int>> tree(n);
    for (auto& e : edges) {
        tree[e[0]].push_back(e[1]);
        tree[e[1]].push_back(e[0]);
    }

    return dfs(tree, values, 0, -1).first;
}
```

[Largest BST Subtree][largest-bst-subtree]

```java
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
```

[Split BST][split-bst]

```java
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
```

[Difference Between Maximum and Minimum Price Sum][difference-between-maximum-and-minimum-price-sum]

```java
private long maxCost;

public long maxOutput(int n, int[][] edges, int[] price) {
    List<Integer>[] tree = new List[n];
    for (int i = 0; i < n; i++) {
        tree[i] = new ArrayList<>();
    }
    for (int[] e : edges) {
        tree[e[0]].add(e[1]);
        tree[e[1]].add(e[0]);
    }

    // roots the tree at node 0
    dfs(tree, 0, -1, price);

    // min price sum is always the node itself
    // so for a particular node as root, its cost is the max of its child paths (excluding itself)
    // also, to maximize the cost, the root node must be a leaf node of the tree (degree == 1)
    return maxCost;
}

private long[] dfs(List<Integer>[] tree, int node, int parent, int[] price) {
    // postorder
    // curr[0]: max path sum from `node` to a leaf node
    // curr[1]: max path sum from `node` to the parent of a leaf node
    long[] curr = {price[node], 0};
    for (int child : tree[node]) {
        if (child != parent) {
            long[] next = dfs(tree, child, node, price);
            // to exclude the leaf node, there are two options:
            // curr[0] + next[1]: max sum among all previous child paths from `node` + `child` path without a leaf
            // curr[1] + next[0]: max sum among all previous child paths from `node` excluding leaves + `child` path with a leaf
            maxCost = Math.max(maxCost, Math.max(curr[0] + next[1], curr[1] + next[0]));
            curr[0] = Math.max(curr[0], next[0] + price[node]);
            curr[1] = Math.max(curr[1], next[1] + price[node]);
        }
    }
    return curr;
}
```

A more complex but general approach is by [rerooting](../subtree/#rerooting).

[Second Minimum Node in a Binary Tree][second-minimum-node-in-a-binary-tree]

```java
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
```

```java
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
```

# Top-down Local State

[Count Good Nodes in Binary Tree][count-good-nodes-in-binary-tree]

```java
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
```

[Maximum Difference Between Node and Ancestor][maximum-difference-between-node-and-ancestor]

```java
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
```

[Pseudo-Palindromic Paths in a Binary Tree][pseudo-palindromic-paths-in-a-binary-tree]

```java
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
```

[Reverse Odd Levels of Binary Tree][reverse-odd-levels-of-binary-tree]

```java
public TreeNode reverseOddLevels(TreeNode root) {
    dfs(root.left, root.right, 0);
    return root;
}

// node1 and node2 are reflection symmetrical
private void dfs(TreeNode node1, TreeNode node2, int level) {
    if (node1 == node2) {
        return;
    }

    if (level % 2 == 0) {
        int tmp = node1.val;
        node1.val = node2.val;
        node2.val = tmp;
    }

    dfs(node1.left, node2.right, level + 1);
    dfs(node1.right, node2.left, level + 1);
}
```

[Reorder Routes to Make All Paths Lead to the City Zero][reorder-routes-to-make-all-paths-lead-to-the-city-zero]

```java
private List<Integer>[] list;

public int minReorder(int n, int[][] connections) {
    this.list = new List[n];
    for (int i = 0; i < n; i++) {
        list[i] = new ArrayList<>();
    }

    for (int[] c : connections) {
        list[c[0]].add(c[1]);
        list[c[1]].add(-c[0]);
    }
    return dfs(0, 0);
}

private int dfs(int prev, int node) {
    int sum = 0;
    for (int next : list[node]) {
        if (Math.abs(next) != prev) {
            // next > 0 means the direction is node -> next
            sum += dfs(node, Math.abs(next)) + (next > 0 ? 1 : 0);
        }
    }
    return sum;
}
```

# Nested DFS

[Path Sum III][path-sum-iii]

```java
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
```

# Backtracking

[Path Sum III][path-sum-iii]

```java
public int pathSum(TreeNode root, int sum) {
    Map<Integer, Integer> prefixSum = new HashMap<>();
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
```

[All Nodes Distance K in Binary Tree][all-nodes-distance-k-in-binary-tree]

```java
private List<Integer> list = new ArrayList<>();
private Map<TreeNode, Integer> map = new HashMap<>();
private TreeNode target;
private int k;

public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
    this.target = target;
    this.k = k;

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
```

# Multiple DFS

[Sum of Distances in Tree][sum-of-distances-in-tree]

```java
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
```

# Postorder

In postorder, we don't have to pass parent node as a parameter of dfs().

[Delete Nodes And Return Forest][delete-nodes-and-return-forest]

```java
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
```

# Greedy

[Binary Tree Cameras][binary-tree-cameras]

```java
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
```

[Smallest Missing Genetic Value in Each Subtree][smallest-missing-genetic-value-in-each-subtree]

```java
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
```

[Number Of Ways To Reconstruct A Tree][number-of-ways-to-reconstruct-a-tree]

```java
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
```

## Dynamic Programming

[Minimum Edge Reversals So Every Node Is Reachable][minimum-edge-reversals-so-every-node-is-reachable]

```c++
    map<pair<int, int>, int> memo;

    int dfs(const vector<vector<int>> &graph, const vector<vector<int>> &revGraph, int node, int parent) {
        pair<int, int> key{node, parent};
        if (memo.contains(key))
        {
            return memo[key];
        }

        int ans = 0;
        for (auto &neighbor : graph[node]) {
            if (neighbor != parent)
            {
                ans += dfs(graph, revGraph, neighbor, node);
            }
        }

        for (auto &neighbor : revGraph[node]) {
            if (neighbor != parent)
            {
                ans += dfs(graph, revGraph, neighbor, node) + 1;
            }
        }
        return memo[key] = ans;
    }

public:
    vector<int> minEdgeReversals(int n, vector<vector<int>>& edges) {
        vector<vector<int>> graph(n);
        vector<vector<int>> revGraph(n);

        for (auto e : edges) {
            graph[e[0]].push_back(e[1]);
            revGraph[e[1]].push_back(e[0]);
        }

        vector<int> answer;
        for (int i = 0; i < n; i++) {
            answer.push_back(dfs(graph, revGraph, i, -1));
        }
        return answer;
    }
```

[all-nodes-distance-k-in-binary-tree]: https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/
[binary-tree-cameras]: https://leetcode.com/problems/binary-tree-cameras/
[convert-bst-to-greater-tree]: https://leetcode.com/problems/convert-bst-to-greater-tree/
[count-good-nodes-in-binary-tree]: https://leetcode.com/problems/count-good-nodes-in-binary-tree/
[count-paths-that-can-form-a-palindrome-in-a-tree]: https://leetcode.com/problems/count-paths-that-can-form-a-palindrome-in-a-tree/
[delete-nodes-and-return-forest]: https://leetcode.com/problems/delete-nodes-and-return-forest/
[difference-between-maximum-and-minimum-price-sum]: https://leetcode.com/problems/difference-between-maximum-and-minimum-price-sum/
[distribute-coins-in-binary-tree]: https://leetcode.com/problems/distribute-coins-in-binary-tree/
[equal-tree-partition]: https://leetcode.com/problems/equal-tree-partition/
[find-bottom-left-tree-value]: https://leetcode.com/problems/find-bottom-left-tree-value/
[flatten-a-multilevel-doubly-linked-list]: https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/
[flatten-binary-tree-to-linked-list]: https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
[house-robber-iii]: https://leetcode.com/problems/house-robber-iii/
[largest-bst-subtree]: https://leetcode.com/problems/largest-bst-subtree/
[maximum-difference-between-node-and-ancestor]: https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/
[maximum-score-after-applying-operations-on-a-tree]: https://leetcode.com/problems/maximum-score-after-applying-operations-on-a-tree/
[maximum-width-of-binary-tree]: https://leetcode.com/problems/maximum-width-of-binary-tree/
[minimum-absolute-difference-in-bst]: https://leetcode.com/problems/minimum-absolute-difference-in-bst/
[minimum-edge-reversals-so-every-node-is-reachable]: https://leetcode.com/problems/minimum-edge-reversals-so-every-node-is-reachable/
[number-of-ways-to-reconstruct-a-tree]: https://leetcode.com/problems/number-of-ways-to-reconstruct-a-tree/
[path-sum-iii]: https://leetcode.com/problems/path-sum-iii/
[pseudo-palindromic-paths-in-a-binary-tree]: https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/
[reorder-routes-to-make-all-paths-lead-to-the-city-zero]: https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/
[reverse-odd-levels-of-binary-tree]: https://leetcode.com/problems/reverse-odd-levels-of-binary-tree/
[second-minimum-node-in-a-binary-tree]: https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/
[smallest-missing-genetic-value-in-each-subtree]: https://leetcode.com/problems/smallest-missing-genetic-value-in-each-subtree/
[split-bst]: https://leetcode.com/problems/split-bst/
[sum-of-distances-in-tree]: https://leetcode.com/problems/sum-of-distances-in-tree/
