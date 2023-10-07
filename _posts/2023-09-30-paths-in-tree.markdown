---
title:  "Paths in Tree"
category: algorithm
tags: tree
---

[Binary Tree Maximum Path Sum][binary-tree-maximum-path-sum]

```java
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
```

[Longest Univalue Path][longest-univalue-path]

```java
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
```

[Longest ZigZag Path in a Binary Tree][longest-zigzag-path-in-a-binary-tree]

```java
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
```

## Leaf-to-leaf Paths

Aggregate paths from current node to all leaf nodes in its subtree.

[Diameter of Binary Tree][diameter-of-binary-tree]

```java
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
```

Similar problem: [Diameter of N-Ary Tree][diameter-of-n-ary-tree]

[Number of Good Leaf Nodes Pairs][number-of-good-leaf-nodes-pairs]

```java
private int distance = 0;
private int pairs = 0;

public int countPairs(TreeNode root, int distance) {
    this.distance = distance;
    dfs(root);
    return pairs;
}

// arr[i]: the number of leaf nodes whose distances to the current node is (i - 1)
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

    // Prefix sum of right
    int[] p = new int[right.length];
    for (int i = 0; i < distance; i++) {
        p[i + 1] = p[i] + right[i];
    }

    for (int i = 1; i < distance; i++) {
        // p[distance - i + 1] = sum(right[0]...right[distance - 1])
        // i.e. count of all right nodes where left[i] + right[j] <= distance
        pairs += left[i] * p[distance - i + 1];
    }

    for (int i = 1; i < distance; i++) {
        count[i + 1] = left[i] + right[i];
    }

    return count;
}
```

[binary-tree-maximum-path-sum]: https://leetcode.com/problems/binary-tree-maximum-path-sum/
[diameter-of-binary-tree]: https://leetcode.com/problems/diameter-of-binary-tree/
[diameter-of-n-ary-tree]: https://leetcode.com/problems/diameter-of-n-ary-tree/
[longest-univalue-path]: https://leetcode.com/problems/longest-univalue-path/
[longest-zigzag-path-in-a-binary-tree]: https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/
[number-of-good-leaf-nodes-pairs]: https://leetcode.com/problems/number-of-good-leaf-nodes-pairs/

