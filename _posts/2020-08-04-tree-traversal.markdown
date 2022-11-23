---
layout: post
title:  "Tree Traversal"
tags: tree
---
# Traversal
## Preorder
[Binary Tree Preorder Traveral][binary-tree-preorder-traversal]

### Recursion
{% highlight java %}
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    helper(root, list);
    return list;
}

private void helper(TreeNode root, List<Integer> list) {
    if (root != null) {
        list.add(root.val);  // preorder
        helper(root.left, list);
        helper(root.right, list);
    }
}
{% endhighlight %}

### Stack
{% highlight java %}
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode node = root;
    while (node != null) {
        list.add(node.val);  // preorder
        if (node.right != null) {
            stack.push(node.right);
        }             
        node = node.left;
        if (node == null && !stack.isEmpty()) {
            node = stack.pop();
        }
    }
    return list;
}
{% endhighlight %}

### Morris

## Inorder
[Binary Tree Inorder Traveral][binary-tree-inorder-traversal]

### Recursion
{% highlight java %}
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    helper(root, list);
    return list;
}

private void helper(TreeNode root, List<Integer> list) {
    if (root != null) {
        helper(root.left, list);
        list.add(root.val);  // inorder
        helper(root.right, list);
    }
}
{% endhighlight %}

### Stack
{% highlight java %}
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode node = root;
    while (node != null || !stack.isEmpty()) {
        if (node != null) {
            stack.push(node);
            node = node.left;
        } else {
            node = stack.pop();
            list.add(node.val);  // inorder
            node = node.right;
        }
    }
    return list;  
}
{% endhighlight %}

## Postorder
[Binary Tree Postorder Traveral][binary-tree-postorder-traversal]

### Recursion
{% highlight java %}
public List<Integer> postorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    traverse(root, list);
    return list;
}

private void traverse(TreeNode root, List<Integer> list) {
    if (root != null) {
        traverse(root.left, list);
        traverse(root.right, list);
        list.add(root.val);
    }
}
{% endhighlight %}

### Stack
{% highlight java %}
public List<Integer> postorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode node = root;
    while (node != null) {
        list.add(node.val);
        if (node.left != null) {
            stack.push(node.left);
        }
        node = node.right;  // right -> left
        if (node == null && !stack.isEmpty()) {
            node = stack.pop();
        }
    }
    Collections.reverse(list);  // reverse
    return list;
}
{% endhighlight %}

## Cache

[Binary Search Tree Iterator II][binary-search-tree-iterator-ii]

{% highlight java %}
class BSTIterator {
    private Deque<TreeNode> stack = new ArrayDeque<>();
    // precomputed values
    private List<Integer> list = new ArrayList<>();
    private TreeNode node;
    // index of the node in list
    private int index = -1;

    public BSTIterator(TreeNode root) {
        this.node = root;
    }
    
    public boolean hasNext() {
        return node != null || !stack.isEmpty() || index < list.size() - 1;
    }
    
    public int next() {
        // check if it's out of the range of list
        if (++index == list.size()) {
            while (node != null) {
                stack.push(node);
                node = node.left;
            }
            
            node = stack.pop();
            list.add(node.val);  // inorder
            node = node.right;
        }
        
        return list.get(index);
    }
    
    public boolean hasPrev() {
        return index > 0;
    }
    
    public int prev() {
        return list.get(--index);
    }
}
{% endhighlight %}

## Vertical

[Binary Tree Vertical Order Traversal][binary-tree-vertical-order-traversal]

{% highlight java %}
public List<List<Integer>> verticalOrder(TreeNode root) {
    List<List<Integer>> list = new ArrayList<>();
    if (root == null) {
        return list;
    }

    Map<Integer, List<Integer>> map = new TreeMap<>();
    Queue<TreeNode> q = new LinkedList<>();
    Queue<Integer> cols = new LinkedList<>();

    q.add(root);
    cols.add(0);

    while (!q.isEmpty()) {
        TreeNode node = q.poll();
        int col = cols.poll();

        map.computeIfAbsent(col, k -> new ArrayList<>()).add(node.val);

        if (node.left != null) {
            q.add(node.left); 
            cols.add(col - 1);
        }

        if (node.right != null) {
            q.add(node.right);
            cols.add(col + 1);
        }
    }

    for (var col : map.values()) {
        list.add(col);
    }

    return list;
}
{% endhighlight %}

[Vertical Order Traversal of a Binary Tree][vertical-order-traversal-of-a-binary-tree]

{% highlight java %}
private Map<Integer, Map<Integer, PriorityQueue<Integer>>> map = new TreeMap<>();

public List<List<Integer>> verticalTraversal(TreeNode root) {
    dfs(root, 0, 0);

    List<List<Integer>> list = new ArrayList<>();
    for (var ys : map.values()) {
        List<Integer> tmp = new ArrayList<>();
        for (var nodes : ys.values()) {
            while (!nodes.isEmpty()) {
                tmp.add(nodes.poll());
            }
        }
        list.add(tmp);
    }
    return list;
}

private void dfs(TreeNode root, int x, int y) {
    if (root == null) {
        return;
    }

    map.putIfAbsent(x, new TreeMap<>());
    map.get(x).putIfAbsent(y, new PriorityQueue<>());
    map.get(x).get(y).offer(root.val);

    dfs(root.left, x - 1, y + 1);
    dfs(root.right, x + 1, y + 1);
}
{% endhighlight %}

## Depth First Search

[Find Largest Value in Each Tree Row][find-largest-value-in-each-tree-row]

{% highlight java %}
public List<Integer> largestValues(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    if (root == null) {
        return list;
    }

    dfs(root, list, 0);
    return list;
}

private void dfs(TreeNode node, List<Integer> list, int depth) {
    if (node == null) {
        return;
    }

    if (depth == list.size()) {
        list.add(node.val);
    } else {
        list.set(depth, Math.max(list.get(depth), node.val));
    }

    dfs(node.left, list, depth + 1);
    dfs(node.right, list, depth + 1);
}
{% endhighlight %}

[Flip Binary Tree To Match Preorder Traversal][flip-binary-tree-to-match-preorder-traversal]

{% highlight java %}
private List<Integer> list = new ArrayList<>();
private int index = 0;

public List<Integer> flipMatchVoyage(TreeNode root, int[] voyage) {
    return dfs(root, voyage) ? list : Arrays.asList(-1);
}

// preorder
private boolean dfs(TreeNode node, int[] voyage) {
    if (node == null) {
        return true;
    }

    if (node.val != voyage[index++]) {
        return false;
    }

    if (node.left != null && node.left.val != voyage[index]) {
        list.add(node.val);
        return dfs(node.right, voyage) && dfs(node.left, voyage);
    }

    return dfs(node.left, voyage) && dfs(node.right, voyage);
}
{% endhighlight %}

# Construction

[Construct Binary Tree from Preorder and Inorder Traversal][construct-binary-tree-from-preorder-and-inorder-traversal]

The key is to find the root in inorder. We can iterate to find it, or keep track of it in a map. Or optimally:

{% highlight java %}
private int pre = 0, in = 0;

public TreeNode buildTree(int[] preorder, int[] inorder) {
    // min int is a virtual parent of root
    return build(preorder, inorder, Integer.MIN_VALUE);
}

private TreeNode build(int[] preorder, int[] inorder, int prevRoot) {
    if (pre >= preorder.length) {
        return null;
    }

    // stops at previous root in inorder
    if (inorder[in] == prevRoot) {
        in++;
        return null;
    }

    TreeNode node = new TreeNode(preorder[pre++]);
    node.left = build(preorder, inorder, node.val);
    node.right = build(preorder, inorder, prevRoot);
    return node;
}
{% endhighlight %}

For example:

```
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
```
```
pre	in	prevRoot
0	0	-2147483648
1	0	3
2	0	9
2	1	3
2	2	-2147483648
3	2	20
4	2	15
4	3	20
4	4	-2147483648
5	4	7
5	4	-2147483648
```

[Construct Binary Tree from Inorder and Postorder Traversal][construct-binary-tree-from-inorder-and-postorder-traversal]

{% highlight java %}
private int in, post;

public TreeNode buildTree(int[] inorder, int[] postorder) {
    this.in = inorder.length - 1;
    this.post = postorder.length - 1;

    // min int is a virtual parent of root
    return build(inorder, postorder, Integer.MIN_VALUE);
}

private TreeNode build(int[] inorder, int[] postorder, int prevRoot) {
    if (post < 0) {
        return null;
    }

    // stops at previous root in inorder
    if (inorder[in] == prevRoot) {
        in--;
        return null;
    }

    TreeNode node = new TreeNode(postorder[post--]);
    node.right = build(inorder, postorder, node.val);
    node.left = build(inorder, postorder, prevRoot);
    return node;
}
{% endhighlight %}

[Construct Binary Tree from Preorder and Postorder Traversal][construct-binary-tree-from-preorder-and-postorder-traversal]

{% highlight java %}
private int preIndex = 0, postIndex = 0;

public TreeNode constructFromPrePost(int[] pre, int[] post) {
    TreeNode root = new TreeNode(pre[preIndex++]);

    // if root.val == post[postIndex]
    // the entire (sub)tree at root is constructed
    if (root.val != post[postIndex]) {
        root.left = constructFromPrePost(pre, post);
    }

    if (root.val != post[postIndex]) {
        root.right = constructFromPrePost(pre, post);
    }

    postIndex++;
    return root;
}
{% endhighlight %}

For example:

```
pre = [1,2,4,5,3,6,7]
post = [4,5,2,6,7,3,1]
```
```
preIndex	postIndex
0		0
1		0
2		0
3		1
4		3
5		3
6		4
```

{% highlight java %}
public TreeNode constructFromPrePost(int[] pre, int[] post) {
    Deque<TreeNode> dq = new ArrayDeque<>();
    dq.offer(new TreeNode(pre[0]));
    for (int i = 1, j = 0; i < pre.length; i++) {
        TreeNode node = new TreeNode(pre[i]);
        while (dq.getLast().val == post[j]) {
            dq.pollLast();
            j++;
        }
        if (dq.getLast().left == null) {
            dq.getLast().left = node;
        } else {
            dq.getLast().right = node;
        }
        dq.offer(node);
    }
    return dq.getFirst();
}
{% endhighlight %}

## BST

[Construct Binary Search Tree from Preorder Traversal][construct-binary-search-tree-from-preorder-traversal]

{% highlight java %}
private int index = 0;

public TreeNode bstFromPreorder(int[] preorder) {
    return build(preorder, Integer.MAX_VALUE);
}

public TreeNode build(int[] preorder, int high) {
    if (index == preorder.length || preorder[index] > high) {
        return null;
    }

    TreeNode root = new TreeNode(preorder[index++]);
    root.left = build(preorder, root.val);
    root.right = build(preorder, high);
    return root;
}
{% endhighlight %}

[Convert Sorted List to Binary Search Tree][convert-sorted-list-to-binary-search-tree]

{% highlight java %}
private ListNode curr;

public TreeNode sortedListToBST(ListNode head) {
    // finds the size of the linked list
    ListNode node = head;
    int size = 0;
    while (node != null) {
        node = node.next;
        size++;
    }

    this.curr = head;

    return convertListToBst(0, size - 1);
}

// inorder
private TreeNode convertListToBst(int low, int high) {
    if (low > high) {
        return null;
    }

    int mid = (low + high) >>> 1;

    TreeNode left = convertListToBst(low, mid - 1);

    TreeNode node = new TreeNode(curr.val);
    node.left = left;

    curr = curr.next;

    node.right = convertListToBst(mid + 1, high);
    return node;
}
{% endhighlight %}

[Convert Sorted Array to Binary Search Tree][convert-sorted-array-to-binary-search-tree]

{% highlight java %}
    // no need for inorder and global variable
    // because we can get current root directly by its index mid
    TreeNode node = new TreeNode(num[mid]);
    node.left = helper(num, low, mid - 1);
    node.right = helper(num, mid + 1, high);
{% endhighlight %}

# Verification

[Verify Preorder Sequence in Binary Search Tree][verify-preorder-sequence-in-binary-search-tree]

{% highlight java %}
public boolean verifyPreorder(int[] preorder) {
    int low = Integer.MIN_VALUE;
    Deque<Integer> st = new ArrayDeque<>();
    for (int p : preorder) {
        if (p < low) {
            return false;
        }

        // monotonically decreasing stack
        // pops left subtree
        while (!st.isEmpty() && p > st.peek()) {
            // the sequence of low is the inorder traversal
            low = st.pop();
        }
        st.push(p);
    }
    return true;
}
{% endhighlight %}

In-place:

{% highlight java %}
public boolean verifyPreorder(int[] preorder) {
    int low = Integer.MIN_VALUE, i = -1;
    for (int p : preorder) {
        if (p < low) {
            return false;
        }

        // pops left subtree
        while (i >= 0 && p > preorder[i]) {
            // the sequence of low is the inorder traversal
            low = preorder[i--];
        }
        preorder[++i] = p;
    }
    return true;
}
{% endhighlight %}

For example:

```
[5,2,1,3,6]
```
```
		low
[5,2,1,3,6]	-2147483648
[5,2,1,3,6]	-2147483648
[5,2,1,3,6]	-2147483648
[5,3,1,3,6]	2
[6,3,1,3,6]	5
```

In-place, no overwriting:

{% highlight java %}
public boolean verifyPreorder(int[] preorder) {
    int low = Integer.MIN_VALUE;
    for (int i = 0; i < preorder.length; i++) {
        if (preorder[i] < low) {
            return false;
        }

        for (int j = i - 1; j >= 0 && preorder[j] < preorder[i]; j--) {
            low = Math.max(low, preorder[j]);
        }
    }
    return true;
}
{% endhighlight %}

# Predecessor/Successor

[Inorder Successor in BST][inorder-successor-in-bst]

{% highlight java %}
public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
    TreeNode node = root, candidate = null;
    // binary search
    while (node != null) {
        if (node.val > p.val) {
            candidate = node;
            node = node.left;
        } else {
            node = node.right;
        }
    }
    return candidate;
}
{% endhighlight %}

[Inorder Successor in BST II][inorder-successor-in-bst-ii]

# Complexity

# Binary Search Tree

[Binary Search Tree Iterator][binary-search-tree-iterator]

{% highlight java %}
private Deque<TreeNode> stack;

public BSTIterator(TreeNode root) {
    this.stack = new ArrayDeque<>();
    leftmostInorder(root);
}

/** @return the next smallest number */
public int next() {
    TreeNode tmpNode = stack.pop();
    leftmostInorder(tmpNode.right);
    return tmpNode.val;
}

/** @return whether we have a next smallest number */
public boolean hasNext() {
    return !stack.isEmpty();
}

private void leftmostInorder(TreeNode node) {
    while (node != null) {
        stack.push(node);
        node = node.left;
    }
}
{% endhighlight %}

[All Elements in Two Binary Search Trees][all-elements-in-two-binary-search-trees/submissions]

{% highlight java %}
public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {
    List<Integer> list = new ArrayList<>();
    Deque<TreeNode> st1 = new ArrayDeque<>(), st2 = new ArrayDeque<>();
    TreeNode node1 = root1, node2 = root2;
    while (true) {
        while (node1 != null) {
            st1.push(node1);
            node1 = node1.left;
        }
        while (node2 != null) {
            st2.push(node2);
            node2 = node2.left;
        }

        if (st1.isEmpty() && st2.isEmpty()) {
            return list;
        }

        if (st2.isEmpty() || (!st1.isEmpty() && st1.peek().val <= st2.peek().val)) {
            node1 = st1.pop();
            list.add(node1.val);
            node1 = node1.right;
        } else {
            node2 = st2.pop();
            list.add(node2.val);
            node2 = node2.right;
        }
    }
    return list;
}
{% endhighlight %}

[all-elements-in-two-binary-search-trees/submissions]: https://leetcode.com/problems/all-elements-in-two-binary-search-trees/submissions/
[binary-search-tree-iterator]: https://leetcode.com/problems/binary-search-tree-iterator/
[binary-search-tree-iterator-ii]: https://leetcode.com/problems/binary-search-tree-iterator-ii/
[binary-tree-vertical-order-traversal]: https://leetcode.com/problems/binary-tree-vertical-order-traversal
[binary-tree-preorder-traversal]: https://leetcode.com/problems/binary-tree-preorder-traversal
[binary-tree-inorder-traversal]: https://leetcode.com/problems/binary-tree-inorder-traversal
[binary-tree-postorder-traversal]: https://leetcode.com/problems/binary-tree-postorder-traversal
[construct-binary-search-tree-from-preorder-traversal]: https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/
[construct-binary-tree-from-inorder-and-postorder-traversal]: https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
[construct-binary-tree-from-preorder-and-inorder-traversal]: https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
[construct-binary-tree-from-preorder-and-postorder-traversal]: https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/
[convert-sorted-array-to-binary-search-tree]: https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
[convert-sorted-list-to-binary-search-tree]: https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/
[find-largest-value-in-each-tree-row]: https://leetcode.com/problems/find-largest-value-in-each-tree-row/
[flip-binary-tree-to-match-preorder-traversal]: https://leetcode.com/problems/flip-binary-tree-to-match-preorder-traversal/
[inorder-successor-in-bst]: https://leetcode.com/problems/inorder-successor-in-bst/
[inorder-successor-in-bst-ii]: https://leetcode.com/problems/inorder-successor-in-bst-ii/
[verify-preorder-sequence-in-binary-search-tree]: https://leetcode.com/problems/verify-preorder-sequence-in-binary-search-tree/
[vertical-order-traversal-of-a-binary-tree]: https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/
