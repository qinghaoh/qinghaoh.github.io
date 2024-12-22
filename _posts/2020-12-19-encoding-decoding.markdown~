---
title:  "Encoding/Decoding"
tag: string
---

# Chunked Transfer Encoding

[Chunked transfer encoding](https://en.wikipedia.org/wiki/Chunked_transfer_encoding): HTTP/1.1

[Encode and Decode Strings][encode-and-decode-strings]

```java
private String toCount(String s) {
    int length = 4;  // 4 chunks
    char[] bytes = new char[length];
    for (int i = length - 1; i >= 0; i--) {
        bytes[length - 1 - i] = (char) (s.length() >> (i * 8) & 0xFF);
    }
    return new String(bytes);
}
```

[Serialize and Deserialize Binary Tree][serialize-and-deserialize-binary-tree]

```java
// preorder
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serialize(root, sb)
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        return deserialize(new LinkedList<>(Arrays.asList(data.split(","))));
    }
    
    private void serialize(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append("#").append(",");
            return;
        }
        
        sb.append(root.val).append(",");
        serialize(root.left, sb);
        serialize(root.right, sb);
    }
    
    private TreeNode deserialize(Queue<String> q) {
        String val = q.poll();
        if ("#".equals(val)) {
            return null;
        }
        
        TreeNode root = new TreeNode(Integer.valueOf(val));
        root.left = deserialize(q);
        root.right = deserialize(q);
        return root;
    }
}
```

[Serialize and Deserialize N-ary Tree][serialize-and-deserialize-n-ary-tree]

```java
class Codec {
    // Encodes a tree to a single string.
    public String serialize(Node root) {
        StringBuilder sb = new StringBuilder();
        serialize(root, sb);
        return sb.toString();
    }
	
    // Decodes your encoded data to tree.
    public Node deserialize(String data) {
        return deserialize(new LinkedList<>(Arrays.asList(data.split(","))));
    }
    
    private void serialize(Node root, StringBuilder sb) {
        if (root == null) {
            sb.append("#").append(",");
            return;
        }
        
        sb.append(root.val).append(",");
        if (root.children.isEmpty()) {
            sb.append("#").append(",");
        } else {
            sb.append(root.children.size()).append(",");
            for (Node child : root.children) {
                serialize(child, sb);
            }
        }
    }
    
    private Node deserialize(Queue<String> q) {
        String val = q.poll();
        if ("#".equals(val)) {
            return null;
        }
        
        // count of children
        String count = q.poll();

        Node root = new Node(Integer.valueOf(val), new ArrayList<>());
        if (!count.equals("#")) {
            for (int i = 0; i < Integer.valueOf(count); i++) {
                root.children.add(deserialize(q));
            }
        }
        return root;
    }
}
```

[Encode N-ary Tree to Binary Tree][encode-n-ary-tree-to-binary-tree]

[Convert a m-ary tree to binary tree](https://en.wikipedia.org/wiki/M-ary_tree#Convert_a_m-ary_tree_to_binary_tree)

[Construct Binary Tree from String][construct-binary-tree-from-string]

```java
public TreeNode str2tree(String s) {
    Deque<TreeNode> st = new ArrayDeque<>();
    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        if (c == ')') {
            st.pop();
        } else if (c != '(') {
            int j = i;
            while (i + 1 < s.length() && Character.isDigit(s.charAt(i + 1))) {
                i++;
            }

            TreeNode node = new TreeNode(Integer.valueOf(s.substring(j, i + 1)));
            if (!st.isEmpty()) {
                TreeNode parent = st.peek();
                if (parent.left != null) {
                    parent.right = node;
                } else {
                    parent.left = node;
                }
            }
            st.push(node);
        }
    }
    return st.isEmpty() ? null : st.peek();
}
```

[Encode and Decode TinyURL][encode-and-decode-tinyurl]

```java
public class Codec {
    // 62 chars
    private static final String ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private static final String PATH = "http://tinyurl.com/";

    private Map<String, String> map = new HashMap<>();
    private Random rand = new Random();
    private String key = getRand();

    private String getRand() {
        StringBuilder sb = new StringBuilder();
        // short url has 6 random chars
        for (int i = 0; i < 6; i++) {
            sb.append(ALPHABET.charAt(rand.nextInt(ALPHABET.length())));
        }
        return sb.toString();
    }

    // Encodes a URL to a shortened URL.
    public String encode(String longUrl) {
        while (map.containsKey(key)) {
            key = getRand();
        }
        map.put(key, longUrl);
        return PATH + key;
    }

    // Decodes a shortened URL to its original URL.
    public String decode(String shortUrl) {
        return map.get(shortUrl.replace(PATH, ""));
    }
}
```

[Find Duplicate Subtrees][find-duplicate-subtrees]

```java
private List<TreeNode> list = new ArrayList<>();
private Map<String, Integer> map = new HashMap<>();

// O(n ^ 2)
public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
    dfs(root);
    return list;
}

private String dfs(TreeNode node) {
    if (node == null) {
        return "#";
    }

    // O(n) concatenation
    String s = node.val + "#" + dfs(node.left) + "#" + dfs(node.right);
    map.put(s, map.getOrDefault(s, 0) + 1);
    if (map.get(s) == 2) {
        list.add(node);
    }
    return s;
}
```

Optimization:

```java
private int id = 0;
private List<TreeNode> list = new ArrayList<>();
// serialized string : id
private Map<String, Integer> map = new HashMap<>();
// id : count
private Map<Integer, Integer> count = new HashMap<>();

// O(n)
public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
    dfs(root);
    return list;
}

private int dfs(TreeNode node) {
    if (node == null) {
        return Integer.MIN_VALUE;
    }

    // O(1) concatenation
    String s = node.val + "#" + dfs(node.left) + "#" + dfs(node.right);
    // if the pattern is new, assign a new id to it
    map.putIfAbsent(s, id++);

    int sid = map.get(s);
    count.put(sid, count.getOrDefault(sid, 0) + 1);
    if (count.get(sid) == 2) {
        list.add(node);
    }
    return sid;
}
```

[Isomorphic Strings][isomorphic-strings]

```java
public boolean isIsomorphic(String s, String t) {
    return transform(s).equals(transform(t));
}

private String transform(String s) {
    Map<Character, Integer> indexMapping = new HashMap<>();
    StringJoiner sj = new StringJoiner("#");

    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        indexMapping.putIfAbsent(c, i);
        sj.add(Integer.toString(indexMapping.get(c)));
    }

    return sj.toString();
}
```

# Run-length encoding

[Run-length encoding (RLE)](https://en.wikipedia.org/wiki/Run-length_encoding): a form of lossless data compression in which runs of data (sequences in which the same data value occurs in many consecutive data elements) are stored as a single data value and count, rather than as the original run.

[String Compression II][string-compression-ii]

```java
public int getLengthOfOptimalCompression(String s, int k) {
    int n = s.length();
    // dp[i][j]: i-th character and j characters are deleted
    int[][] dp = new int[n + 1][k + 2];
    for (int i = 0; i < dp.length; i++) {
        Arrays.fill(dp[i], n + 1);
    }
    dp[0][0] = 0;

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= k; j++) {
            // deletes current character
            dp[i][j + 1] = Math.min(dp[i][j + 1], dp[i - 1][j]);

            // checks how far it can go to delete every character that is not the same to the i-th character
            // while making sure the total deletion <= k
            int count = 0, deletion = 0;
            for (int l = i; l <= n; l++) {
                if (s.charAt(i - 1) == s.charAt(l - 1)) {
                    count++;
                } else {
                    deletion++;
                }

                // more deletions than allowed
                if (j + deletion > k) {
                    break;
                }

                // length of the string representation of count
                int length = count == 1 ? 0 : (int)Math.log10(count) + 1;

                // +1 is the current character s.charAt(i)
                dp[l][j + deletion] = Math.min(dp[l][j + deletion], dp[i - 1][j] + 1 + length);
            }
        }
    }
    return dp[n][k];
}
```

[construct-binary-tree-from-string]: https://leetcode.com/problems/construct-binary-tree-from-string/
[encode-and-decode-strings]: https://leetcode.com/problems/encode-and-decode-strings/
[encode-and-decode-tinyurl]: https://leetcode.com/problems/encode-and-decode-tinyurl/
[encode-n-ary-tree-to-binary-tree]: https://leetcode.com/problems/encode-n-ary-tree-to-binary-tree/
[find-duplicate-subtrees]: https://leetcode.com/problems/find-duplicate-subtrees/
[isomorphic-strings]: https://leetcode.com/problems/isomorphic-strings/
[serialize-and-deserialize-binary-tree]: https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
[serialize-and-deserialize-n-ary-tree]: https://leetcode.com/problems/serialize-and-deserialize-n-ary-tree/
[string-compression-ii]: https://leetcode.com/problems/string-compression-ii/
