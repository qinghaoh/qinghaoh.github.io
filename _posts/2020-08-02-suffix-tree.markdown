---
layout: post
title:  "Suffix Tree"
tags: tree
usemathjax: true
---
[Suffix Tree](https://en.wikipedia.org/wiki/Suffix_tree)

[Ukkonen's Algorithm](https://cp-algorithms.com/string/suffix-tree-ukkonen.html): time complexity of tree construction: \\(O(nlog(k))\\)

[String Matching in an Array][string-matching-in-an-array]

{% highlight java %}
public List<String> stringMatching(String[] words) {
    TrieNode root = new TrieNode();
    // builds suffix trie in O(n ^ 2)
    for (String w : words) {
        for (int i = 0; i < w.length(); i++) {
            insert(root, w.substring(i));
        }
    }

    List<String> list = new ArrayList<>();
    for (String w : words) {
        TrieNode node = root;
        int i = 0, m = w.length();
        while (i < m && node.children[w.charAt(i) - 'a'] != null) {
            node = node.children[w.charAt(i++) - 'a'];
        }
        if (i == m && node.count > 1) {
            list.add(w);
        }
    }
    return list;
}

class TrieNode {
    TrieNode[] children = new TrieNode[26];
    int count = 0;
}

private void insert(TrieNode root, String word) {        
    TrieNode node = root;
    for (char ch : word.toCharArray()) {
        if (node.children[ch - 'a'] == null) {
            node.children[ch - 'a'] = new TrieNode();
        }
        node = node.children[ch - 'a'];
        node.count++;
    }
}
{% endhighlight %}

[string-matching-in-an-array]: https://leetcode.com/problems/string-matching-in-an-array
