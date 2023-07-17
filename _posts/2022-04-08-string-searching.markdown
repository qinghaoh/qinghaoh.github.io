---
title:  "String Searching"
tags: string
---
# Iteration

[Length of the Longest Valid Substring][length-of-the-longest-valid-substring]

{% highlight java %}
private static final int MAX_FORBIDDEN_LEN = 10;

public int longestValidSubstring(String word, List<String> forbidden) {
    Set<String> set = new HashSet(forbidden);
    int max = 0, n = word.length(), i = n - 1, j = n;
    while (i >= 0) {
        // if word.substring(i + 1, j) is valid
        // finds the longest k' where word.substring(i, k') is not forbidden
        // then word.substring(i, k' + 1) is valid
        for (int k = i; k < Math.min(i + MAX_FORBIDDEN_LEN, j); k++) {
            if (set.contains(word.substring(i, k + 1))) {
                j = k;
                break;
            }
        }
        max = Math.max(max, j - i--);
        // now word.substring(i + 1, j) is valid
    }
    return max;
}
{% endhighlight %}

# KMP

[Knuth–Morris–Pratt (KMP) algorithm](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)

* Construct an auxiliary array `lps[]` of the same size as pattern
* `lps[i]` is the length of the longest matching proper prefix of the sub-pattern `pat[0...i]`, which is also a suffix of the sub-pattern `pat[0...i]`. A proper prefix of a string is a prefix that is not equal to the string itself.

For example, for the pattern `AABAACAABAA`, 
```
lps[] = [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]
```

{% highlight java %}
int[] computeLps(String s) {
    int m = s.length();
    int[] lps = new int[m];

    // lps[0] == 0
    for (int i = 1, j = 0; i < m; i++) {
        while (j > 0 && s.charAt(i) != s.charAt(j)) {
            j = lps[j - 1];
        }
        
        if (s.charAt(i) == s.charAt(j)) {
            lps[i] = ++j;
        }
    }
    return lps;
}
{% endhighlight %}

[Longest Happy Prefix][longest-happy-prefix]

{% highlight java %}
public String longestPrefix(String s) {
    return s.substring(0, computeLps(s)[s.length() - 1]);
}
{% endhighlight %}

[Shortest Palindrome][shortest-palindrome]

{% highlight java %}
public String shortestPalindrome(String s) {
    // "abace" -> "ec" + "aba" + "ce"
    // finds the longest prefix palindrome of s
    int[] lps = computeLps(s + "#" + new StringBuilder(s).reverse().toString());
    return new StringBuilder(s.substring(lps[lps.length - 1])).reverse().toString() + s;
}
{% endhighlight %}

[Sum of Scores of Built Strings][sum-of-scores-of-built-strings]

{% highlight java %}
public long sumScores(String s) {
    int[] lps = computeLps(s);
    int m = lps.length;
    // count[i]: occurrences of s[i] as the last character of a postfix of s[j...i] which equals a prefix of s,
    //   where j != 0
    int[] count = new int[m];
    for (int i = 0; i < m; i++) {
        int j = lps[i];
        // +1 is s.substring(0, j), because lps[j - 1] doesn't include s.substring[0, j] itself
        count[i] = j == 0 ? 0 : count[j - 1] + 1;
    }
    // adding count[i] of all chars yields the total score
    return Arrays.stream(count).asLongStream().sum() + m;
}
{% endhighlight %}

For example, `s = "babab"`

```
lps   = [0, 0, 1, 2, 3]
count = [0, 0, 1, 1, 2] 
               |     |
              "b"    "b"   (count[2])
                     "bab" (+1)
```

[Maximum Deletions on a String][maximum-deletions-on-a-string]

{% highlight java %}
public int deleteString(String s) {
    int n = s.length();
    int[] dp = new int[n];
    dp[n - 1] = 1;

    for (int i = n - 2; i >= 0; i--) {
        dp[i] = 1;
        int[] lps = computeLps(s.substring(i, n));
        for (int j = 1; i + j < n; j += 2) {
            // uses KMP LPS to quickly find the prefix which can be split to two identical parts
            // e.g. "aaab"
            // if i == 0, j == 1, then lps[1] = 1, which means "aa" is the good prefix
            if (lps[j] == j / 2 + 1) {
                dp[i] = Math.max(dp[i], 1 + dp[i + lps[j]]);
            }
        }
    }
    return dp[0];
}
{% endhighlight %}

* Search pattern in text with the help of `lps[]`

{% highlight java %}
// finds the start indices of matches
List<Integer> kmp(String text, String pattern) {
    int n = text.length(), m = pattern.length();
    int[] lps = computeLps(pattern);

    List<Integer> list = new ArrayList<>();
    for (int i = 0, j = 0; i < n; i++) {
        while (j == m || (j > 0 && pattern.charAt(j) != text.charAt(i))) {
            j = lps[j - 1];
        }

        if (pattern.charAt(j) == text.charAt(i)) {
            if (++j == m) {
                list.add(i - j + 1);
            }
        }
    }
    return list;
}
{% endhighlight %}

[Remove All Occurrences of a Substring][remove-all-occurrences-of-a-substring]

{% highlight java %}
public String removeOccurrences(String s, String part) {
    int n = s.length(), m = part.length();
    int[] lps = computeLps(part);

    Deque<Character> st = new ArrayDeque<>();

    // stores pattern index j so that after character deletion it can be restored
    int[] index = new int[n + 1];

    for (int i = 0, j = 0; i < n; i++) {
        st.push(s.charAt(i));

        if (st.peek() == part.charAt(j)) {
            // stores the next index of j
            index[st.size()] = ++j;

            if (j == m) {
                // deletes the whole part when a match is found
                int count = m;
                while (count > 0) {
                    st.pop();
                    count--;
                }

                // restores the index of j to find next match
                j = st.isEmpty() ? 0 : index[st.size()];
            }
        } else {
            if (j > 0) {
                j = lps[j - 1];
                st.pop();
                i--;
            } else {
                // resets the stored index
                index[st.size()] = 0;
            }
        }
    }

    return new StringBuilder(st.stream().map(Object::toString).collect(Collectors.joining())).reverse().toString();
}
{% endhighlight %}

[Find All Good Strings][find-all-good-strings]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;
private String s1, s2, evil;
private int[] memo = new int[1 << 17];
private int[] lps;

public int findGoodStrings(int n, String s1, String s2, String evil) {
    this.s1 = s1;
    this.s2 = s2;
    this.evil = evil;
    this.lps = computeLps(evil);

    return dfs(0, 0, true, true);
}

// builds one character at each level
private int dfs(int index, int evilMatched, boolean startInclusive, boolean endInclusive) {
    // KMP found a match of evil
    // stops searching
    if (evilMatched == evil.length()) {
        return 0;
    }

    // built a good string
    if (index == s1.length()) {
        return 1;
    }

    int key = getKey(index, evilMatched, startInclusive, endInclusive);
    if (memo[key] != 0) {
        return memo[key];
    }

    int count = 0;
    char from = startInclusive ? s1.charAt(index) : 'a';
    char to = endInclusive ? s2.charAt(index) : 'z';
    for (char c = from; c <= to; c++) {
        // KMP to count the number of matches of pattern `evil` in current built text (path from root to c)
        int j = evilMatched;
        while (j > 0 && evil.charAt(j) != c) {
            j = lps[j - 1];
        }
        if (c == evil.charAt(j)) {
            j++;
        }
        count = (count + dfs(index + 1, j, startInclusive && (c == from), endInclusive && (c == to))) % MOD;
    }
    return memo[key] = count;
}

private int getKey(int n, int m, boolean b1, boolean b2) {
    // 9 bits to store n (2 ^ 9 = 512)
    // 6 bits to store m (2 ^ 6 = 64)
    // 1 bit to store b1
    // 1 bit to store b2
    // 17 bits in total
    return (n << 8) | (m << 2) | ((b1 ? 1 : 0) << 1) | (b2 ? 1 : 0);
}
{% endhighlight %}

For example, `n = 2, s1 = "aa", s2 = "da", evil = "b"`

![Good strings](/assets/img/algorithm/find_all_good_strings.png)

# Z Function

[Z-function and its calculation](https://cp-algorithms.com/string/z-function.html): `z[i]` is the length of the longest string that is, at the same time, a prefix of `s` and a prefix of `s.substring(i)`.

{% highlight java %}
// O(n)
private int[] computeZ(String s) {
    int n = s.length();
    int[] z = new int[n];
    for (int i = 1, l = 0, r = 0; i < n; i++) {
        if (i <= r) {
            z[i] = Math.min(r - i + 1, z[i - l]);
        }

        while (i + z[i] < n && s.charAt(z[i]) == s.charAt(i + z[i])) {
            z[i]++;
        }

        if (i + z[i] - 1 > r) {
            l = i;
            r = i + z[i] - 1;
        }
    }
    return z;
}
{% endhighlight %}

[Sum of Scores of Built Strings][sum-of-scores-of-built-strings]

{% highlight java %}
public long sumScores(String s) {
    return Arrays.stream(computeZ(s)).asLongStream().sum() + s.length();
}
{% endhighlight %}

# Rolling Hash

[Longest Happy Prefix][longest-happy-prefix]

{% highlight java %}
public String longestPrefix(String s) {
    long h1 = 0, h2 = 0, mul = 1, mod = (long)1e9 + 7;
    int len = 0;
    for (int i = 0, j = s.length() - 1; j > 0; i++, j--) {
        h1 = (h1 * 26 + s.charAt(i) - 'a') % mod;
        h2 = (h2 + mul * (s.charAt(j) - 'a')) % mod;
        mul = mul * 26 % mod;
        if (h1 == h2) {
            // compares the string every time you find a matching hash
            // but only for characters we haven't checked before
            if (s.substring(len, i + 1).compareTo(s.substring(j + len)) == 0) {
                len = i + 1;
            }
        }
    }
    return s.substring(0, len);
}
{% endhighlight %}

[Rabin-Karp algorithm](https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm): a string-searching algorithm that uses hashing to find an exact match of a pattern string in a text. It uses a rolling hash to quickly filter out positions of the text that cannot match the pattern, and then checks for a match at the remaining positions.

$$ h(x) = \sum_{i = 0}^n a^{n - i}s_{i}$$

[Longest Duplicate Substring][longest-duplicate-substring]

{% highlight java %}
// O(nlog(n))
public String longestDupSubstring(String s) {
    // binary search
    int low = 0, high = s.length() - 1;
    while (low < high) {
        int mid = low + (high - low + 1) / 2;
        if (search(s, mid) != null) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return search(s, low);
}

/**
 * Searchs input string for duplicate substring with target length by Rabin-Karp Algorithm.
 * @param s input string
 * @param len target length
 * @return duplicate substring with target length; null if not found
 */
private String search(String s, int len) {
    if (len == 0) {
        return "";
    }

    // polynomial rolling hash
    long mod = (1 << 31) - 1, a = 256, h = 0;
    for (int i = 0; i < len; i++) {
        h = (h * a + s.charAt(i)) % mod;
    }

    long coeff = 1;
    for (int i = 1; i < len; i++) {
        coeff = (coeff * a) % mod;
    }

    // hash : start indexes
    Map<Long, List<Integer>> map = new HashMap<>();

    // start index == 0
    map.computeIfAbsent(h, k -> new ArrayList<>()).add(0);

    // start index > 0
    int start = 0;
    while (start + len < s.length()) {
        // rolling hash
        h = ((h + mod - coeff * s.charAt(start) % mod) * a + s.charAt(start + len)) % mod;
        start++;

        // Rabin-Karp collision check
        map.putIfAbsent(h, new ArrayList<>());
        for (int i : map.get(h)) {
            if (s.substring(start, start + len).equals(s.substring(i, i + len))) {
                return s.substring(i, i + len);
            }
        }
        map.get(h).add(start);
    }

    // no duplicate substring found
    return null;
}
{% endhighlight %}

# Suffix Automaton

[Suffix Automaton](https://cp-algorithms.com/string/suffix-automaton.html)

[Longest Common Subpath][longest-common-subpath]

{% highlight java %}
public class Solution {
    public int longestCommonSubpath(int n, int[][] paths) {
        // builds suffix automaton from the shortest path (denoted as path0)
        return new SuffixAutomaton(Arrays.stream(paths).min(Comparator.comparingInt(p -> p.length)).get())
                .longestCommonSubpath(paths);
    }

    // for a non-empty substring t of string s, endpos(t) is the set of all positions in the string s,
    // in which the occurrences of t end
    // e.g. s = "abcbc", endpos("bc") = {2, 4}
    // endpos-equivalent substrings correspond to the same state
    class State {
        // len: length of the longest substring match of the state
        // link: minLen(v) = len(link(v)) + 1
        //   link(v) represents a suffix of w, where w is the longest substring of state v
        int len = 0, link = -1;
        // next transitions
        Map<Integer, Integer> next = new HashMap<>();
        // lcs: longest common subpath (LCS) among all paths
        int lcs;
    }

    class SuffixAutomaton {
        private final State[] states;
        private int size = 1, last = 0;

        public SuffixAutomaton(int n) {
            // the number of states of a string of length n doesn't exceed 2 * n - 1
            this.states = new State[2 * n];
            for (int i = 0; i < states.length; i++) {
                states[i] = new State();
            }
        }

        public SuffixAutomaton(int[] path) {
            this(path.length);
            build(path);
        }

        public void extend(int c) {
            int curr = size++;
            states[curr].len = states[last].len + 1;

            int p = last;
            // only adds a new transition of c if it doesn't exist already
            while (p != -1 && !states[p].next.containsKey(c)) {
                states[p].next.put(c, curr);
                p = states[p].link;
            }

            if (p == -1) {
                states[curr].link = 0;
            } else {
                int q = states[p].next.get(c);
                if (states[p].len + 1 == states[q].len) {
                    // (p, q) is continuous
                    states[curr].link = q;
                } else {
                    // splits q into two substates
                    int clone = size++;
                    states[clone].len = states[p].len + 1;
                    states[clone].next = new HashMap<>(states[q].next);
                    states[clone].link = states[q].link;
                    while (p != -1 && states[p].next.replace(c, q, clone)) {
                        p = states[p].link;
                    }
                    states[q].link = states[curr].link = clone;
                }
            }
            last = curr;
        }

        // O(n * log(k))
        public void build(int[] path) {
            for (int c : path) {
                extend(c);
            }
            
            // LCS is the longest substring match for the first (shortest) path
            for (State s : states) {
                s.lcs = s.len;
            }
        }

        // calculates the LCS of each state
        private void lcs(int[] path) {
            // lcs[i]: LCS ending at state i
            int[] lcs = new int[size];

            // state and len of the current matching part
            int p = 0, len = 0;
            
            // for each position in `path`, finds the lcs of `path` and path0 ending in that position
            for (int c : path) {
                // shortens the current matching part
                while (states[p].link >= 0 && !states[p].next.containsKey(c)) {
                    p = states[p].link;
                    len = states[p].len;
                }
                
                if (states[p].next.containsKey(c)) {
                    p = states[p].next.get(c);
                    lcs[p] = Math.max(lcs[p], ++len);

                    // traverses by suffix links to backfill all states with LCS's
                    // each state on the traversal is a suffix of longest substring matching of p
                    int q = states[p].link;
                    while (lcs[q] < states[q].len) {
                        lcs[q] = states[q].len;
                        q = states[q].link;
                    }
                }
            }

            // state[i].lcs is the min among LCS of all paths at the state 
            for (int i = 0; i < size; i++) {
                states[i].lcs = Math.min(states[i].lcs, lcs[i]);
            }
        }

        public int longestCommonSubpath(int[][] paths) {
            for (int[] p : paths) {
                lcs(p);
            }

            // finds the max LCS among all states
            int max = 0;
            for (int i = 0; i < size; i++) {
                max = Math.max(max, states[i].lcs);
            }
            return max;
        }
    }
}
{% endhighlight %}

# Suffix Tree

[Suffix Tree](https://en.wikipedia.org/wiki/Suffix_tree)

[Ukkonen's Algorithm](https://cp-algorithms.com/string/suffix-tree-ukkonen.html): time complexity of tree construction: \\(O(nlog(k))\\)

[String Matching in an Array][string-matching-in-an-array]

{% highlight java %}
public List<String> stringMatching(String[] words) {
    TrieNode root = new TrieNode();
    for (String w : words) {
        for (int i = 0; i < w.length(); i++) {
            insert(root, w.substring(i));
        }
    }

    List<String> list = new ArrayList<>();
    for (String w : words) {
        TrieNode node = root;
        for (char ch : w.toCharArray()) {
            node = node.children[ch - 'a'];
        }
        if (node.count > 1) {
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
        // increases the counter of all nodes on the path
        // because we will search for "substring" instead of "suffix" in the trie,
        // we can't only mark the last node
        node.count++;
    }
}
{% endhighlight %}

[find-all-good-strings]: https://leetcode.com/problems/find-all-good-strings/
[length-of-the-longest-valid-substring]: https://leetcode.com/problems/length-of-the-longest-valid-substring/
[longest-common-subpath]: https://leetcode.com/problems/longest-common-subpath/
[longest-duplicate-substring]: https://leetcode.com/problems/longest-duplicate-substring/
[longest-happy-prefix]: https://leetcode.com/problems/longest-happy-prefix/
[maximum-deletions-on-a-string]: https://leetcode.com/problems/maximum-deletions-on-a-string/
[remove-all-occurrences-of-a-substring]: https://leetcode.com/problems/remove-all-occurrences-of-a-substring/
[shortest-palindrome]: https://leetcode.com/problems/shortest-palindrome/
[string-matching-in-an-array]: https://leetcode.com/problems/string-matching-in-an-array
[sum-of-scores-of-built-strings]: https://leetcode.com/problems/sum-of-scores-of-built-strings/
