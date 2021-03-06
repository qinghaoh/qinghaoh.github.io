---
layout: post
title:  "Substring"
tags: string
---
# Dynamic Programming

[Longest Repeating Substring][longest-repeating-substring]

{% highlight java %}
// O(n ^ 2)
public int longestRepeatingSubstring(String S) {
    int n = S.length();
    // dp[i][j]: length of longeset common suffix of S.substring(0, i + 1) and S.substring(0, j + 1)
    int[][] dp = new int[n + 1][n + 1];
    int max = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = i + 1; j <= n; j++) {
            if (S.charAt(i - 1) == S.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                max = Math.max(max, dp[i][j]);
            }
        }
    }
    return max;
}
{% endhighlight %}

[Encode String with Shortest Length][encode-string-with-shortest-length]

{% highlight java %}
// O(n ^ 4)
public String encode(String s) {
    int n = s.length();

    // dp[i][j]: s.substring(i, j + 1) in encoded form
    String[][] dp = new String[n][n];

    for (int len = 0; len < n; len++) {
        for (int i = 0; i < n - len; i++) {
            int j = i + len;
            String sub = s.substring(i, j + 1);     // length == (len + 1)
            dp[i][j] = sub;

            // when String length is less than 5, encoding won't shorten it
            if (len > 3) {
                // splits the substring
                for (int k = i; k < j; k++) {
                    if (dp[i][k].length() + dp[k + 1][j].length() < dp[i][j].length()) {
                        dp[i][j] = dp[i][k] + dp[k + 1][j];
                    }
                }

                // checks if the substring contains repeatable substrings
                for (int k = 0; k < sub.length(); k++) {
                    String repeatableSub = sub.substring(0, k + 1);   // length == k + 1
                    // the first condition sometimes shortcuts so replaceAll is not called in every iteration
                    if (sub.length() % repeatableSub.length() == 0 
                       && sub.replaceAll(repeatableSub, "").length() == 0) {
                          String decodedString = sub.length() / repeatableSub.length() + "[" + dp[i][i + k] + "]";
                        if (decodedString.length() < dp[i][j].length()) {
                            dp[i][j] = decodedString;
                        }

                        // shorter repeated pattern is always better than longer one
                        // e.g. 4[ab] is bettter than 2[abab]
                        break;
                    }
                }
            }
        }
    }

    return dp[0][n - 1];
}
{% endhighlight %}

# String Matching

## KMP

[Knuth–Morris–Pratt (KMP) algorithm](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)

[Shortest Palindrome][shortest-palindrome]

{% highlight java %}
// KMP
// O(n + k)
public String shortestPalindrome(String s) {
    // finds the longest palindrome substring that starts from index 0
    int n = s.length();
    String r = new StringBuilder(s).reverse().toString();
    String str = s + "#" + r;

    // partial match table
    int[] table = new int[str.length()];
    int i = 1, index = 0;
    while (i < str.length()) {
        if (str.charAt(index) == str.charAt(i)) {
            table[i++] = ++index;
        } else {
            if (index > 0) {
                index = table[index - 1];
            } else {
                i++;
            }
        }
    }
    return new StringBuilder(s.substring(table[table.length - 1])).reverse().toString() + s;
}
{% endhighlight %}

[Rabin-Karp algorithm](https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm): a string-searching algorithm that uses hashing to find an exact match of a pattern string in a text. It uses a rolling hash to quickly filter out positions of the text that cannot match the pattern, and then checks for a match at the remaining positions.

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

    long coff = 1;
    for (int i = 1; i < len; i++) {
        coff = (coff * a) % mod;
    }

    // hash : start indexes
    Map<Long, List<Integer>> map = new HashMap<>();

    // start index == 0
    map.computeIfAbsent(h, k -> new ArrayList<>()).add(0);

    // start index > 0
    int start = 0;
    while (start + len < s.length()) {
        // rolling hash
        h = ((h + mod - coff * s.charAt(start) % mod) * a + s.charAt(start + len)) % mod;
        start++;

        // Rabin-Karp
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

# Greedy

[Stamping The Sequence][stamping-the-sequence]

{% highlight java %}
private int questions = 0;  // count of wildcard '?'
private String stamp;
private char[] t;

// reverse
public int[] movesToStamp(String stamp, String target) {
    int m = stamp.length(), n = target.length();
    this.stamp = stamp;
    this.t = target.toCharArray();

    // visited[i] indicates the string t.substring(i, i + m) is stamped or not
    // this avoids stamps at the same place
    boolean[] visited = new boolean[n - m + 1];
    List<Integer> list = new ArrayList<>();
    while (questions < n) {
        boolean stamped = false;
        for (int i = 0; i <= n - m && questions < n; i++) {
            if (!visited[i] && canStamp(i)) {
                doStamp(i);
                stamped = true;
                list.add(0, i);
                visited[i] = true;
            }
        }

        if (!stamped) {
            return new int[0];
        }
    }

    return list.stream().mapToInt(Integer::valueOf).toArray();
}

private boolean canStamp(int pos) {
    for (int i = 0; i < stamp.length(); i++) {
        if (t[pos + i] != '?' && t[pos + i] != stamp.charAt(i)) {
            return false;
        }
    }
    return true;
}

private void doStamp(int pos) {
    for (int i = 0; i < stamp.length(); i++) {
        if (t[pos + i] != '?') {
            t[pos + i] = '?';
            questions++;
        }
    }
}
{% endhighlight %}

[encode-string-with-shortest-length]: https://leetcode.com/problems/encode-string-with-shortest-length/
[longest-duplicate-substring]: https://leetcode.com/problems/longest-duplicate-substring/
[longest-repeating-substring]: https://leetcode.com/problems/longest-repeating-substring/
[shortest-palindrome]: https://leetcode.com/problems/shortest-palindrome/
[stamping-the-sequence]: https://leetcode.com/problems/stamping-the-sequence/
