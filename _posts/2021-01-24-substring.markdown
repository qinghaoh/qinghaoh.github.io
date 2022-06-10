---
layout: post
title:  "Substring"
tags: string
usemathjax: true
---
# Index Map

[Longest Substring Without Repeating Characters][longest-substring-without-repeating-characters]

# Dynamic Programming

[Word Break][word-break]

{% highlight java %}
public boolean wordBreak(String s, List<String> wordDict) {
    Set<String> dict = new HashSet<>(wordDict);
    int n = s.length();
    // dp[i]: s.substring(0, i)
    boolean[] dp = new boolean[n + 1];
    dp[0] = true;

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < i; j++) {
            if (dp[j] && dict.contains(s.substring(j, i))) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[n];
}
{% endhighlight %}

[Word Break II][word-break-ii]

Recursion + Memoization:

{% highlight java %}
private Map<String, List<String>> map = new HashMap<>();

public List<String> wordBreak(String s, List<String> wordDict) {
    if (map.containsKey(s)) {
        return map.get(s);
    }

    List<String> list = new ArrayList<String>();
    if (wordDict.contains(s)) {
        list.add(s);
    }

    for (int i = 1 ; i < s.length(); i++) {
        String sub = s.substring(i);
        if (wordDict.contains(sub)) {
            for (String w : wordBreak(s.substring(0, i) , wordDict)) {
                list.add(w + " " + sub);
            }
        }
    }
    map.put(s, list);
    return list;
}
{% endhighlight %}

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

# Sliding Window

[Distinct Echo Substrings][distinct-echo-substrings]

{% highlight java %}
public int distinctEchoSubstrings(String text) {
    Set<String> set = new HashSet<>();
    int n = text.length();
    for (int len = 1; len <= n / 2; len++) {
        for (int l = 0, r = len, count = 0; l < n - len; l++, r++) {
            if (text.charAt(l) == text.charAt(r)) {
                count++;
            } else {
                count = 0;
            }

            if (count == len) {
                set.add(text.substring(l - len + 1, l + 1));
                count--;
            }
        }
    }
    return set.size();
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

[distinct-echo-substrings]: https://leetcode.com/problems/distinct-echo-substrings/
[encode-string-with-shortest-length]: https://leetcode.com/problems/encode-string-with-shortest-length/
[find-all-good-strings]: https://leetcode.com/problems/find-all-good-strings/
[longest-repeating-substring]: https://leetcode.com/problems/longest-repeating-substring/
[longest-substring-without-repeating-characters]: https://leetcode.com/problems/longest-substring-without-repeating-characters/
[stamping-the-sequence]: https://leetcode.com/problems/stamping-the-sequence/
[word-break]: https://leetcode.com/problems/word-break/
[word-break-ii]: https://leetcode.com/problems/word-break-ii/
