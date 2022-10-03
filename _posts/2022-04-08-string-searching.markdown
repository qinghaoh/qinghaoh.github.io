---
layout: post
title:  "String Searching"
tags: string
usemathjax: true
---
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
        // KMP to count the number of matches of pattern evil in text current built string (ending at c)
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

# Z Function

https://cp-algorithms.com/string/z-function.html

[Sum of Scores of Built Strings][sum-of-scores-of-built-strings]

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

[find-all-good-strings]: https://leetcode.com/problems/find-all-good-strings/
[longest-duplicate-substring]: https://leetcode.com/problems/longest-duplicate-substring/
[longest-happy-prefix]: https://leetcode.com/problems/longest-happy-prefix/
[maximum-deletions-on-a-string]: https://leetcode.com/problems/maximum-deletions-on-a-string/
[remove-all-occurrences-of-a-substring]: https://leetcode.com/problems/remove-all-occurrences-of-a-substring/
[shortest-palindrome]: https://leetcode.com/problems/shortest-palindrome/
[sum-of-scores-of-built-strings]: https://leetcode.com/problems/sum-of-scores-of-built-strings/
