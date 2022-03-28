---
layout: post
title:  "Palindrome"
tags: string
---
# Palindrome String

{% highlight java %}
public boolean isPalindrome(String s) {
    int left = 0, right = s.length() - 1;
    while (left < right) {
        if (s.charAt(left++) != s.charAt(right--)) {
            return false;
        }
    }
    return true;
}
{% endhighlight %}

# Palindrome Number

[Palindrome Number][palindrome-number]

{% highlight java %}
public boolean isPalindrome(int x) {
    if (x < 0 || (x % 10 == 0 && x != 0)) {
        return false;
    }

    // half revert x
    int reverted = 0;
    while (x > reverted) {
        reverted = reverted * 10 + x % 10;
        x /= 10;
    }
    return x == reverted || x == reverted / 10;
}
{% endhighlight %}

# Construction

[Prime Palindrome][prime-palindrome]

A positive integer (in decimal notation) is divisible by 11 iff the difference of the sum of the digits in even-numbered positions and the sum of digits in odd-numbered positions is divisible by 11.

{% highlight java %}
public int primePalindrome(int N) {
    if (N >= 8 && N <= 11) {
        return 11;
    }

    // palindrome with even number of digits is divisible by 11
    // x has at most 5 digits
    for (int x = 1; x < 100000; x++) {
        // builds palindrome with odd number of digits
        String s = String.valueOf(x), r = new StringBuilder(s).reverse().toString();
        int k = Integer.valueOf(s + r.substring(1));
        if (k >= N && isPrime(k)) {
            return k;
        }
    }
    return -1;
}

private boolean isPrime(int x) {
    if (x < 2 || x % 2 == 0) {
        return x == 2;
    }

    for (int i = 3; i * i <= x; i += 2) {
        if (x % i == 0)  {
            return false;
        }
    }
    return true;
}
{% endhighlight %}

[Super Palindromes][super-palindromes]

{% highlight java %}
public int superpalindromesInRange(String left, String right) {
    List<Long> palindromes = new ArrayList<>();
    for (long i = 1; i <= 9; i++) {
        palindromes.add(i);
    }

    // (10 ^ 18) ^ 0.5 = 10 ^ 9
    // the upper limit of left half is 10 ^ (9 / 2 + 1)
    for (long i = 1; i < 10000; i++) {
        String l = Long.toString(i), r = new StringBuilder(l).reverse().toString();
        // even
        palindromes.add(Long.parseLong(l + r));
        // odd
        for (long d = 0; d < 10; d++) {
            palindromes.add(Long.parseLong(l + d + r));
        }
    }

    int count = 0;
    long low = Long.parseLong(left), high = Long.parseLong(right);
    for (long palindrome : palindromes) {
        long square = palindrome * palindrome;
        if (!isPalindrome(Long.toString(square))) {
            continue;
        }
        if (low <= square && square <= high) {
            count++;
        }
    }
    return count;
}

private boolean isPalindrome(String s) {
}
{% endhighlight %}

[Sum of k-Mirror Numbers][sum-of-k-mirror-numbers]

{% highlight java %}
private List<Long> prev = new ArrayList<>(), curr = new ArrayList<>();
private int[] arr = new int[64];

public long kMirror(int k, int n) {
    prev.add(0l);
    curr.add(0l);

    long sum = 0;
    // adds single-digit numbers to curr mirror list
    for (long i = 1; n > 0 && i < 10; i++) {
        curr.add(i);
        if (isMirror(i, k)) {
            sum += i;
            n--;
        }
    }
    return sum + generate(k, n, 10);
}

// generates mirrors to make sure curr mirror list is in order
// firstMul: power of 10
private long generate(int k, int n, long firstMul) {
    List<Long> mirrors = new ArrayList<>();
    long sum = 0;
    for (int i = 0; n > 0 && i < 10; i++) {
        for (int j = 0; n > 0 && j < prev.size(); j++) {
            long num = firstMul * i + prev.get(j) * 10 + i;

            // excludes leading zeros when checking if isMirror
            if (i != 0 && isMirror(num, k)) {
                sum += num;
                n--;
            }

            // includes leading zerors when generating next level
            mirrors.add(num);
        }
    }

    prev = curr;
    curr = mirrors;

    return sum + (n == 0 ? 0 : generate(k, n, firstMul * 10));
}

private boolean isMirror(long num, int base) {
    int j = -1;
    while (num != 0) {
        arr[++j] = (int)(num % base);
        num /= base;
    }

    int i = 0;
    while (i < j) {
        if (arr[i++] != arr[j--]) {
            return false;
        }
    }
    return true;
}
{% endhighlight %}

[Palindrome Pairs][palindrome-pairs]

{% highlight java %}
public List<List<Integer>> palindromePairs(String[] words) {
    Map<String, Integer> map = new HashMap<>();
    for (int i = 0; i < words.length; i++) {
        map.put(words[i], i);
    }

    // finds all "word - reverse(word)" pairs
    // this case is handled separately to avoid duplicates
    List<List<Integer>> list = new ArrayList<>();
    for(int i = 0; i < words.length; i++){
        int index = map.getOrDefault(new StringBuilder(words[i]).reverse().toString(), i);
        if (index != i) {
            list.add(Arrays.asList(i, index));
        }
    }

    for (int i = 0; i < words.length; i++) {
        String w = words[i];
        for (int j = 0; j <= w.length(); j++) {
            // s1.substring(0, j) is palindrome
            // s1.substring(j + 1) == reverse(s2) => (s2, s1)
            if (j > 0 && isPalindrome(w.substring(0, j))) {
                int index = map.getOrDefault(new StringBuilder(w.substring(j)).reverse().toString(), i);
                if (index != i) {
                    list.add(Arrays.asList(index, i));
                }
            }

            // s1.substring(j + 1) is palindrome
            // s1.substring(0, j) == reverse(s2) => (s1, s2)
            if (j < w.length() && isPalindrome(w.substring(j))) {
                int index = map.getOrDefault(new StringBuilder(w.substring(0, j)).reverse().toString(), i);
                if (index != i) {
                    list.add(Arrays.asList(i, index));
                }
            }
        }
    }

    return list;
}

private boolean isPalindrome(String s) {
}
{% endhighlight %}

[Find the Closest Palindrome][find-the-closest-palindrome]

{% highlight java %}
public String nearestPalindromic(String n) {
    int len = n.length();
    if (len == 1) {
        return String.valueOf(Integer.valueOf(n) - 1);
    }

    if (n.equals("1" + "0".repeat(len - 2) + "1")) {
        return "9".repeat(len - 1);
    }

    if (n.equals("9".repeat(len))) {
        return "1" + "0".repeat(len - 1) + "1";
    }

    if (n.equals("1" + "0".repeat(len - 1))) {
        return "9".repeat(len - 1);
    }

    String root = n.substring(0, (len + 1) / 2);
    String root1 = String.valueOf(Integer.valueOf(root) - 1);
    String root2 = String.valueOf(Integer.valueOf(root) + 1);

    long nl = Long.valueOf(n);
    long p2 = palindrome(root2, len / 2);
    long pl = p2;
    long diff = Math.abs(p2 - nl);

    if (!isPalindrome(n)) {
        long p = palindrome(root, len / 2);
        if (Math.abs(p - nl) <= diff) {
            diff = Math.abs(p - nl);
            pl = p;
        }
    }

    long p1 = palindrome(root1, len / 2);
    if (Math.abs(p1 - nl) <= diff) {
        diff = Math.abs(p1 - nl);
        pl = p1;
    }
    return String.valueOf(pl);
}

private long palindrome(String root, int len) {
    return Long.valueOf(root + new StringBuilder(root.substring(0, len)).reverse().toString());
}
{% endhighlight %}

# Greedy

[Construct K Palindrome Strings][construct-k-palindrome-strings]

{% highlight java %}
public boolean canConstruct(String s, int k) {
    // bit vector
    int odd = 0;
    for (char c : s.toCharArray()) {
        odd ^= 1 << (c - 'a');
    }

    // if a bit is 1, it must be the center of a palindrome
    return s.length() >= k && Integer.bitCount(odd) <= k;
}
{% endhighlight %}

[Minimum Number of Moves to Make Palindrome][minimum-number-of-moves-to-make-palindrome]

{% highlight java %}
public int minMovesToMakePalindrome(String s) {
    int count = 0;
    while (s.length() > 0) {
        // finds the occurrence of the end char closest to the start char
        int i = s.indexOf(s.charAt(s.length() - 1));
        if (i == s.length() - 1) {
            // the last char is the center
            count += i / 2;
        } else {
            count += i;
            // swaps the found char with the start char
            // deletes the two ends of the string
            s = s.substring(0, i) + s.substring(i + 1);
        }
        s = s.substring(0, s.length() - 1);
    }
    return count;
}
{% endhighlight %}

# Expand Around Center

[Palindromic Substring][palindromic-substring]

{% highlight java %}
public int countSubstrings(String s) {
    int n = s.length(), count = 0;
    for (int center = 0; center <= 2 * n - 1; center++) {
        int left = center / 2, right = left + center % 2;
        while (left >= 0 && right < n && s.charAt(left) == s.charAt(right)) {
            count++;
            left--;
            right++;
        }
    }
    return count;
}
{% endhighlight %}

[Longest Palindromic Substring][longest-palindromic-substring]

{% highlight java %}
// O(n ^ 2)
public String longestPalindrome(String s) {
    String result = "";
    int n = s.length(), max = 0;
    for (int center = 0; center <= 2 * n - 1; center++) {
        int left = center / 2, right = left + center % 2;
        while (left >= 0 && right < n && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }

        if (right - left - 1 > max) {
            max = right - left - 1;
            result = s.substring(left + 1, right);
        }
    }
    return result;
}
{% endhighlight %}

[Palindrome Partitioning II][palindrome-partitioning-ii]

{% highlight java %}
public int minCut(String s) {
    int n = s.length();
    // dp[i]: min cut of s.substring(0, i + 1)
    int[] dp = new int[n];
    // initialization
    // partitions into one-char groups
    for (int i = 0; i < n; i++) {
        dp[i] = i;
    }

    for (int center = 0; center <= 2 * n - 1; center++) {
        int left = center / 2, right = left + center % 2;
        while (left >= 0 && right < n && s.charAt(left) == s.charAt(right)) {
            // s.substring(0, left) + current palindrome == s.substring(0, right + 1);
            dp[right] = Math.min(dp[right], left == 0 ? 0 : dp[left - 1] + 1);
            left--;
            right++;
        }
    }
    return dp[n - 1];
}
{% endhighlight %}

# Manacher's Algorithm

[Manacher's algorithm](https://en.wikipedia.org/wiki/Longest_palindromic_substring#Manacher's_algorithm): find all palindromic substrings in `O(n)`.

[Longest Palindromic Substring][longest-palindromic-substring]

{% highlight java %}
// O(n)
public String longestPalindrome(String s) {
    // string ms = s with a bogus character (eg. '#') inserted between each character
    // (including outer boundaries)
    StringJoiner sj = new StringJoiner("#", "#", "#");
    for (char ch : s.toCharArray()) {
        sj.add(String.valueOf(ch));
    }
    String ms = sj.toString();

    int n = ms.length();

    // radii[i]: radius of the longest palindrome centered at ms[i]
    int[] radii = new int[n];

    // center and right bound of the longest palindromic substring so far
    int c = -1, r = -1;
    int max = Integer.MIN_VALUE, pos = -1;

    for (int i = 0; i < n; i++) {
        // 2 * c - i is the mirrored center of i
        // radius is bounded by the outer palindrome
        // or it's a candidate
        radii[i] = i < r ? Math.min(radii[2 * c - i], r - i) : 1;

        // determines the longest palindrome in [center - radius, center + radius]
        while (i - radii[i] >= 0 && i + radii[i] < n && ms.charAt(i + radii[i]) == ms.charAt(i - radii[i])) {
            radii[i]++;
        }

        // updates r and c if current right bound is beyond r
        if (i + radii[i] > r) {
            r = i + radii[i];
            c = i;
        }

        // tracks the max and its position
        if (radii[i] > max) {
            max = radii[i];
            pos = i;
        }
    }

    StringBuilder sb = new StringBuilder();
    for (int i = pos - radii[pos] + 1; i <= pos + radii[pos] - 1; i++) {
        if (ms.charAt(i) != '#') {
            sb.append(ms.charAt(i));
        }
    }
    return sb.toString();
}
{% endhighlight %}

[Maximum Product of the Length of Two Palindromic Substrings][maximum-product-of-the-length-of-two-palindromic-substrings]

{% highlight java %}
public long maxProduct(String s) {
    // Manacher's Algorithm (https://cp-algorithms.com/string/manacher.html)
    int n = s.length();
    int[] radii = new int[n];
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 1 : Math.min(radii[l + r - i], r - i + 1);
        while (0 <= i - k && i + k < n && s.charAt(i - k) == s.charAt(i + k)) {
            k++;
        }

        radii[i] = k--;
        if (i + k > r) {
            l = i - k;
            r = i + k;
        }
    }

    // r[i]: max length of palindrome whose left is i
    int[] r = new int[n];
    Queue<Integer> q1 = new LinkedList<>(), q2 = new LinkedList<>();;
    // from right to left
    // finds the max palindrome whose left bound >= i
    for (int i = n - 1; i > 0; i--) {
        // left bound = center - radius
        // loops until the current index is covered by the palindrome in queue front
        while (!q1.isEmpty() && q1.peek() - radii[q1.peek()] >= i) {
            q1.poll();
        }
        r[i] = 1 + (q1.isEmpty() ? 0 : (q1.peek() - i) * 2);
        q1.offer(i);
    }

    // l: max length of palindrome whose right is i
    long l = 0, product = 0;
    // from left to right
    // finds the max palindrome whose right bound <= i
    for (int i = 0; i < n - 1; i++) {
        // right bound = center + radius
        // loops until the current index is covered by the palindrome in queue front
        while (!q2.isEmpty() && q2.peek() + radii[q2.peek()] <= i) {
            q2.poll();
        }
        l = Math.max(l, 1 + (q2.isEmpty() ? 0 : (i - q2.peek()) * 2));
        product = Math.max(product, l * r[i + 1]);
        q2.offer(i);
    }
    return product;
}
{% endhighlight %}

# Dynamic Programming

[Palindrome Partitioning IV][palindrome-partitioning-iv]

{% highlight java %}
public boolean checkPartitioning(String s) {
    int n = s.length();
    // dp[i][j]: s.substring(i, j + 1)
    boolean[][] dp = new boolean[n][n];

    for (int i = n - 1; i >= 0; i--) {
        for (int j = i; j < n; j++) {
            if (s.charAt(i) == s.charAt(j)) {
                dp[i][j] = (i > j - 2 ? true : dp[i + 1][j - 1]);
            } else {
                dp[i][j] = false;
            }
        }
    }

    for (int i = 1; i < n - 1; i++) {
        for (int j = i; j < n - 1; j++) {
            if (dp[0][i - 1] && dp[i][j] && dp[j + 1][n - 1]) {
                return true;
            }
        }
    }
    return false;
}
{% endhighlight %}

[Longest Palindromic Subsequence][longest-palindromic-subsequence]

{% highlight java %}
public int longestPalindromeSubseq(String s) {
    int n = s.length();
    // dp[i][j]: s.substring(i, j + 1)
    int[][] dp = new int[n][n];

    for (int i = n - 1; i >= 0; i--) {
        dp[i][i] = 1;
        for (int j = i + 1; j < n; j++) {
            if (s.charAt(i) == s.charAt(j)) {
                dp[i][j] = dp[i + 1][j - 1] + 2;
            } else {
                dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp[0][n - 1];
}
{% endhighlight %}

[Maximum Product of the Length of Two Palindromic Subsequences][maximum-product-of-the-length-of-two-palindromic-subsequences]

{% highlight java %}
public int maxProduct(String s) {
    int n = s.length(), max = 0;
    for (int i = 0; i < (1 << n); i++) {
        StringBuilder left = new StringBuilder(), right = new StringBuilder();
        for (int j = 0; j < n; j++) {
            char c = s.charAt(j);
            if ((i & (1 << j)) != 0) {
                left.append(c);
            } else {
                right.append(c);
            }
        }
        max = Math.max(max, longestPalindromeSubseq(left) * longestPalindromeSubseq(right));
    }
    return max;
}
{% endhighlight %}

[Palindrome Removal][palindrome-removal]

{% highlight java %}
public int minimumMoves(int[] arr) {
    int n = arr.length;
    // dp[i][j]: minimum number of moves for arr[i...j]
    int[][] dp = new int[n][n];

    for (int i = 0; i < n; i++) {
        dp[i][i] = 1;
    }

    for (int i = 0; i < n - 1; i++) {
        dp[i][i + 1] = arr[i] == arr[i + 1] ? 1 : 2;
    }

    for (int len = 3; len <= n; len++) {
        for (int i = 0, j = i + len - 1; j < n; i++, j++) {
            dp[i][j] = Integer.MAX_VALUE;

            if (arr[i] == arr[j]) {
                dp[i][j] = dp[i + 1][j - 1];
            }

            for (int k = i; k < j; k++) {
                dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k + 1][j]);
            }
        }
    }

    return dp[0][n - 1];
}
{% endhighlight %}

[Palindrome Partitioning III][palindrome-partitioning-iii]

{% highlight java %}
public int palindromePartition(String s, int k) {
    int n = s.length();

    // dp1[i][j]: minimum steps to make s.substring(i, j + 1) palindrome
    int[][] dp1 = new int[n][n];

    for (int i = n - 1; i >= 0; i--) {
        // dp1[i][i] == 0
        for (int j = i + 1; j < n; j++) {
            dp1[i][j] = dp1[i + 1][j - 1] + (s.charAt(i) == s.charAt(j) ? 0 : 1);
        }
    }

    // dp[i][m]: s.substring(0, i), m chunks
    int[][] dp2 = new int[n + 1][k + 1];
    for (int i = 1; i <= n; i++) {
        dp2[i][1] = dp1[0][i - 1];
    }

    for (int m = 2; m <= k; m++) {
        // dp[m][m] = 0
        for (int i = m + 1; i <= n; i++) {
            dp2[i][m] = Integer.MAX_VALUE;
            for (int j = m - 1; j < i; j++) {
                dp2[i][m] = Math.min(dp2[i][m], dp2[j][m - 1] + dp1[j][i - 1]);
            }
        }
    }
    return dp2[n][k];
}
{% endhighlight %}

[Maximize Palindrome Length From Subsequences][maximize-palindrome-length-from-subsequences]

{% highlight java %}
public int longestPalindrome(String word1, String word2) {
    String s = word1 + word2;
    int n = s.length(), n1 = word1.length();
    // dp[i][j]: s.substring(i, j + 1)
    int[][] dp = new int[n][n];

    int max = 0;
    for (int i = n - 1; i >= 0; i--) {
        dp[i][i] = 1;
        for (int j = i + 1; j < n; j++) {
            if (s.charAt(i) == s.charAt(j)) {
                dp[i][j] = dp[i + 1][j - 1] + 2;
                if (i < n1 && j >= n1) {
                    max = Math.max(max, dp[i][j]);
                }
            } else {
                dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }
    return max;
}
{% endhighlight %}

[Count Different Palindromic Subsequences][count-different-palindromic-subsequences]

{% highlight java %}
private static final int MOD = (int)1e9 + 7, NUM = 4;

public int countPalindromicSubsequences(String S) {
    int n = S.length();
    int[] prev = new int[n + 1], next = new int[n + 1];

    int[] index = new int[NUM];
    Arrays.fill(index, -1);
    for (int i = 0; i < n; i++) {
        prev[i] = index[S.charAt(i) - 'a'];
        index[S.charAt(i) - 'a'] = i;
    }

    Arrays.fill(index, n);
    for (int i = n - 1; i >= 0; i--) {
        next[i] = index[S.charAt(i) - 'a'];
        index[S.charAt(i) - 'a'] = i;
    }

    int[][] dp = new int[n][n];

    // "a"
    for (int i = 0; i < n; i++) {
        dp[i][i] = 1;
    }

    for (int len = 1; len < n; len++) {
        for (int i = 0, j = i + len; j < n; i++, j++) {
            if (S.charAt(i) != S.charAt(j)) {
                dp[i][j] = dp[i][j - 1] + dp[i + 1][j] - dp[i + 1][j - 1];
            } else {
                // [i, j]
                int low = next[i], high = prev[j];
                dp[i][j] = dp[i + 1][j - 1] * 2;  // w/ and w/o S(i) & S(j)
                if (low == high) {  // a...a...a, only one char 'a' inside
                    dp[i][j]++;  // +{"aa"}
                } else if (low > high) {  // a...a, no char 'a' inside
                    dp[i][j] += 2;  // +{"a", "aa"}
                } else {  // a...a...a...a
                    dp[i][j] -= dp[low + 1][high - 1];
                }
            }
            dp[i][j] = Math.floorMod(dp[i][j], MOD);
        }
    }
    return dp[0][n - 1];
}
{% endhighlight %}

[construct-k-palindrome-strings]: https://leetcode.com/problems/construct-k-palindrome-strings/
[count-different-palindromic-subsequences]: https://leetcode.com/problems/count-different-palindromic-subsequences/
[find-the-closest-palindrome]: https://leetcode.com/problems/find-the-closest-palindrome/
[longest-palindromic-subsequence]: https://leetcode.com/problems/longest-palindromic-subsequence/
[longest-palindromic-substring]: https://leetcode.com/problems/longest-palindromic-substring/
[maximize-palindrome-length-from-subsequences]: https://leetcode.com/problems/maximize-palindrome-length-from-subsequences/
[maximum-product-of-the-length-of-two-palindromic-subsequences]: https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/
[maximum-product-of-the-length-of-two-palindromic-substrings]: https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/
[minimum-number-of-moves-to-make-palindrome]: https://leetcode.com/problems/minimum-number-of-moves-to-make-palindrome/
[palindrome-number]: https://leetcode.com/problems/palindrome-number/
[palindrome-pairs]: https://leetcode.com/problems/palindrome-pairs/
[palindrome-partitioning-ii]: https://leetcode.com/problems/palindrome-partitioning-ii/
[palindrome-partitioning-iii]: https://leetcode.com/problems/palindrome-partitioning-iii/
[palindrome-partitioning-iv]: https://leetcode.com/problems/palindrome-partitioning-iv/
[palindrome-removal]: https://leetcode.com/problems/palindrome-removal/
[palindromic-substring]: https://leetcode.com/problems/palindromic-substring/
[prime-palindrome]: https://leetcode.com/problems/prime-palindrome/
[sum-of-k-mirror-numbers]: https://leetcode.com/problems/sum-of-k-mirror-numbers/
[super-palindromes]: https://leetcode.com/problems/super-palindromes/
