---
title:  "Palindrome"
category: algorithm
tags: string
---
## Palindrome String

```java
public boolean isPalindrome(String s) {
    int left = 0, right = s.length() - 1;
    while (left < right) {
        if (s.charAt(left++) != s.charAt(right--)) {
            return false;
        }
    }
    return true;
}
```

## Palindrome Number

[Palindrome Number][palindrome-number]

```java
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
```

## Construction

[Prime Palindrome][prime-palindrome]

A positive integer (in decimal notation) is divisible by 11 iff the difference of the sum of the digits in even-numbered positions and the sum of digits in odd-numbered positions is divisible by 11.

```java
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
```

[Super Palindromes][super-palindromes]

```java
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
```

[Sum of k-Mirror Numbers][sum-of-k-mirror-numbers]

```java
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
```

## Trie

[Palindrome Pairs][palindrome-pairs]

```java
public List<List<Integer>> palindromePairs(String[] words) {
    TrieNode root = new TrieNode();

    // builds the trie of reversed words
    for (int i = 0; i < words.length; i++) {
        String word = words[i];
        TrieNode curr = root;
        for (int j = word.length() - 1; j >= 0; j--) {
            if (isPalindrome(word, 0, j + 1)) {
                // e.g. words[0] = "cdeedcba"
                // i = 0, j = 5
                // root -> curr = "ab", curr -> end = prefix = "cdeedc"
                curr.palindromePrefixWordIndices.add(i);
            }
            char ch = word.charAt(j);
            curr.children.putIfAbsent(ch, new TrieNode());
            curr = curr.children.get(ch);
        }
        curr.wordIndex = i;
    }

    // searches for pairs
    List<List<Integer>> list = new ArrayList<>();
    for (int i = 0; i < words.length; i++) {
        String word = words[i];
        TrieNode curr = root;
        int j = 0;
        while (j < word.length() && curr != null) {
            // e.g. word = "abcded", root -> curr = "abc" (original word was the reverse: "cba")
            if (curr.wordIndex >= 0 && isPalindrome(word, j, word.length())) {
                list.add(Arrays.asList(i, curr.wordIndex));
            }

            curr = curr.children.get(word.charAt(j++));
        }

        // curr char == the last char of word
        if (curr != null) {
            // the pair are the reverse of each other
            // e.g. word = "abc", root -> curr = "abc" (original word was the reverse: "cba")
            // `curr.wordIndex != i` ensures distinct indices
            if (curr.wordIndex >= 0 && curr.wordIndex != i) {
                list.add(Arrays.asList(i, curr.wordIndex));
            }

            // e.g. word = "abc", root -> curr = "abc", curr -> end = "ded"
            // (original word was the reverse: "dedcba")
            for (int index : curr.palindromePrefixWordIndices) {
                list.add(Arrays.asList(i, index));
            }
        }
    }
    return list;
}

class TrieNode {
    // index of word ending at the current node
    // -1 if no word ends here
    int wordIndex = -1;
    Map<Character, TrieNode> children = new HashMap<>();
    List<Integer> palindromePrefixWordIndices = new ArrayList<>();
}

private boolean isPalindrome(String s, int start, int end) {
    int i = start, j = end - 1;
    while (i < j) {
        if (s.charAt(i++) != s.charAt(j--)) {
            return false;
        }
    }
    return true;
}
```

[Find the Closest Palindrome][find-the-closest-palindrome]

```java
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
```

## Greedy

[Construct K Palindrome Strings][construct-k-palindrome-strings]

```java
public boolean canConstruct(String s, int k) {
    // bit vector
    int odd = 0;
    for (char c : s.toCharArray()) {
        odd ^= 1 << (c - 'a');
    }

    // if a bit is 1, it must be the center of a palindrome
    return s.length() >= k && Integer.bitCount(odd) <= k;
}
```

[Minimum Number of Moves to Make Palindrome][minimum-number-of-moves-to-make-palindrome]

```java
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
```

## Expand Around Center

[Palindromic Substring][palindromic-substring]

```java
public int countSubstrings(String s) {
    int n = s.length(), count = 0;
    for (int center = 0; center < 2 * n; center++) {
        int left = center / 2, right = left + center % 2;
        while (left >= 0 && right < n && s.charAt(left) == s.charAt(right)) {
            count++;
            left--;
            right++;
        }
    }
    return count;
}
```

[Longest Palindromic Substring][longest-palindromic-substring]

```java
// O(n ^ 2)
public String longestPalindrome(String s) {
    String result = "";
    int n = s.length(), max = 0;
    for (int center = 0; center < 2 * n; center++) {
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
```

[Maximum Number of Non-overlapping Palindrome Substrings][maximum-number-of-non-overlapping-palindrome-substrings]

```java
public int maxPalindromes(String s, int k) {
    int n = s.length(), count = 0, start = 0;
    for (int center = 0; center < 2 * n; center++) {
        int left = center / 2, right = left + center % 2;
        while (left >= start && right < n && s.charAt(left) == s.charAt(right)) {
            // 435. Non-overlapping Intervals
            if (right + 1 - left >= k) {
                count++;
                start = right + 1;
                break;
            }
            left--;
            right++;
        }
    }
    return count;
}
```

[Palindrome Partitioning II][palindrome-partitioning-ii]

```java
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
```

## Manacher's Algorithm

[Manacher's algorithm](https://en.wikipedia.org/wiki/Longest_palindromic_substring#Manacher's_algorithm): find all palindromic substrings in `O(n)`.

[Longest Palindromic Substring][longest-palindromic-substring]

```java
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
```

[Maximum Product of the Length of Two Palindromic Substrings][maximum-product-of-the-length-of-two-palindromic-substrings]

```java
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
```

## Dynamic Programming

Iteration pattern:

```java
for (int i = n - 1; i >= 0; i--) {
    for (int j = i; j < n; j++) {
```

[Palindrome Partitioning IV][palindrome-partitioning-iv]

```java
public boolean checkPartitioning(String s) {
    int n = s.length();
    // dp[i][j]: s.substring(i, j + 1)
    boolean[][] dp = new boolean[n][n];

    for (int i = n - 1; i >= 0; i--) {
        dp[i][i] == true;
        for (int j = i + 1; j < n; j++) {
            if (s.charAt(i) == s.charAt(j)) {
                dp[i][j] = (j == i + 1 ? true : dp[i + 1][j - 1]);
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
```

[Longest Palindromic Subsequence][longest-palindromic-subsequence]

```java
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
```

[Palindrome Removal][palindrome-removal]

```java
public int minimumMoves(int[] arr) {
    int n = arr.length;
    // dp[i][j]: minimum number of moves for arr[i...j]
    int[][] dp = new int[n][n];

    for (int i = n - 1; i >= 0; i--) {
        dp[i][i] = 1;
        if (i < n - 1) {
            dp[i][i + 1] = arr[i] == arr[i + 1] ? 1 : 2;
        }

        for (int j = i + 2; j < n; j++) {
            dp[i][j] = Integer.MAX_VALUE;

            if (arr[i] == arr[j]) {
                // arr[i] and arr[j] can be removed together with the last move of arr[(i + 1)...(j - 1)]
                dp[i][j] = dp[i + 1][j - 1];
            }

            // or, splits arr[i...j] in two parts
            for (int k = i; k < j; k++) {
                dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k + 1][j]);
            }
        }
    }

    return dp[0][n - 1];
}
```

[Minimum Changes to Make K Semi-palindromes][minimum-changes-to-make-k-semi-palindromes]

```c++
int minimumChanges(string s, int k) {
    int n = s.size();
    // Find all d's for each length.
    // A semi-palindrome has at least 2 characters.
    vector<vector<int>> divisors(n + 1, vector<int>(1, 1));
    for (int d = 2; d < n; d++) {
        for (int v = 2 * d; v <= n; v += d) {
            divisors[v].push_back(d);
        }
    }

    // [i, j, d]: s.substr(i, j - i + 1)
    // semi[i][j][0] = min(semi[i][j][d]), where d > 0
    vector<vector<vector<int>>> semi(n, vector<vector<int>>(n, vector<int>(n)));
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            semi[i][j][0] = n;
            for (int d : divisors[j - i + 1]) {
                if (i + d < n && j - d >= 0) {
                    semi[i][j][d] = semi[i + d][j - d][d];
                }
                for (int m = 0; m < d; m++) {
                    semi[i][j][d] += s[i + m] != s[j - d + 1 + m];
                }
                semi[i][j][0] = min(semi[i][j][0], semi[i][j][d]);
            }
        }
    }

    vector<vector<int>> dp(n + 1, vector<int>(k + 1));
    // m = 1
    for (int i = 1; i < n; i++) {
        dp[i + 1][1] = semi[0][i][0];
    }
    for (int m = 2; m <= k; m++) {
        for (int i = 2 * m; i <= n; i++) {
            dp[i][m] = n;
            for (int j = 2 * (m - 1); j < i - 1; j++) {
                dp[i][m] = min(dp[i][m], dp[j][m - 1] + semi[j][i - 1][0]);
            }
        }
    }
    return dp[n][k];
}
```

[Maximum Product of the Length of Two Palindromic Subsequences][maximum-product-of-the-length-of-two-palindromic-subsequences]

```java
public int maxProduct(String s) {
    int n = s.length(), max = 0;
    // iterates all masks
    for (int i = 0; i < (1 << n); i++) {
        StringBuilder ones = new StringBuilder(), zeros = new StringBuilder();
        for (int j = 0; j < n; j++) {
            ((i & (1 << j)) == 0 ? zeros : ones).append(s.charAt(j));
        }
        max = Math.max(max, longestPalindromeSubseq(ones) * longestPalindromeSubseq(zeros));
    }
    return max;
}
```

This problem can be solved by backtracking, too. It reflects the close connection between bitmask and backtracking.

[Palindrome Partitioning III][palindrome-partitioning-iii]

```java
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

    // dp2[i][m]: s.substring(0, i), m chunks
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
```

[Maximize Palindrome Length From Subsequences][maximize-palindrome-length-from-subsequences]

```java
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
```

[Count Different Palindromic Subsequences][count-different-palindromic-subsequences]

```java
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
```

[Count Palindromic Subsequences][count-palindromic-subsequences]

```c++
// @param d: iteration direction.
auto precompute(string s, int d = 1) {
    int n = s.length();
    vector<int> freqs(10);
    freqs[s[d > 0 ? 0 : n - 1] - '0'] = 1;

    // dp[i][j][k]: occurrences of ordered pair ('j', 'k') in s[0...i] (prefix) or s[i...(n - 1)] (suffix).
    vector<array<array<int, 10>, 10>> dp(n);

    for (int i = d > 0 ? 1 : n - 2; i >= 0 && i < n; i += d) {
        int digit = s[i] - '0';
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                dp[i][j][k] = dp[i - d][j][k] + (k == digit ? freqs[j] : 0);
            }
        }
        freqs[digit]++;
    }
    return dp;
}

public:
int countPalindromes(string s) {
    // Prefix and suffix
    auto pDp = precompute(s), sDp = precompute(s, -1);

    const int mod = 1e9 + 7;
    int cnt = 0, n = s.length();
    for (int i = 2; i < n - 2; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                cnt = (cnt + static_cast<long long>(pDp[i - 1][j][k]) * sDp[i + 1][j][k]) % mod;
            }
        }
    }
    return cnt;
}
```

{: .prompt-info }
> The `dp` array in this solution is not a prefix sum array of counts of ordered pairs. e.g. `s = "103301"`, `pDp[5][0][1] = 2` and `pDp[4][0][1] = 0`, we cannot say between index 4 and 5, `s` contains `pDp[5][0][1] - pDp[4][0][1] = 2` ordered pairs.

[construct-k-palindrome-strings]: https://leetcode.com/problems/construct-k-palindrome-strings/
[count-different-palindromic-subsequences]: https://leetcode.com/problems/count-different-palindromic-subsequences/
[count-palindromic-subsequences]: https://leetcode.com/problems/count-palindromic-subsequences/
[find-the-closest-palindrome]: https://leetcode.com/problems/find-the-closest-palindrome/
[longest-palindromic-subsequence]: https://leetcode.com/problems/longest-palindromic-subsequence/
[longest-palindromic-substring]: https://leetcode.com/problems/longest-palindromic-substring/
[maximize-palindrome-length-from-subsequences]: https://leetcode.com/problems/maximize-palindrome-length-from-subsequences/
[maximum-number-of-non-overlapping-palindrome-substrings]: https://leetcode.com/problems/maximum-number-of-non-overlapping-palindrome-substrings/
[maximum-product-of-the-length-of-two-palindromic-subsequences]: https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/
[maximum-product-of-the-length-of-two-palindromic-substrings]: https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/
[minimum-changes-to-make-k-semi-palindromes]: https://leetcode.com/problems/minimum-changes-to-make-k-semi-palindromes/
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
