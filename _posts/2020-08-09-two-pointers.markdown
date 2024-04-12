---
title:  "Two Pointers"
category: algorithm
tags: array
---
[Number of Zero-Filled Subarrays][number-of-zero-filled-subarrays]

```java
public long zeroFilledSubarray(int[] nums) {
    long count = 0;
    for (int i = 0, j = 0; j < nums.length; j++) {
        if (nums[j] != 0) {
            // i is to the right of j, which makes (j - i + 1) == 0
            i = j + 1;
        }
        count += j - i + 1;
    }
    return count;
}
```

[Remove All Adjacent Duplicates in String II][remove-all-adjacent-duplicates-in-string-ii]

```java
public String removeDuplicates(String s, int k) {
    int n = s.length();
    // count[i]: number of duplicates ending at i
    int[] count = new int[n];

    char[] c = s.toCharArray();
    int i = 0, j = 0;
    while (j < n) {
        // copies the j-th char to the i-th position
        c[i] = c[j];

        count[i] = i > 0 && c[i - 1] == c[i] ? count[i - 1] + 1 : 1;
        if (count[i] == k) {
            i -= k;
        }

        i++;
        j++;
    }

    return new String(c, 0, i);
}
```

Another solution is to use stack.

[Backspace String Compare][backspace-string-compare]

Scan from the end of the Strings.

```java
public boolean backspaceCompare(String S, String T) {
    int i = S.length() - 1, j = T.length() - 1;
    int back = 0;
    while (true) {
        back = 0;
        while (i >= 0 && (back > 0 || S.charAt(i) == '#')) {
            back += S.charAt(i) == '#' ? 1 : -1;
            i--;
        }
        back = 0;
        while (j >= 0 && (back > 0 || T.charAt(j) == '#')) {
            back += T.charAt(j) == '#' ? 1 : -1;
            j--;
        }
        if (i >= 0 && j >= 0 && S.charAt(i) == T.charAt(j)) {
            i--;
            j--;
        } else {
            break;
        }
    }
    return i == -1 && j == -1;
}
```

[Maximum Number of People That Can Be Caught in Tag][maximum-number-of-people-that-can-be-caught-in-tag]

```java
public int catchMaximumAmountofPeople(int[] team, int dist) {
    int n = team.length, count = 0;
    // i: it
    // j: non-it
    for (int i = 0, j = 0; i < n; i++) {
        if (team[i] == 1) {
            // out of reach
            while (j < i - dist) {
                j++;
            }

            // attempts to finds the next non-it that can be caught
            // non-strict < ensures j is still within range after j++
            while (j < Math.min(i + dist, n) && team[j] == 1) {
                j++;
            }

            if (j < n && team[j] == 0) {
                count++;
                j++;
            }
        }
    }
    return count;
}
```

[Trapping Rain Water][trapping-rain-water]

```c++
int trap(vector<int>& height) {
    int water = 0, leftMax = 0, rightMax = 0;
    int left = 0, right = height.size() - 1;
    while (left < right) {
        // Increment the pointer of the shorter side to converge towards the taller side.
        // This approach helps the left and right pointers meet at the maximum height,
        // thus eliminating the need for an additional pass to determine the highest point.
        if (height[left] < height[right]) {
            leftMax = max(leftMax, height[left]);
            water += leftMax - height[left++];
        } else {
            rightMax = max(rightMax, height[right]);
            water += rightMax - height[right--];
        }
    }
    return water;
}
```

[Container With Most Water][container-with-most-water]

```java
public int maxArea(int[] height) {
    int area = 0, left = 0, right = height.length - 1;
    while (left < right) {
        area = Math.max(area, Math.min(height[left], height[right]) * (right - left));
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }
    return area;
}
```

[Number of Subsequences That Satisfy the Given Sum Condition][number-of-subsequences-that-satisfy-the-given-sum-condition]

[Modular arithmetic properties](https://en.wikipedia.org/wiki/Modular_arithmetic#Properties)

```java
private int MOD = (int)1e9 + 7;

public int numSubseq(int[] nums, int target) {
    Arrays.sort(nums);

    // (A * B) mod C = (A mod C * B mod C) mod C
    int[] pows = new int[nums.length];
    pows[0] = 1;
    for (int k = 1; k < nums.length; k++) {
        pows[k] = pows[k - 1] * 2 % MOD;
    }

    int i = 0, j = nums.length - 1, count = 0;
    while (i <= j) {
        if (nums[i] + nums[j] > target) {
            j--;
        } else {
            count = (count + pows[j - i]) % MOD;
            i++;
        }
    }
    return count;
}
```

[Shortest Word Distance II][shortest-word-distance-ii]

```java
private Map<String, List<Integer>> map;

public WordDistance(String[] words) {
    map = new HashMap<>();
    for (int i = 0; i < words.length; i++) {
        map.computeIfAbsent(words[i], k -> new ArrayList<>()).add(i);
    }
}

public int shortest(String word1, String word2) {
    int i = 0, j = 0, min = Integer.MAX_VALUE;
    List<Integer> list1 = map.get(word1), list2 = map.get(word2);
    while (i < list1.size() && j < list2.size()) {
        int index1 = list1.get(i), index2 = list2.get(j);
        min = Math.min(min, Math.abs(index1 - index2));
        if (index1 < index2) {
            i++;
        } else {
            j++;
        }
    }
    return min;
}
```

[The Latest Time to Catch a Bus][the-latest-time-to-catch-a-bus]

```java
public int latestTimeCatchTheBus(int[] buses, int[] passengers, int capacity) {
    Arrays.sort(buses);
    Arrays.sort(passengers);

    int n = buses.length, m = passengers.length;
    for (int i = 0, j = 0; i < n; i++) {
        // fills the bus with passengers
        int count = 0;
        while (j < m && passengers[j] <= buses[i] && count < capacity) {
            j++;
            count++;
        }

        if (i == n - 1) {
            int t = 0;
            // searches for the first unique position
            if (count < capacity) {
                // arrives before bus time and after the last passenger
                t = buses[i];
                for (int k = j - 1; k >= 0 && passengers[k] == t; k--, t--);
            } else {
                // arrives before the last passenger time who is on board
                t = passengers[j - 1] - 1;
                for (int k = j - 2; k >= 0 && passengers[k] == t; k--, t--);
            }
            return t;
        }
    }
    return -1;
}
```

# Unique Characters

[Count Unique Characters of All Substrings of a Given String][count-unique-characters-of-all-substrings-of-a-given-string]

```java
private final int MOD = (int)1e9 + 7;

public int uniqueLetterString(String s) {
    int[][] index = new int[26][2];  // last two indexes of each character
    for (int i = 0; i < index.length; i++) {
        Arrays.fill(index[i], -1);
    }

    int sum = 0;
    for (int i = 0; i < s.length(); i++) {
        int c = s.charAt(i) - 'A';
        // e.g. AxxAxxxA: 2 * 3
        // index[c][0] --> index[c][1] --> i
        sum = (sum + (i - index[c][1]) * (index[c][1] - index[c][0]) % MOD) % MOD;
        index[c][0] = index[c][1];
        index[c][1] = i;
    }

    // calculates index[c][1] --> s.length()
    for (int c = 0; c < index.length; c++) {
        sum = (sum + (s.length() - index[c][1]) * (index[c][1] - index[c][0]) % MOD) % MOD;
    }
    return sum;
}
```

[Count Substrings That Differ by One Character][count-substrings-that-differ-by-one-character]

```java
public int countSubstrings(String s, String t) {
    int count = 0 ;
    // each unmatched pair (i, j) is counted once and only once
    // e.g.
    //  - (s[2], t[2]) is in the first loop
    //  - (s[2], t[5]) is in the first loop
    //  - (s[2], t[1]) is in the second loop
    for (int i = 0; i < s.length(); i++) {
        count += helper(s, t, i, 0);
    }

    for (int j = 1; j < t.length(); j++) {
        count += helper(s, t, 0, j);
    }
    return count;
}

// i and j are starting indexes of s and t respectively
private int helper(String s, String t, int i, int j) {
    int count = 0, prev = 0, curr = 0;
    while (i < s.length() && j < t.length()) {
        curr++;
        // e.g. UMMUMMMMU
        // the number of substrings which contains the middle 'U' as the only unmatched char
        // == (2 + 1) * (4 + 1) == 15
        if (s.charAt(i++) != t.charAt(j++)) {
            prev = curr;
            curr = 0;
        }
        count += prev;
    }
    return count;
}
```

[Sort Transformed Array][sort-transformed-array]

[One Edit Distance][one-edit-distance]

```java
public boolean isOneEditDistance(String s, String t) {
    if (s.equals(t) || Math.abs(s.length() - t.length()) > 1) {
        return false;
    }

    int i = 0, j = 0;
    boolean hasDiff = false;
    while (i < s.length() && j < t.length()) {
        if (s.charAt(i) != t.charAt(j)) {
            if (hasDiff) {
                return false;
            }
            hasDiff = true;
            if (s.length() > t.length()) {
                j--;
            } else if (s.length() < t.length()) {
                i--;
            }
        }
        i++;
        j++;
    }
    return true;
}
```

[Maximum Score of a Good Subarray][maximum-score-of-a-good-subarray]

```java
public int maximumScore(int[] nums, int k) {
    int n = nums.length, i = k, j = k;
    int score = nums[k], min = nums[k];
    while (i > 0 || j < n - 1) {
        if (i == 0) {
            j++;
        } else if (j == n - 1) {
            i--;
        } else if (nums[i - 1] < nums[j + 1]) {
            // invariant:
            // the current subarray always has the highest score
            // among all subarrays of the same size
            j++;
        } else {
            i--;
        }
        min = Math.min(min, Math.min(nums[i], nums[j]));
        score = Math.max(score, min * (j - i + 1));
    }
    return score;
}
```

# Two Passes

[Push Dominoes][push-dominoes]

```java
public String pushDominoes(String dominoes) {
    char[] chars = dominoes.toCharArray();
    int n = chars.length;

    int[] forces = new int[n];
    int force = 0;

    // -> right
    for (int i = 0; i < n; i++) {
        if (chars[i] == 'R') {
            force = n;
        } else if (chars[i] == 'L') {
            force = 0;
        } else {
            force = Math.max(force - 1, 0);
        }
        forces[i] += force;
    }

    // <- left
    force = 0;
    for (int i = n - 1; i >= 0; i--) {
        if (chars[i] == 'L') {
            force = n;
        } else if (chars[i] == 'R') {
            force = 0;
        } else {
            force = Math.max(force - 1, 0);
        }
        forces[i] -= force;

        chars[i] = forces[i] < 0 ? 'L' : (forces[i] > 0 ? 'R' : '.');
    }
    return new String(chars);
}
```

```
.L.R...LR..L..

[  0,  0,0,14, 13, 12, 11,  0,14, 13, 12,  0,0,0]
[-13,-14,0, 0,-11,-12,-13,-14, 0,-12,-13,-14,0,0]

[-13,-14,0,14,  2,  0, -2,-14,14,  1, -1,-14,0,0]

LL.RR.LLRRLL..
```

[Count Collisions on a Road][count-collisions-on-a-road]

```java
public int countCollisions(String directions) {
    int n = directions.length(), left = 0, right = n - 1;

    // left end cars which are moving towards left will not collide
    while (left < n && directions.charAt(left) == 'L') {
        left++;
    }

    // right end cars which are moving towards right will not collide
    while (right >= 0 && directions.charAt(right) == 'R') {
        right--;
    }

    // all cars in between will collide
    int count = 0;
    for (int i = left; i <= right; i++) {
        if (directions.charAt(i) != 'S') {
            count++;
        }
    }
    return count;
}
```

[Get the Maximum Score][get-the-maximum-score]

```java
private static final int MOD = (int)1e9 + 7;

public int maxSum(int[] nums1, int[] nums2) {
    int i = 0, j = 0, n = nums1.length, m = nums2.length;
    // sum of elements between the knots (comment elements) on each path
    long dp1 = 0, dp2 = 0;
    while (i < n || j < m) {
        if (i < n && (j == m || nums1[i] < nums2[j])) {
            dp1 += nums1[i++];
        } else if (j < m && (i == n || nums1[i] > nums2[j])) {
            dp2 += nums2[j++];
        } else {
            // common elements
            // resets both sums to the same value
            dp1 = dp2 = Math.max(dp1, dp2) + nums1[i];
            i++;
            j++;
        }
    }
    return (int)(Math.max(dp1, dp2) % MOD);
}
```

# K Pointers

[Intersection of Three Sorted Arrays][intersection-of-three-sorted-arrays]

```java
if (arr1[p1] == arr2[p2] && arr2[p2] == arr3[p3]) {
    list.add(arr1[p1]);
    p1++;
    p2++;
    p3++;
} else {
    if (arr1[p1] < arr2[p2]) {
        p1++;
    } else if (arr2[p2] < arr3[p3]) {
        p2++;
    } else {  // arr1[p1] >= arr2[p2] && arr2[p2] >= arr3[p3]
        p3++;
    }
}
```

[Longest Chunked Palindrome Decomposition][longest-chunked-palindrome-decomposition]

```java
public int longestDecomposition(String text) {
    // greedy
    int n = text.length(), count = 0;
    int left = 0, right = n;
    int low = left + 1, high = right - 1;
    while (low <= high) {
        if (text.substring(left, low).equals(text.substring(high, right))) {
            count += 2;
            left = low;
            right = high;
        }
        low++;
        high--;
    }

    if (left < right) {
        count++;
    }
    return count;
}
```

[Count Subarrays With Fixed Bounds][count-subarrays-with-fixed-bounds]

```java
public long countSubarrays(int[] nums, int minK, int maxK) {
    int n = nums.length, pOut = -1, pMin = -1, pMax = -1;
    long count = 0;
    for (int i = 0; i < n; i++) {
        // out of range
        if (nums[i] < minK || nums[i] > maxK) {
            pOut = i;
        }
        if (nums[i] == minK) {
            pMin = i;
        }
        if (nums[i] == maxK) {
            pMax = i;
        }

        // the subarrays end at i
        // the start index is in the range [pOut + 1, min(pMin, pMax)]
        count += Math.max(0, Math.min(pMin, pMax) - pOut);
    }
    return count;
}
```

# DFS

[Check if an Original String Exists Given Two Encoded Strings][check-if-an-original-string-exists-given-two-encoded-strings]

```java
private static final int MAX_DIFF = 1000;
private Boolean[][][] memo;

public boolean possiblyEquals(String s1, String s2) {
    // dp[i][j][d]:
    // d = diff + MAX_DIFF
    // if s1[i:] truncated by `diff` characters if diff > 0
    // and s2[j:] truncated by `-diff` characters if diff < 0 are equal
    this.memo = new Boolean[s1.length() + 1][s2.length() + 1][2 * MAX_DIFF];

    return dfs(s1, s2, 0, 0, 0);
}

// two pointers
private boolean dfs(String s1, String s2, int i, int j, int diff) {
    int n1 = s1.length(), n2 = s2.length();
    if (i == n1 && j == n2) {
        return diff == 0;
    }

    if (memo[i][j][diff + MAX_DIFF] != null) {
        return memo[i][j][diff + MAX_DIFF];
    }

    // literal matching on s1[i] and s2[j]
    if (i < n1 && j < n2 && diff == 0 && s1.charAt(i) == s2.charAt(j)) {
        if (dfs(s1, s2, i + 1, j + 1, 0)) {
            return memo[i][j][MAX_DIFF] = true;
        }
    }

    // literal matching on s1[i]
    if (i < n1 && Character.isLetter(s1.charAt(i)) && diff > 0 && dfs(s1, s2, i + 1, j, diff - 1)) {
        return memo[i][j][diff + MAX_DIFF] = true;
    }

    // literal matching on s2[j]
    if (j < n2 && Character.isLetter(s2.charAt(j)) && diff < 0 && dfs(s1, s2, i, j + 1, diff + 1)) {
        return memo[i][j][diff + MAX_DIFF] = true;
    }

    // wildcard matching on s1[i]
    for (int k = i, val = 0; k < n1 && Character.isDigit(s1.charAt(k)); k++) {
        val = val * 10 + (s1.charAt(k) - '0');
        if (dfs(s1, s2, k + 1, j, diff - val)) {
            return memo[i][j][diff + MAX_DIFF] = true;
        }
    }

    // wildcard matching on s2[j]
    for (int k = j, val = 0; k < n2 && Character.isDigit(s2.charAt(k)); k++) {
        val = val * 10 + (s2.charAt(k) - '0');
        if (dfs(s1, s2, i, k + 1, diff + val)) {
            return memo[i][j][diff + MAX_DIFF] = true;
        }
    }

    return memo[i][j][diff + MAX_DIFF] = false;
}
```

[backspace-string-compare]: https://leetcode.com/problems/backspace-string-compare/
[check-if-an-original-string-exists-given-two-encoded-strings]: https://leetcode.com/problems/check-if-an-original-string-exists-given-two-encoded-strings/
[container-with-most-water]: https://leetcode.com/problems/container-with-most-water/
[count-collisions-on-a-road]: https://leetcode.com/problems/count-collisions-on-a-road/
[count-subarrays-with-fixed-bounds]: https://leetcode.com/problems/count-subarrays-with-fixed-bounds/
[count-substrings-that-differ-by-one-character]: https://leetcode.com/problems/count-substrings-that-differ-by-one-character/
[count-unique-characters-of-all-substrings-of-a-given-string]: https://leetcode.com/problems/count-unique-characters-of-all-substrings-of-a-given-string/
[get-the-maximum-score]: https://leetcode.com/problems/get-the-maximum-score/
[intersection-of-three-sorted-arrays]: https://leetcode.com/problems/intersection-of-three-sorted-arrays/
[longest-chunked-palindrome-decomposition]: https://leetcode.com/problems/longest-chunked-palindrome-decomposition/
[maximum-number-of-people-that-can-be-caught-in-tag]: https://leetcode.com/problems/maximum-number-of-people-that-can-be-caught-in-tag/
[maximum-score-of-a-good-subarray]: https://leetcode.com/problems/maximum-score-of-a-good-subarray/
[number-of-subsequences-that-satisfy-the-given-sum-condition]: https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/
[number-of-zero-filled-subarrays]: https://leetcode.com/problems/number-of-zero-filled-subarrays/
[one-edit-distance]: https://leetcode.com/problems/one-edit-distance/
[push-dominoes]: https://leetcode.com/problems/push-dominoes/
[remove-all-adjacent-duplicates-in-string-ii]: https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/
[shortest-word-distance-ii]: https://leetcode.com/problems/shortest-word-distance-ii/
[sort-transformed-array]: https://leetcode.com/problems/sort-transformed-array/
[the-latest-time-to-catch-a-bus]: https://leetcode.com/problems/the-latest-time-to-catch-a-bus/
[trapping-rain-water]: https://leetcode.com/problems/trapping-rain-water/
