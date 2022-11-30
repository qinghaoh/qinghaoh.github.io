---
layout: post
title:  "Prefix Sum"
---
# Fundamentals

The basic template for prefix sum creation is:

{% highlight java %}
int[] p = new int[n + 1];
for (int i = 0: i < n; i++) {
    p[i + 1] = p[i] + nums[i];
}

// sum[i...j] = p[j + 1] - p[i]
{% endhighlight %}

We don't always have to create an array to record prefix sums. Sometimes we can use a running prefix sum instead. For example:

[Subarray Sum Equals K][subarray-sum-equals-k]

{% highlight java %}
public int subarraySum(int[] nums, int k) {
    // prefix sum : count
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, 1);  // p[0]

    int sum = 0, count = 0;
    for (int num : nums) {
        sum += num;
        count += map.getOrDefault(sum - k, 0);
        map.put(sum, map.getOrDefault(sum, 0) + 1);
    }

    return count;
}
{% endhighlight %}

Similar problem: [Maximum Size Subarray Sum Equals k][maximum-size-subarray-sum-equals-k]

{% highlight java %}
public int maxSubArrayLen(int[] nums, int k) {
    // prefix sum : index of first occurrence
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, -1);

    int max = 0, sum = 0;;
    for (int i = 0; i < nums.length; i++) {
        sum += nums[i];
        if (map.containsKey(sum - k)) {
            max = Math.max(max, i - map.get(sum - k));
        } else {
            map.put(sum, i);
        }
    }
    return max;
}
{% endhighlight %}

[Contiguous Array][contiguous-array]

{% highlight java %}
public int findMaxLength(int[] nums) {
    // prefix sum : index of first occurrence
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, -1);

    // count(ones) - count(zeros)
    int diff = 0, max = 0;
    for (int i = 0; i < nums.length; i++) {
        diff += nums[i] == 0 ? -1 : 1;
        if (map.containsKey(diff)) {
            max = Math.max(max, i - map.get(diff));
        } else {
            map.put(diff, i);
        }
    }
    return max;
}
{% endhighlight %}

[Longest Well-Performing Interval][longest-well-performing-interval]

{% highlight java %}
public int longestWPI(int[] hours) {
    // prefix sum : index of first occurrence
    Map<Integer, Integer> map = new HashMap<>();
    int max = 0, score = 0;
    for (int i = 0; i < hours.length; i++) {
        // finds the longest subarray with positive sum
        score += hours[i] > 8 ? 1 : -1;
        if (score > 0) {
            max = i + 1;
        } else {
            map.putIfAbsent(score, i);
            // when score <= 0,
            // the first occurrence of (score - 1) is always before (score - k), where k > 1
            // i.e. map[score - 1] is the farthest from current position
            if (map.containsKey(score - 1)) {
                max = Math.max(max, i - map.get(score - 1));
            }
        }
    }
    return max;
}
{% endhighlight %}

[Maximize the Beauty of the Garden][maximize-the-beauty-of-the-garden]

{% highlight java %}
public int maximumBeauty(int[] flowers) {
    // flower : first prefix sum of this flower
    Map<Integer, Integer> map = new HashMap<>();
    int sum = 0, max = Integer.MIN_VALUE;
    for (int f : flowers) {
        if (map.containsKey(f)) {
            max = Math.max(max, sum - map.get(f) + 2 * f);
        }

        // counts positive beauty only,
        // because we can always not pick the negative beauty in a window
        if (f > 0) {
            sum += f;
        }

        map.putIfAbsent(f, sum);
    }
    return max;
}
{% endhighlight %}

# Variants

**Product**

[Product of the Last K Numbers][product-of-the-last-k-numbers]

{% highlight java %}
private List<Integer> p = new ArrayList<>(); 

public ProductOfNumbers() {
    p.add(1);
}

public void add(int num) {
    if (num > 0) {
        p.add(p.get(p.size() - 1) * num);
    } else {
        p = new ArrayList<>();
        p.add(1);
    }
}

public int getProduct(int k) {
    int n = p.size();
    return k < n ? p.get(n - 1) / p.get(n - k - 1) : 0;
}
{% endhighlight %}

[Product of Array Except Self][product-of-array-except-self]

{% highlight java %}
public int[] productExceptSelf(int[] nums) {
    int n = nums.length;
    int[] answer = new int[n];

    // prefix product, no last element
    answer[0] = 1;
    for (int i = 0; i < n - 1; i++) {
        answer[i + 1] = nums[i] * answer[i];
    }

    int product = 1;
    for (int i = n - 1; i >= 0; i--) {
        answer[i] *= product;
        product *= nums[i];
    }
    return answer;
}
{% endhighlight %}

**Mod**

[Make Sum Divisible by P][make-sum-divisible-by-p]

{% highlight java %}
public int minSubarray(int[] nums, int p) {
    // target remainder
    int r = 0;
    for (int num : nums) {
        r = (r + num) % p;
    }
    if (r == 0) {
        return 0;
    }

    // prefix mod : index of last occurrence
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, -1);

    int n = nums.length, min = n, m = 0;
    for (int i = 0; i < n; i++) {
        m = (m + nums[i] % p) % p;
        int d = (m - r + p) % p;
        if (map.containsKey(d)) {
            min = Math.min(min, i - map.get(d));
        }
        map.put(m, i);
    }
    return min == n ? -1 : min;
}
{% endhighlight %}

**Exclusive Or**

[Count Triplets That Can Form Two Arrays of Equal XOR][count-triplets-that-can-form-two-arrays-of-equal-xor]

[Can Make Palindrome from Substring][can-make-palindrome-from-substring]

{% highlight java %}
public List<Boolean> canMakePaliQueries(String s, int[][] queries) {
    int n = s.length();
    // 26 bits to represent prefix xor
    int[] p = new int[n + 1];
    for (int i = 0; i < n; i++) {
        p[i + 1] = p[i] ^ (1 << (s.charAt(i) - 'a'));
    }

    List<Boolean> answer = new ArrayList<>();
    for (int[] q : queries) {
        int odd = Integer.bitCount(p[q[1] + 1] ^ p[q[0]]);
        answer.add(odd <= 2 * q[2] + 1);
    }
    return answer;
}
{% endhighlight %}

[Number of Wonderful Substrings][number-of-wonderful-substrings]

{% highlight java %}
private static final int NUM_CHARS = 10;

public long wonderfulSubstrings(String word) {
    // map[i]: count of bit mask i
    long[] map = new long[1 << NUM_CHARS];
    map[0] = 1;

    // bits to represent prefix xor
    long count = 0;
    int p = 0;
    for (int i = 0; i < word.length(); i++) {
        p ^= 1 << (word.charAt(i) - 'a');
        count += map[p];
        for (int j = 0; j < NUM_CHARS; j++) {
            count += map[p ^ (1 << j)];
        }
        map[p]++;
    }
    return count;
}
{% endhighlight %}

**Multi-dimension**

[Sum of Beauty of All Substrings][sum-of-beauty-of-all-substrings]

{% highlight java %}
int[][] p = new int[26][n + 1];
for (int i = 0; i < n; i++) {
    for (int k = 0; k < p.length; k++) {
        p[k][i + 1] = p[k][i] + (k == s.charAt(i) - 'a' ? 1 : 0);
    }
}
{% endhighlight %}

[Minimum Absolute Difference Queries][minimum-absolute-difference-queries]

{% highlight java %}
private static final int MAX_NUM = 100;

public int[] minDifference(int[] nums, int[][] queries) {
    int n = nums.length, m = queries.length;
    int[][] p = new int[n + 1][MAX_NUM + 1];
    int[] count = new int[MAX_NUM + 1];
    for (int i = 0; i < n; i++) {
        count[nums[i]]++;
        p[i + 1] = Arrays.copyOf(count, count.length);
    }

    int[] ans = new int[m];
    for (int i = 0; i < m; i++) {
        ans[i] = Integer.MAX_VALUE;
        for (int j = 0, prev = 0; j < p[0].length; j++) {
            if (p[queries[i][1] + 1][j] != p[queries[i][0]][j]) {
                if (prev != 0) {
                    ans[i] = Math.min(ans[i], j - prev);
                }
                prev = j;
            }
        }
        if (ans[i] == Integer.MAX_VALUE) {
            ans[i] = -1;
        }
    }
    return ans;
}
{% endhighlight %}

**Frequency**

[Sum of Floored Pairs][sum-of-floored-pairs]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int sumOfFlooredPairs(int[] nums) {
    int max = Arrays.stream(nums).max().getAsInt();

    int[] freq = new int[max + 1];
    for (int num : nums) {
        freq[num]++;
    }

    for (int i = 1; i < freq.length; i++) {
        freq[i] += freq[i - 1];
    }

    // if floor(nums[i] / nums[j]) = k,
    // then k * nums[j] <= nums[i] < ((k + 1) * nums[j])
    Integer[] dp = new Integer[max + 1];
    int count = 0;
    for (int num : nums) {
        if (dp[num] != null) {
            count = (count + dp[num]) % MOD;
            continue;
        }

        // initial interval: [low, high]
        int curr = 0, k = 1, low = num, high = Math.min(2 * num - 1, max);
        while (low <= max) {
            curr = (int)(curr + ((freq[high] - freq[low - 1]) * (long)k) % MOD) % MOD;
            low += num;
            high = Math.min(high + num, max);
            k++;
        }
        count = (count + (dp[num] = curr)) % MOD;
    }
    return count;
}
{% endhighlight %}

**Linked List**

[Remove Zero Sum Consecutive Nodes from Linked List][remove-zero-sum-consecutive-nodes-from-linked-list]

{% highlight java %}
public ListNode removeZeroSumSublists(ListNode head) {
    ListNode dummy = new ListNode();
    dummy.next = head;

    // prefix sum : last node with this value
    Map<Integer, ListNode> map = new HashMap<>();
    map.put(0, dummy);

    int sum = 0;
    for (ListNode curr = dummy; curr != null; curr = curr.next) {
        sum += curr.val;
        map.put(sum, curr);
    }

    // links to the last node that has the same prefix sum
    sum = 0;
    for (ListNode curr = dummy; curr != null; curr = curr.next) {
        sum += curr.val;
        curr.next = map.get(sum).next;
    }
    return dummy.next;
}
{% endhighlight %}

[Maximum Absolute Sum of Any Subarray][maximum-absolute-sum-of-any-subarray]

```
abs(subarray) = p[i] - p[j] <= max(p) - min(p)
```

**Diff**

[Substring With Largest Variance][substring-with-largest-variance]

{% highlight java %}
public int largestVariance(String s) {
    Set<Character> chars = s.chars().mapToObj(ch -> (char)ch).collect(Collectors.toSet());
    int max = 0;
    // the order is ch1 then ch2
    // i.e. ch1 == 'a', ch2 == 'b' is different from ch1 == 'b', ch1 == 'a'
    for (char ch1 : chars) {
        for (char ch2 : chars) {
            if (ch1 != ch2) {
                // variance = #ch1 - #ch2
                // splits the string into two parts
                // variance = (#ch1_in_left + #ch1_in_right) - (#ch2_in_left + #ch2_in_right)
                //          = (#ch1_in_left - #ch2_in_left) + (#ch1_in_right - #ch2_in_right)
                // now the problem is to find the min left (and thus max right)
                int variance = 0, left = 0, minLeft = s.length(); 
                for (char ch : s.toCharArray()) {
                    if (ch == ch1) {
                        variance++;
                    } else if (ch == ch2) {
                        minLeft = Math.min(minLeft, left);
                        left = --variance; 
                    }
                    max = Math.max(max, variance - minLeft); 
                }
            }
        }
    }
    return max;
}
{% endhighlight %}

This problem can also be solved by [Kadane's Algorithm](../../../2020/10/03/kadanes.html).

**Ordered Pair**

[Count Palindromic Subsequences][count-palindromic-subsequences]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int countPalindromes(String s) {
    int n = s.length();
    int[][][] pp = prefixSum(s), ss = prefixSum(new StringBuilder(s).reverse().toString());

    long count = 0;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 2; k < n - 2; k++) {
                count = (count + ((long)pp[i][j][k] * ss[i][j][n - k - 1]) % MOD) % MOD;
            }
        }
    }
    return (int)count;
}

private int[][][] prefixSum(String s) {
    int n = s.length();
    // p[i][j]: occurrences of 'i' in s.substring(0, j)
    int[][] p = new int[10][n + 1];
    // pp[i][j][k]: occurrences of ordered pair ('i', 'j') in s.substring(0, k)
    int[][][] pp = new int[10][10][n + 1];
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < 10; i++) {
            p[i][k + 1] = p[i][k] + (s.charAt(k) - '0' == i ? 1 : 0);
            for (int j = 0; j < 10; j++) {
                pp[i][j][k + 1] = (pp[i][j][k] + (s.charAt(k) - '0' == j ? p[i][k] : 0)) % MOD;
            }
        }
    }
    return pp;
}
{% endhighlight %}

**Prefix Sum**

[Sum of Total Strength of Wizards][sum-of-total-strength-of-wizards]

{% highlight java %}
private int MOD = (int)1e9 + 7;

public int totalStrength(int[] strength) {
    int n = strength.length;
    long[] pp = new long[n + 1];

    // prefix sum
    // strength[i] = p[i + 1] - p[i]
    for (int i = 0; i < n; i++) {
        pp[i + 1] = (pp[i] + strength[i]) % MOD;
    }

    // prefix sum of prefix sum
    // p[i + 1] = pp[i + 1] - pp[i]
    for (int i = 0; i < n; i++) {
        pp[i + 1] = (pp[i] + pp[i + 1]) % MOD;
    }

    Deque<Integer> st = new ArrayDeque<>();
    long res = 0;
    for (int i = 0; i <= n; i++) {
        while (!st.isEmpty() && (i == n || strength[i] < strength[st.peek()])) {
            int j = st.pop();
            int k = st.isEmpty() ? -1 : st.peek();
            // j is min in (k, i)
            //
            // sum of subarrays including j that start with m:
            // s(m) = sum(sum(m, m + 1, ..., t))_{j <= t < i}
            //      = sum(p[t + 1] - p[m])_{j <= t < i}
            //      = sum(p[t + 1])_{j <= t < i} - p[m] * (i - j)
            //      = pp[i] - pp[j] - p[m] * (i - j)
            //
            // sum of all subarrays including j:
            // s = sum(s(m))_(k < m <= j)
            //   = sum(pp[i] - pp[j] - (i - j) * p[m])_{k < m <= j}
            //   = (pp[i] - pp[j]) * (j - k) - sum(p[m])_{k < m <= j} * (i - j)
            //   = (pp[i] - pp[j]) * (j - k) - (pp[j] - pp[k]) * (i - j)
            res = (res + (MOD + (pp[i] - pp[j] + MOD) % MOD * (j - k) % MOD - (pp[j] - pp[Math.max(0, k)] + MOD) % MOD * (i - j) % MOD) * strength[j]) % MOD;
        }
        st.push(i);
    }
    return (int)res;
}
{% endhighlight %}

# Range Sum

[Number of Ways of Cutting a Pizza][number-of-ways-of-cutting-a-pizza]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;
private Integer[][][] memo;

public int ways(String[] pizza, int k) {
    int m = pizza.length, n = pizza[0].length();
    this.memo = new Integer[k][m][n];

    // p[i][j] is the total number of apples in pizza[i:][j:]
    int[][] p = new int[m + 1][n + 1];
    for (int i = m - 1; i >= 0; i--) {
        for (int j = n - 1; j >= 0; j--) {
            p[i][j] = p[i][j + 1] + p[i + 1][j] - p[i + 1][j + 1] + (pizza[i].charAt(j) == 'A' ? 1 : 0);
        }
    }
    return dfs(m, n, k - 1, 0, 0, p);
}

private int dfs(int m, int n, int cuts, int i, int j, int[][] p) {
    // if the remain piece has no apple
    if (p[i][j] == 0) {
        return 0;
    }

    // found valid way after using k - 1 cuts
    if (cuts == 0) {
        return 1;
    }

    if (memo[cuts][i][j] != null) {
        return memo[cuts][i][j];
    }

    int count = 0;
    // cuts in horizontal
    for (int r = i + 1; r < m; r++)  {
        // cuts if the upper piece contains at least one apple
        if (p[i][j] - p[r][j] > 0) {
            count = (count + dfs(m, n, cuts - 1, r, j, p)) % MOD;
        }
    }

    // cuts in vertical
    for (int c = j + 1; c < n; c++) {
        // cuts if the left piece contains at least one apple
        if (p[i][j] - p[i][c] > 0) {
            count = (count + dfs(m, n, cuts - 1, i, c, p)) % MOD;
        }
    }
    return memo[cuts][i][j] = count;
}
{% endhighlight %}

# Discrete

[Maximum Fruits Harvested After at Most K Steps][maximum-fruits-harvested-after-at-most-k-steps]

{% highlight java %}
public int maxTotalFruits(int[][] fruits, int startPos, int k) {
    int n = fruits.length, m = Math.max(startPos, fruits[n - 1][0]);

    int[] p = new int[m + 2];
    for (int i = 0, j = 0; i <= m; i++) {
        p[i + 1] = p[i] + (j < n && fruits[j][0] == i ? fruits[j++][1] : 0);
    }

    int max = 0;
    // moves i steps to right, then turns left
    for (int i = 0; i <= Math.min(k, m - startPos); i++) {
        int left = Math.max(0, startPos - Math.max(0, k - 2 * i));
        int right = startPos + i;
        max = Math.max(max, p[right + 1] - p[left]);
    }

    // moves i steps to left, then turns right
    for (int i = 0; i <= Math.min(k, startPos); i++) {
        int right = Math.min(m, startPos + Math.max(0, k - 2 * i));
        int left = startPos - i;
        max = Math.max(max, p[right + 1] - p[left]);
    }
    return max;
}
{% endhighlight %}

[Find Good Days to Rob the Bank][find-good-days-to-rob-the-bank]

{% highlight java %}
public List<Integer> goodDaysToRobBank(int[] security, int time) {
    int n = security.length;
    int[] p = new int[n], s = new int[n];
    for (int i = 1; i < n; i++) {
        p[i] = (security[i - 1] >= security[i]) ? p[i - 1] + 1 : 0;
    }

    for (int i = n - 1; i > 0; i--) {
        s[i - 1] = (security[i - 1] <= security[i]) ? s[i] + 1 : 0;
    }

    List<Integer> list = new ArrayList<>();
    for (int i = time; i < n - time; i++) {
        if (p[i] >= time && s[i] >= time) {
            list.add(i);
        }
    }
    return list;
}
{% endhighlight %}

# Rolling Prefix Sum

[Change Minimum Characters to Satisfy One of Three Conditions][change-minimum-characters-to-satisfy-one-of-three-conditions]

{% highlight java %}
public int minCharacters(String a, String b) {
    int m = a.length(), n = b.length(), min = m + n;
    int[] c1 = new int[26], c2 = new int[26];
    for (char c : a.toCharArray()) {
        c1[c - 'a']++;
    }
    for (char c : b.toCharArray()) {
        c2[c - 'a']++;
    }

    for (int i = 0; i < 26; i++) {
        // Condition 3
        min = Math.min(min, m + n - c1[i] - c2[i]);

        // rolling prefix sum
        if (i > 0) {
            c1[i] += c1[i - 1];
            c2[i] += c2[i - 1];
        }

        // exludes 'z'
        if (i < 25) {
            // Condition 1
            min = Math.min(min, m - c1[i] + c2[i]);
            // Condition 2
            min = Math.min(min, n - c2[i] + c1[i]);
        }
    }
    return min;
}
{% endhighlight %}

# Pre-computation

[Sum Of Special Evenly-Spaced Elements In Array][sum-of-special-evenly-spaced-elements-in-array]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int[] solve(int[] nums, int[][] queries) {
    int n = nums.length;

    // pre-computation (y ^ 2 <= n)
    int sqrtn = (int)Math.sqrt(n) + 1;
    // map[i][j]: when y == i, the answer from j to the end
    int[][] map = new int[sqrtn][n];
    for (int i = 1; i * i <= n; i++) {
        for (int j = n - 1; j >= 0; j--) {
            if (i + j >= n) {
                map[i][j] = nums[j];
            } else {
                // sums backwards
                map[i][j] = (map[i][i + j] + nums[j]) % MOD;
            }
        }
    }

    int m = queries.length;
    int[] answer = new int[m];
    for (int i = 0; i < m; i++) {
        int x = queries[i][0], y = queries[i][1];
        if ((long)y * (long)y <= n) {
            answer[i] = map[y][x];
        } else {
            // y is large enough, no pre-computation
            for (int j = x; j < n; j += y) {
                answer[i] = (answer[i] + nums[j]) % MOD;
            }
        }
    }
    return answer;
}
{% endhighlight %}

# Dynamic Programming

[Maximum Number of Non-Overlapping Subarrays With Sum Equals Target][maximum-number-of-non-overlapping-subarrays-with-sum-equals-target]

{% highlight java %}
public int maxNonOverlapping(int[] nums, int target) {
    // prefix sum : max number of non-empty non-overlapping subarrays
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, 0);

    int sum = 0, count = 0;
    for (int i = 0; i < nums.length; i++) {
        sum += nums[i];
        if (map.containsKey(sum - target)) {
            count = Math.max(count, map.get(sum - target) + 1);
        }

        // later sum can always overwrite, because `count` is guaranteed >=
        map.put(sum, count);
    }
    return count;
}
{% endhighlight %}

[Count Subarrays With More Ones Than Zeros][count-subarrays-with-more-ones-than-zeros]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int subarraysWithMoreZerosThanOnes(int[] nums) {
    int n = nums.length;
    // dp[i]: count of subarrays ending at i
    int[] dp = new int[n];

    // {prefix sum : last index}
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, -1);

    // like sin(x)
    int sum = 0, count = 0;
    for (int i = 0; i < n; i++) {
        // prefix sum of #1 - #0
        sum += nums[i] == 0 ? -1 : 1;
        if (map.containsKey(sum)) {
            // in (prev, i], #0 == #1
            int prev = map.get(sum);
            dp[i] = prev < 0 ? 0 : dp[prev];

            // "valley" in (prev, i]: first 0's, then 1's
            // so for all subarrays ending at i, i.e. nums[j...i] where prev + 1 < j <= i
            // #1 > #0
            if (nums[i] == 1) {
                dp[i] += i - prev - 1;
            }
        } else if (sum > 0) {
            // "valley" in [0, i]
            // for all the subarrays ending at i, #1 > #0
            dp[i] = i + 1;
        }
        count = (count + dp[i]) % MOD;
        map.put(sum, i);
    }
    return count;
}
{% endhighlight %}

Another solution is to use two DP variables to track states:

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int subarraysWithMoreZerosThanOnes(int[] nums) {
    // {prefix sum, count of indices with this prefix sum}
    Map<Integer, Integer> map = new HashMap<>();
    // no elements means #0 == #1 == 0
    map.put(0, 1);

    // dp0: #0 == #1
    // dp1: #1 > #0
    int dp0 = 0, dp1 = 0, sum = 0, count = 0;
    for (int num : nums) {
        // dp values at index "i - 1" (prev)
        int pdp0 = dp0, pdp1 = dp1;
        // prefix sum of #1 - #0
        sum += num == 0 ? -1 : 1;

        dp0 = map.getOrDefault(sum, 0);
        map.put(sum, dp0 + 1);

        if (num == 0) {
            // pdp0 doesn't contribute to dp1
            // because num == 0 on top of pdp0 will result in #0 > #1
            // adjusts the equation to help understanding:
            //   dp0 + dp1 == pdp1
            dp1 = (pdp1 - dp0 + MOD) % MOD;
        } else {
            dp1 = (pdp0 + pdp1 + 1) % MOD;
        }
        count = (count + dp1) % MOD;
    }
    return count;
}
{% endhighlight %}

[Number of Ways to Separate Numbers][number-of-ways-to-separate-numbers]

{% highlight java %}
private static final int MOD = (int)1e9 + 7;

public int numberOfCombinations(String num) {
    int n = num.length();
    if (num.charAt(0) == '0') {
        return 0;
    }

    // lcp[i][j]: length of the long common prefix of s.substring(i) and s.substring(j)
    int[][] lcp = new int[n + 1][n + 1];
    for (int i = n - 1; i >= 0 ; i--) {
        for (int j = n - 1; j >= 0; j--) {
            if (num.charAt(i) == num.charAt(j)) {
                lcp[i][j] = lcp[i + 1][j + 1] + 1;
            }
        }
    }

    // dp[i][j]: number of ways when the last number is num[i, j]
    // len = j - i + 1, length of the last number
    // - Case 1:
    //   second last number has less digits than last number
    //   dp[i][j] += dp[i - len + 1][i - 1] + ... + dp[i - 1][i - 1]
    // - Case 2:
    //   second last number has the same number of digits as last number
    //   if num[i - length : i - 1] <= num[i : j],
    //   dp[i][j] += dp[i - length][i - 1]

    // prefix sum:
    // p[i + 1][j] = dp[0][j] + dp[1][j] ... + dp[i][j]
    long[][] p = new long[n + 1][n];

    // p[1][j] = dp[0][j] = 1
    Arrays.fill(p[1], 1);

    // last number is num[i:j]
    for (int i = 1; i < n; i++) {
        // no leading zero
        if (num.charAt(i) == '0') {
            p[i + 1] = p[i];
            continue;
        }

        for (int j = i; j < n; j++) {
            int len = j - i + 1, prevStart = i - len;

            // number of ways introduced by the current digit
            long count = 0;

            // start of second last number must be >= 0
            if (prevStart < 0) {
                // Case 1 only:
                // dp[0][i - 1] + dp[1][i - 1] + ... + dp[i - 1][i - 1]
                // == p[i][i - 1]
                count += p[i][i - 1];
            } else {
                // Case 1:
                // dp[prevStart][i - 1] + ... + dp[i - 1][i - 1]
                count += (p[i][i - 1] - p[prevStart + 1][i - 1] + MOD) % MOD;

                // Case 2:
                if (lcp[prevStart][i] >= len ||  // second last number == last number
                    // second last number < last number
                    num.charAt(prevStart + lcp[prevStart][i]) - num.charAt(i + lcp[prevStart][i]) < 0) {
                    // dp[i - length][i - 1]
                    long tmp = (p[prevStart + 1][i - 1] - p[prevStart][i - 1] + MOD) % MOD;
                    count = (count + tmp + MOD) % MOD;
                }
            }
            p[i + 1][j] = (p[i][j] + count) % MOD;
        }
    }
    return (int)p[n][n - 1];
}
{% endhighlight %}

# Bounded Sum

[Max Sum of Rectangle No Larger Than K][max-sum-of-rectangle-no-larger-than-k]

{% highlight java %}
public int maxSumSubmatrix(int[][] matrix, int k) {
    // 2D Kadane's algorithm
    int m = matrix.length, n = matrix[0].length;
    int max = Integer.MIN_VALUE;

    // finds the rectangle with maxSum <= k in O(logN) time
    TreeSet<Integer> set = new TreeSet<>();

    // assumes the number of rows is larger than columns
    for (int r1 = 0; r1 < n; r1++) {
        // accumulate sums for rows in [r1, r2]
        int[] nums = new int[m];
        for (int r2 = r1; r2 < n; r2++) {
            for (int i = 0; i < m; i++) {
                nums[i] += matrix[i][r2];
            }

            // stores prefix sums
            set.clear();
            set.add(0);  // p[0]

            int prefixSum = 0;
            for (int num : nums) {
                prefixSum += num;
                // finds the max targetSum which satifies
                // sum == prefixSum - targetSum <= k
                // i.e. targetSum >= prefixSum - k
                Integer targetSum = set.ceiling(prefixSum - k);
                if (targetSum != null) {
                    max = Math.max(max, prefixSum - targetSum);
                }
                set.add(prefixSum);
            }
        }
    }

    return max;
}
{% endhighlight %}

# Prefix + Suffix Sum

This can be extended to a very useful technique: for each element `arr[i]` in an array, compute its `left[i]` and `right[i]` for a particular variable (e.g. sum, min, max, etc.). Eventually we get two auxiliary arrays `left` and `right`.

[Maximum Number of Ways to Partition an Array][maximum-number-of-ways-to-partition-an-array]

{% highlight java %}
public int waysToPartition(int[] nums, int k) {
    int n = nums.length;
    long sum = Arrays.stream(nums).	asLongStream().sum();

    // sum of left part minus sum of right part
    // diff[i] = (nums[0] + ... + nums[i - 1]) - (nums[i] + ... + nums[n - 1])
    // where 1 <= i < n

    // prefix and suffix
    // frequency map of diff[1..i] and diff[(i + 1)..(n - 1)] respectively
    Map<Long, Integer> p = new HashMap<>(), s = new HashMap<>();
    // running sum
    long left = 0, right = 0;
    for (int i = 0; i < n - 1; i++) {
        left += nums[i];
        right = sum - left;
        s.compute(left - right, (key, v) -> v == null ? 1 : v + 1);
    }

    // no replacement
    int ways = s.getOrDefault(0l, 0);

    // if we replace nums[i] with k,
    // then diff[1...i] decrease by d, and diff[(i + 1)...(n - 1)] increase by d
    // where d = k - nums[i]
    // the ways is the number of 0s in this new diff array.
    left = 0;
    for (int i = 0; i < n; i++) {
        left += nums[i];
        right = sum - left;
        long d = k - nums[i];

        // replaces nums[i] with k
        // we don't actually modify the diff arrays

        // diff[1...i] decrease by d, which means we need to find p[d] before the replacement
        // so that after the replacement, these diff elements with value d will become 0
        // similarly, we need to find s[-d]
        ways = Math.max(ways, p.getOrDefault(d, 0) + s.getOrDefault(-d, 0));

        // transfers the frequency from suffix map to prefix map
        s.compute(left - right, (key, v) -> v == null ? 1 : v - 1);
        p.compute(left - right, (key, v) -> v == null ? 1 : v + 1);
    }
    return ways;
}
{% endhighlight %}

[Super Washing Machines][super-washing-machines]

{% highlight java %}
public int findMinMoves(int[] machines) {
    int n = machines.length;
    int sum = Arrays.stream(machines).sum();
    if (sum % n != 0) {
        return -1;
    }

    // prefix and suffix sum
    int[] p = new int[n], s = new int[n];
    for (int i = 1; i < n; i++) {
        p[i] = p[i - 1] + machines[i - 1];
    }
    for (int i = n - 2; i >= 0; i--) {
        s[i] = s[i + 1] + machines[i + 1];
    }

    // minimum moves is the maximum dresses that pass through for each single machine
    int move = 0, avg = sum / n, expLeft = 0, expRight = sum - avg;
    for (int i = 0; i < n; i++) {
        move = Math.max(move, Math.max(expLeft - p[i], 0) + Math.max(expRight - s[i], 0));
        expLeft += avg;
        expRight -= avg;
    }
    return move;
}
{% endhighlight %}

Optimization:

{% highlight java %}
public int findMinMoves(int[] machines) {
    int n = machines.length;
    int sum = Arrays.stream(machines).sum();
    if (sum % n != 0) {
        return -1;
    }

    // minimum moves is the maximum dresses that pass through for each single machine
    int target = sum / n, move = 0, toRight = 0;
    for (int m : machines) {
        // for each machines, toRight = right - left
        // if toRight > 0, left -> right
        // if toRight < 0, right -> left
        toRight += m - target;
        move = Math.max(move, Math.max(Math.abs(toRight), m - target));
    }
    return move;
}
{% endhighlight %}

# Expand Around Center

[Count Subarrays With Median K][count-subarrays-with-median-k]

{% highlight java %}
public int countSubarrays(int[] nums, int k) {
    int n = nums.length, kIndex = 0;
    for (int i = 0; i < n; i++) {
        if (nums[i] == k) {
            kIndex = i;
        }
    }

    int sum = 0;
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = kIndex; i >= 0; i--) {
        sum += (int)Math.signum(nums[i] - k);
        map.put(sum, map.getOrDefault(sum, 0) + 1);
    }

    sum = 0;
    int count = 0;
    for (int i = kIndex; i < n; i++) {
        sum += (int)Math.signum(nums[i] - k);
        count += map.getOrDefault(-sum, 0) + map.getOrDefault(-sum + 1, 0);
    }
    return count;
}
{% endhighlight %}

[can-make-palindrome-from-substring]: https://leetcode.com/problems/can-make-palindrome-from-substring/
[change-minimum-characters-to-satisfy-one-of-three-conditions]: https://leetcode.com/problems/change-minimum-characters-to-satisfy-one-of-three-conditions/
[contiguous-array]: https://leetcode.com/problems/contiguous-array/
[count-palindromic-subsequences]: https://leetcode.com/problems/count-palindromic-subsequences/
[count-subarrays-with-median-k]: https://leetcode.com/problems/count-subarrays-with-median-k/
[count-subarrays-with-more-ones-than-zeros]: https://leetcode.com/problems/count-subarrays-with-more-ones-than-zeros/
[count-triplets-that-can-form-two-arrays-of-equal-xor]: https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/
[find-good-days-to-rob-the-bank]: https://leetcode.com/problems/find-good-days-to-rob-the-bank/
[longest-well-performing-interval]: https://leetcode.com/problems/longest-well-performing-interval/
[make-sum-divisible-by-p]: https://leetcode.com/problems/make-sum-divisible-by-p/
[max-sum-of-rectangle-no-larger-than-k]: https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/
[maximize-the-beauty-of-the-garden]: https://leetcode.com/problems/maximize-the-beauty-of-the-garden/
[maximum-absolute-sum-of-any-subarray]: https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/
[maximum-fruits-harvested-after-at-most-k-steps]: https://leetcode.com/problems/maximum-fruits-harvested-after-at-most-k-steps/
[maximum-number-of-non-overlapping-subarrays-with-sum-equals-target]: https://leetcode.com/problems/maximum-number-of-non-overlapping-subarrays-with-sum-equals-target/
[maximum-number-of-ways-to-partition-an-array]: https://leetcode.com/problems/maximum-number-of-ways-to-partition-an-array/
[maximum-size-subarray-sum-equals-k]: https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/
[minimum-absolute-difference-queries]: https://leetcode.com/problems/minimum-absolute-difference-queries/
[number-of-ways-of-cutting-a-pizza]: https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/
[number-of-ways-to-separate-numbers]: https://leetcode.com/problems/number-of-ways-to-separate-numbers/
[number-of-wonderful-substrings]: https://leetcode.com/problems/number-of-wonderful-substrings/
[product-of-array-except-self]: https://leetcode.com/problems/product-of-array-except-self/
[product-of-the-last-k-numbers]: https://leetcode.com/problems/product-of-the-last-k-numbers/
[remove-zero-sum-consecutive-nodes-from-linked-list]: https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/
[subarray-sum-equals-k]: https://leetcode.com/problems/subarray-sum-equals-k/
[substring-with-largest-variance]: https://leetcode.com/problems/substring-with-largest-variance/
[sum-of-floored-pairs]: https://leetcode.com/problems/sum-of-floored-pairs/
[sum-of-special-evenly-spaced-elements-in-array]: https://leetcode.com/problems/sum-of-special-evenly-spaced-elements-in-array/
[sum-of-total-strength-of-wizards]: https://leetcode.com/problems/sum-of-total-strength-of-wizards/
[sum-of-beauty-of-all-substrings]: https://leetcode.com/problems/sum-of-beauty-of-all-substrings/
[super-washing-machines]: https://leetcode.com/problems/super-washing-machines/
