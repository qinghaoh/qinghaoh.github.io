---
title:  "Prefix Sum"
category: algorithm
---
## Fundamentals

The basic template for prefix sum creation is:

```c++
vector<int> p(n + 1);
for (int i = 0: i < n; i++) {
    p[i + 1] = p[i] + nums[i];
}

// sum[i...j] = p[j + 1] - p[i]
```

Sometimes we can use a *running prefix sum* instead of a prefix array. For example:

[Subarray Sum Equals K][subarray-sum-equals-k]

```c++
int subarraySum(vector<int>& nums, int k) {
    // {prefix sum, count}
    {% raw %}unordered_map<int, int> mp{{0, 1}};  // p[0]{% endraw %}
    int sum = 0, cnt = 0;
    for (int num : nums) {
        sum += num;
        cnt += mp[sum - k];
        mp[sum]++;
    }
    return cnt;
}
```

[Contiguous Array][contiguous-array]

```c++
    // {prefix sum, index of first occurrence}
    {% raw %}unordered_map<int, int> mp{{0, -1}};  // p[0]{% endraw %}
    // count(ones) - count(zeros)
```

[Longest Well-Performing Interval][longest-well-performing-interval]

```c++
int longestWPI(vector<int>& hours) {
    // {prefix sum : index of first occurrence}
    unordered_map<int, int> mp;
    int mx = 0, score = 0;
    for (int i = 0; i < hours.size(); i++) {
        // Finds the longest subarray with positive sum
        score += hours[i] > 8 ? 1 : -1;
        if (score > 0) {
            mx = i + 1;
        } else {
            mp.insert({score, i});
            // When key < 0, the map is monotonically descreasing.
            // So, mp[score - 1] is the farthest from current position
            if (mp.contains(score - 1)) {
                mx = max(mx, i - mp[score - 1]);
            }
        }
    }
    return mx;
}
```

{: .prompt-tip }
> The prefix sum technique is often used to identify subarrays where the count of elements either less than or greater than a given number num predominates.

[Count Subarrays With Median K][count-subarrays-with-median-k]

```c++
int countSubarrays(vector<int>& nums, int k) {
    int kIndex = ranges::find(nums, k) - nums.begin(), sum = 0;
    function<int(int)> signum = [](int a) {
        return (a > 0) - (a < 0);
    };

    // {prefix sum, count}
    unordered_map<int, int> mp;
    for (int i = kIndex, sum = 0; i >= 0; i--) {
        sum += signum(nums[i] - k);
        mp[sum]++;
    }

    int cnt = 0;
    for (int i = kIndex, sum = 0; i < nums.size(); i++) {
        sum += signum(nums[i] - k);
        // Odd-size and even-size
        cnt += mp[-sum] + mp[-sum + 1];
    }
    return cnt;
}
```

[Remove Zero Sum Consecutive Nodes from Linked List][remove-zero-sum-consecutive-nodes-from-linked-list]

```c++
ListNode* removeZeroSumSublists(ListNode* head) {
    ListNode* dummy = new ListNode(0, head);

    // {prefix sum, last node with this value}
    {% raw %}unordered_map<int, ListNode*> mp{{0, dummy}};{% endraw %}

    int sum = 0;
    for (ListNode* itr = dummy; itr; itr = itr->next) {
        mp[sum += itr->val] = itr;
    }

    // Pass two: links to the last node that has the same prefix sum.
    sum = 0;
    for (ListNode* itr = dummy; itr; itr = itr->next) {
        itr->next = mp[sum += itr->val]->next;
    }
    return dummy->next;
}
```

{: .prompt-info }
> To solve the problem with a single pass, we can use the running sum method. However, this requires removing map entries for nodes located between the current node and the most recent node having the same prefix sum.

[Maximum Number of Non-Overlapping Subarrays With Sum Equals Target][maximum-number-of-non-overlapping-subarrays-with-sum-equals-target]

```c++
int maxNonOverlapping(vector<int>& nums, int target) {
    // {prefix sum, max number of non-empty non-overlapping subarrays with sum equals the prefix sum}
    {% raw %}unordered_map<int, int> mp{{0, 0}};{% endraw %}
    int sum = 0, cnt = 0;
    for (const auto& num : nums) {
        sum += num;
        if (mp.contains(sum - target)) {
            cnt = max(cnt, mp[sum - target] + 1);
        }

        // If the prefix sums at two indices are equal,
        // `cnt` at the second index is always no less than that at the first index.
        mp[sum] = cnt;
    }
    return cnt;
}
```

[Maximize the Beauty of the Garden][maximize-the-beauty-of-the-garden]

```java
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
```

## Generalizing Prefix Operations: Beyond Sum

* Product

[Product of Array Except Self][product-of-array-except-self]

* Mod

[Make Sum Divisible by P][make-sum-divisible-by-p]

```c++
int minSubarray(vector<int>& nums, int p) {
    // Target remainder
    int r = accumulate(nums.begin(), nums.end(), 0, [&](int a, int b) { return (a + b) % p; });

    // Indices of the last occurrence of prefix_sum % p
    {% raw %}unordered_map<int, int> lastIndices = {{0, -1}};{% endraw %}
    int n = nums.size(), mn = n;
    for (int i = 0, curr = 0; i < n; i++) {
        curr = (curr + nums[i]) % p;
        lastIndices[curr] = i;

        // Finds the last index of the remainder that needs to be subtracted.
        int d = (curr - r + p) % p;
        if (lastIndices.contains(d)) {
            mn = min(mn, i - lastIndices[d]);
        }
    }
    return mn == n ? -1 : mn;
}
```

* Exclusive Or

[Count Triplets That Can Form Two Arrays of Equal XOR][count-triplets-that-can-form-two-arrays-of-equal-xor]

[Can Make Palindrome from Substring][can-make-palindrome-from-substring]

```c++
vector<bool> canMakePaliQueries(string s, vector<vector<int>>& queries) {
    int n = s.length();
    // 26 bits to represent prefix xor
    vector<int> p(n + 1);
    for (int i = 0; i < n; i++) {
        p[i + 1] = p[i] ^ (1 << (s[i] - 'a'));
    }

    vector<bool> answer;
    for (const auto& q : queries) {
        answer.push_back(popcount(static_cast<unsigned int>(p[q[1] + 1] ^ p[q[0]])) <= 2 * q[2] + 1);
    }
    return answer;
}
```

[Number of Wonderful Substrings][number-of-wonderful-substrings]

```c++
long long wonderfulSubstrings(string word) {
    const int NUM_CHARS = 10;
    // Freqs map of bit mask
    vector<long long> freqs(1 << NUM_CHARS);
    freqs[0] = 1;

    // Bits to represent prefix xor
    long long cnt = 0;
    int p = 0;
    for (int i = 0; i < word.length(); i++) {
        p ^= 1 << (word[i] - 'a');
        cnt += freqs[p];
        for (int j = 0; j < NUM_CHARS; j++) {
            cnt += freqs[p ^ (1 << j)];
        }
        freqs[p]++;
    }
    return cnt;
}
```

* Multi-dimension

[Sum of Beauty of All Substrings][sum-of-beauty-of-all-substrings]

```java
int[][] p = new int[26][n + 1];
for (int i = 0; i < n; i++) {
    for (int k = 0; k < p.length; k++) {
        p[k][i + 1] = p[k][i] + (k == s.charAt(i) - 'a' ? 1 : 0);
    }
}
```

[Palindrome Rearrangement Queries][palindrome-rearrangement-queries]

```c++
vector<bool> canMakePalindromeQueries(string s, vector<vector<int>>& queries) {
    int n = s.size();
    // Prefix sum of number of different chars at symmetric positions
    vector<int> p1(n + 1);
    for (int i = 0; i < n - 1 - i; i++) {
	p1[i + 1] = p1[i] + (s[i] != s[n - 1 - i]);
    }

    valarray<int> freqs(26);
    vector<valarray<int>> p2 = {freqs};
    for (char ch : s) {
	freqs[ch - 'a']++;
	p2.push_back(freqs);
    }

    vector<bool> answer;
    for (const auto& q : queries) {
	//  --a1----b1---b2----a2--
	//  -d2-c2-----------c1-d1-
	// x1 and x2 are symmetric.
	int a1 = q[0], b1 = q[1] + 1, a2 = n - q[0], b2 = n - 1 - q[1];
	int c1 = q[2], d1 = q[3] + 1, c2 = n - q[2], d2 = n - 1 - q[3];

	// No differences allowed outside the query ranges.
	// e.g. R1, R2 and R3:
	//   -----a1 b1---d2 c2---
	//   | R1 |    |R2|   |R3|
	if ((min(a1, d2) && p1[min(a1, d2)])
	    || (n / 2 > max(b1, c2) && p1[n / 2] - p1[max(b1, c2)])
	    || (d2 > b1 && p1[d2] - p1[b1])
	    || (a1 > c2 && p1[a1] - p1[c2])) {
	    answer.push_back(false);
	} else {
	    valarray<int> f1 = p2[b1] - p2[a1], f2 = p2[d1] - p2[c1];
	    // Trims to the overlapping ranges.
	    if (c1 > b2) {
		f1 -= p2[min(c1, a2)] - p2[b2];
	    }
	    if (a2 > d1) {
		f1 -= p2[a2] - p2[max(d1, b2)];
	    }
	    if (a1 > d2) {
		f2 -= p2[min(a1, c2)] - p2[d2];
	    }
	    if (c2 > b1) {
		f2 -= p2[c2] - p2[max(b1, d2)];
	    }
	    // Frequency map of the overlapping ranges on both sides should be identical.
	    answer.push_back((f1 >= 0 && f2 >= 0 && f1 == f2).min());
	}
    }
    return answer;
}
```

[Count Increasing Quadruplets][count-increasing-quadruplets]

```java
public long countQuadruplets(int[] nums) {
    int n = nums.length;
    // p[i][j]: number of elements < i in the first j elements
    // it won't exceed the array boundary since nums[i] < n
    int[][] p = new int[n + 1][n + 1];
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= n; i++) {
            p[i][j + 1] = p[i][j] + (nums[j] < i ? 1 : 0);
        }
    }

    long count = 0;
    for (int j = 0; j < n; j++) {
        for (int k = j + 1; k < n; k++) {
            if (nums[j] > nums[k]) {
                // p[nums[k]][j + 1]: number of "i"s that i < j < k and nums[i] < nums[k] < nums[j]
                // k - p[nums[j]][k + 1]: number of elements >= nums[j] in the first k elements
                //   since all integers of nums are unique, >= nums[j] can be simplified as > nums[j]
                // n - nums[j]: number of elements > nums[j], since nums is a permutation
                // n - nums[j] - (k - p[nums[j]][k + 1]): number of "l"s that j < k < l and nums[k] < nums[j] < nums[l]
                count += p[nums[k]][j + 1] * (n - nums[j] - (k - p[nums[j]][k + 1]));
            }
        }
    }
    return count;
}
```

Another solution is by [dynamic programming](../dynamic-programming-vi).
 
* Frequency

[Sum of Floored Pairs][sum-of-floored-pairs]

```c++
int sumOfFlooredPairs(vector<int>& nums) {
    const int mod = 1e9 + 7;
    int mx = *ranges::max_element(nums);

    vector<int> freqs(mx + 1);
    for (int num : nums) {
        freqs[num]++;
    }

    for (int i = 0; i < mx; i++) {
        freqs[i + 1] += freqs[i];
    }

    unordered_map<int, int> mp;
    int cnt = 0;
    for (int num : nums) {
        if (mp.contains(num)) {
            cnt = (cnt + mp[num]) % mod;
            continue;
        }

        // If floor(nums[i] / nums[j]) = k, then k * nums[j] <= nums[i] < (k + 1) * nums[j].
        // Lets low = k * nums[j], high = (k + 1) * nums[j].
        // For all elements in the range [low, high], floor(num / nums[j]) == k.
        //
        // Counts all the pairs with `num` as denominator.
        // In each iteration, counts the pairs with floor() == k
        // When k increments, increases low and high by `num` to update the boundaries.
        int curr = 0, low = num, high = min(2 * num - 1, mx);
        long long k = 1;
        while (low <= mx) {
            curr = (curr + ((freqs[high] - freqs[low - 1]) * k) % mod) % mod;
            low += num;
            high = min(high + num, mx);
            k++;
        }
        cnt = (cnt + (mp[num] = curr)) % mod;
    }
    return cnt;
}
```

* Linked List

[Remove Zero Sum Consecutive Nodes from Linked List][remove-zero-sum-consecutive-nodes-from-linked-list]

* Diff

[Substring With Largest Variance][substring-with-largest-variance]

```java
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
```

This problem can also be solved by [Kadane's Algorithm](../kadanes).

* Deviation

[Super Washing Machines][super-washing-machines]

```c++
int findMinMoves(vector<int>& machines) {
    int n = machines.size();
    int sum = accumulate(machines.begin(), machines.end(), 0);
    if (sum % n) {
        return -1;
    }

    // The algorithm abstracts away the specifics of dress transfers, focusing instead on the net effect.
    int moves = 0, throughput = 0, avg = sum / n;
    for (const int& machine : machines) {
        // machine - avg: the net dresses to be transferred from this machine to achieve balance.
        throughput += machine - avg;

        // throughput: cumulative net dresses transferred: positive for outgoing, negative for incoming.
        // e.g. if the "machine - avg" array is [-2, -3, 5], then the throughput at machines[1] is -5.
        // Only if it gets 5 dresses can machines[0] and machines[1] have the same number of dresses in the end.
        // So, the abs of throughput is one determining factor of the final result.
        //
        // At each move, a machine can give out at most 1 dress, but can get at most 2 dresses.
        // So, the number of given out dresses is the bottleneck and we don't need abs on (machine - avg)
        // e.g., the "machine - avg" array [-1, 2, -1] requires 2 moves, while [1, -2, 1] requires only one move
        moves = max(max(moves, abs(throughput)), machine - avg);
    }
    return moves;
}
```

* Prefix Sum

[Sum of Total Strength of Wizards][sum-of-total-strength-of-wizards]

```java
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
```

[Maximum Absolute Sum of Any Subarray][maximum-absolute-sum-of-any-subarray]

```
abs(nums[i...j)) = abs(p[j] - p[i]) = max(p[i], p[j]) - min(p[i], p[j]), where 0 <= i < j <= n

max(abs(nums[i...j])) = max(max(p[i], p[j]) - min(p[i], p[j])) = max(p) - min(p)
```

## Range Sum

[Number of Ways of Cutting a Pizza][number-of-ways-of-cutting-a-pizza]

```java
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
    // If the remaining piece has no apple
    if (p[i][j] == 0) {
        return 0;
    }

    // Found valid way after using k - 1 cuts
    if (cuts == 0) {
        return 1;
    }

    if (memo[cuts][i][j] != null) {
        return memo[cuts][i][j];
    }

    int count = 0;
    // Cuts in horizontal
    for (int r = i + 1; r < m; r++)  {
        // Cuts if the upper piece contains at least one apple
        if (p[i][j] - p[r][j] > 0) {
            count = (count + dfs(m, n, cuts - 1, r, j, p)) % MOD;
        }
    }

    // cuts in vertical
    for (int c = j + 1; c < n; c++) {
        // Cuts if the left piece contains at least one apple
        if (p[i][j] - p[i][c] > 0) {
            count = (count + dfs(m, n, cuts - 1, i, c, p)) % MOD;
        }
    }
    return memo[cuts][i][j] = count;
}
```

Running sum:

[Count Submatrices with Top-Left Element and Sum Less Than k][count-submatrices-with-top-left-element-and-sum-less-than-k]

```c++
int countSubmatrices(vector<vector<int>>& grid, int k) {
    int ans = 0;
    for (int i = 0; i < grid.size(); i++) {
        for (int j = 0; j < grid[0].size(); j++) {
            if (i > 0) {
                grid[i][j] += grid[i - 1][j];
            }
            if (j > 0) {
                grid[i][j] += grid[i][j - 1];
            }
            if (i > 0 && j > 0) {
                grid[i][j] -= grid[i - 1][j - 1];
            }

            if (grid[i][j] > k) {
                break;
            }
            ans++;
        }
    }
    return ans;
}
```

## Discrete Prefix Sum

[Maximum Fruits Harvested After at Most K Steps][maximum-fruits-harvested-after-at-most-k-steps]

```c++
int maxTotalFruits(vector<vector<int>>& fruits, int startPos, int k) {
    int n = fruits.size(), end = max(startPos, fruits[n - 1][0]);
    vector<int> p(end + 2);
    for (int i = 0, j = 0; i <= end; i++) {
        p[i + 1] = p[i] + (j < n && fruits[j][0] == i ? fruits[j++][1] : 0);
    }

    int mx = 0;
    // Moves i steps to right, then turns left
    for (int i = 0; i <= min(k, end - startPos); i++) {
        int left = max(0, startPos - max(0, k - 2 * i));
        int right = startPos + i;
        mx = max(mx, p[right + 1] - p[left]);
    }

    // Moves i steps to left, then turns right
    for (int i = 0; i <= min(k, startPos); i++) {
        int right = min(end, startPos + max(0, k - 2 * i));
        int left = startPos - i;
        mx = max(mx, p[right + 1] - p[left]);
    }
    return mx;
}
```

## Rolling Prefix Sum

[Change Minimum Characters to Satisfy One of Three Conditions][change-minimum-characters-to-satisfy-one-of-three-conditions]

```java
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
```

## Pre-computation

[Sum Of Special Evenly-Spaced Elements In Array][sum-of-special-evenly-spaced-elements-in-array]

```java
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
```

## Dynamic Programming

[Count Subarrays With More Ones Than Zeros][count-subarrays-with-more-ones-than-zeros]

```java
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
```

Another solution is to use two DP variables to track states:

```java
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
```

[Number of Ways to Separate Numbers][number-of-ways-to-separate-numbers]

```java
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
```

## Bounded Sum

[Max Sum of Rectangle No Larger Than K][max-sum-of-rectangle-no-larger-than-k]

```java
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
```

## Prefix + Suffix Sum

This can be extended to a very useful technique: for each element `arr[i]` in an array, compute its `left[i]` and `right[i]` for a particular variable (e.g. sum, min, max, etc.). Eventually we get two auxiliary arrays `left` and `right`.

[Find Good Days to Rob the Bank][find-good-days-to-rob-the-bank]

```c++
vector<int> goodDaysToRobBank(vector<int>& security, int time) {
    int n = security.size();
    vector<int> p(n), s(n);
    for (int i = 1; i < n; i++) {
        p[i] = (security[i] <= security[i - 1]) ? p[i - 1] + 1 : 0;
    }

    for (int i = n - 2; i >= 0; i--) {
        s[i] = (security[i] <= security[i + 1]) ? s[i + 1] + 1 : 0;
    }

    vector<int> res;
    for (int i = time; i < n - time; i++) {
        if (p[i] >= time && s[i] >= time) {
            res.push_back(i);
        }
    }
    return res;
}
```

{: .prompt-tip }
> The array `p` is sized `n` instead of the usual `n + 1`. This is because `security[0]` does not have a preceding element (`security[-1]` doesn't exist). As a result, we ignore the corresponding value for `security[0]` in `p`, hence the reduced size.

[Maximum Number of Ways to Partition an Array][maximum-number-of-ways-to-partition-an-array]

```java
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
```

[can-make-palindrome-from-substring]: https://leetcode.com/problems/can-make-palindrome-from-substring/
[change-minimum-characters-to-satisfy-one-of-three-conditions]: https://leetcode.com/problems/change-minimum-characters-to-satisfy-one-of-three-conditions/
[contiguous-array]: https://leetcode.com/problems/contiguous-array/
[count-increasing-quadruplets]: https://leetcode.com/problems/count-increasing-quadruplets/
[count-subarrays-with-median-k]: https://leetcode.com/problems/count-subarrays-with-median-k/
[count-subarrays-with-more-ones-than-zeros]: https://leetcode.com/problems/count-subarrays-with-more-ones-than-zeros/
[count-submatrices-with-top-left-element-and-sum-less-than-k]: https://leetcode.com/problems/count-submatrices-with-top-left-element-and-sum-less-than-k/
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
[number-of-ways-of-cutting-a-pizza]: https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/
[number-of-ways-to-separate-numbers]: https://leetcode.com/problems/number-of-ways-to-separate-numbers/
[number-of-wonderful-substrings]: https://leetcode.com/problems/number-of-wonderful-substrings/
[palindrome-rearrangement-queries]: https://leetcode.com/problems/palindrome-rearrangement-queries/
[product-of-array-except-self]: https://leetcode.com/problems/product-of-array-except-self/
[remove-zero-sum-consecutive-nodes-from-linked-list]: https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/
[subarray-sum-equals-k]: https://leetcode.com/problems/subarray-sum-equals-k/
[substring-with-largest-variance]: https://leetcode.com/problems/substring-with-largest-variance/
[sum-of-floored-pairs]: https://leetcode.com/problems/sum-of-floored-pairs/
[sum-of-special-evenly-spaced-elements-in-array]: https://leetcode.com/problems/sum-of-special-evenly-spaced-elements-in-array/
[sum-of-total-strength-of-wizards]: https://leetcode.com/problems/sum-of-total-strength-of-wizards/
[sum-of-beauty-of-all-substrings]: https://leetcode.com/problems/sum-of-beauty-of-all-substrings/
[super-washing-machines]: https://leetcode.com/problems/super-washing-machines/
