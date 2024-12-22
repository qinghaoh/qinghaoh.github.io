---
title:  "Backtracking"
category: algorithm
tags: [dfs, backtracking]
---
# Fundamentals

Backtracking = DFS + pruning

```java
private void backtrack(var i) {
    for (var i : space) {
        backtrack();
    }
}
```

[Permutations][permutations]

```java
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> list = new ArrayList<>();
    backtrack(list, new ArrayList<>(), nums);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tmpList, int[] nums) {
    if (tmpList.size() == nums.length) {
        list.add(new ArrayList<>(tmpList));
        return;
    }

    // increment
    for (int i = 0; i < nums.length; i++) { 
        // the search space of each layer excludes already visited elements
        if (!tmpList.contains(nums[i])) {
            tmpList.add(nums[i]);
            backtrack(list, tmpList, nums);
            tmpList.remove(tmpList.size() - 1);
        }
    }
}
```

```java
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> list = new ArrayList<>();
    backtrack(list, Arrays.stream(nums).boxed().collect(Collectors.toList()), 0);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tmpList, int index) {
    if (index == tmpList.size()) {
        list.add(new ArrayList<>(tmpList));
        return;
    }

    // swap
    for (int i = index; i < tmpList.size(); i++) {
        Collections.swap(tmpList, i, index);
        backtrack(list, tmpList, index + 1);
        Collections.swap(tmpList, index, i);
    }
}
```

[Permutations II][permutations-ii]

```java
public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(nums);
    backtrack(list, new ArrayList<>(), nums, new boolean[nums.length]);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tmpList, int[] nums, boolean[] used) {
    if (tmpList.size() == nums.length) {
        list.add(new ArrayList<>(tmpList));
        return;
    }

    // increment
    for (int i = 0; i < nums.length; i++) {
        // the search space of each layer includes only the first of equal elements
        // e.g. 2, 2, 2
        //      ^
        if (used[i] || i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
            continue;
        }

        used[i] = true;
        tmpList.add(nums[i]);
        backtrack(list, tmpList, nums, used);
        used[i] = false;
        tmpList.remove(tmpList.size() - 1);
    }
}
```

[Palindrome Permutation II][palindrome-permutation-ii]

[Beautiful Arrangement][beautiful-arrangement]

[Subsets][subsets]

```java
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> list = new ArrayList<>();
    backtrack(list, new ArrayList<>(), nums, 0);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tmpList, int[] nums, int index) {
    if (index == nums.length) {
        list.add(new ArrayList<>(tmpList));
        return;
    }

    backtrack(list, tmpList, nums, index + 1);
    tmpList.add(nums[index]);
    backtrack(list, tmpList, nums, index + 1);
    tmpList.remove(tmpList.size() - 1);
}
```

[Subsets II][subsets-ii]

```java
public List<List<Integer>> subsetsWithDup(int[] nums) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(nums);
    backtrack(list, new ArrayList<>(), nums, 0);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tmpList, int[] nums, int index) {
    list.add(new ArrayList<>(tmpList));

    for (int i = index; i < nums.length; i++) {
        if (i > index && nums[i] == nums[i - 1]) {
            continue;
        }

        tmpList.add(nums[i]);
        backtrack(list, tmpList, nums, i + 1);
        tmpList.remove(tmpList.size() - 1);
    }
}
```

[Closest Dessert Cost][closest-dessert-cost]

```java
private int diff = 10001;

public int closestCost(int[] baseCosts, int[] toppingCosts, int target) {
    for (int b : baseCosts) {
        backtrack(toppingCosts, 0, target - b);
    }
    return target - diff;
}

private void backtrack(int[] nums, int index, int t) {
    if ((Math.abs(t) < Math.abs(diff)) || (Math.abs(t) == Math.abs(diff) && t > 0)) {
        diff = t;
    }

    if (index == nums.length || t <= 0) {
        return;
    }

    backtrack(nums, index + 1, t);
    backtrack(nums, index + 1, t - nums[index]);
    backtrack(nums, index + 1, t - 2 * nums[index]);
}
```

[Letter Tile Possibilities][letter-tile-possibilities]

```java
public int numTilePossibilities(String tiles) {
    int[] count = new int[26];
    for (char c : tiles.toCharArray()) {
        count[c - 'A']++;
    }
    return backtrack(count);
}

private int backtrack(int[] count) {
    int sum = 0;
    for (int i = 0; i < 26; i++) {
        if (count[i] == 0) {
            continue;
        }
        sum++;
        count[i]--;
        sum += backtrack(count);
        count[i]++;
    }
    return sum;
}
```

[Maximum Score Words Formed by Letters][maximum-score-words-formed-by-letters]

```java
public int maxScoreWords(String[] words, char[] letters, int[] score) {
    int[] count = new int[score.length];
    for (char c : letters) {
        count[c - 'a']++;
    }
    return backtrack(words, count, score, 0);
}

private int backtrack(String[] words, int[] count, int[] score, int index) {
    int max = 0;
    for (int i = index; i < words.length; i++) {
        int sum = 0;
        boolean isValid = true;
        for (char c : words[i].toCharArray()) {
            if (count[c - 'a']-- == 0) {
                isValid = false;
            }
            sum += score[c - 'a'];
        }

        if (isValid) {
            sum += backtrack(words, count, score, i + 1);
            max = Math.max(sum, max);
        }

        for (char c : words[i].toCharArray()) {
            count[c - 'a']++;
            sum = 0;
        }
    }
    return max;
}
```

[Combination Sum][combination-sum]

```java
public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> list = new ArrayList<>();
    backtrack(list, new ArrayList<>(), candidates, 0, target);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tmpList, int[] nums, int index, int target) {
    if (target == 0) {
        list.add(new ArrayList<>(tmpList));
        return;
    }

    for (int i = index; i < nums.length; i++) {
        if (nums[i] <= target) {
            tmpList.add(nums[i]);
            backtrack(list, tmpList, nums, i, target - nums[i]);
            tmpList.remove(tmpList.size() - 1);
        } 
    }
}
```

[Combination Sum II][combination-sum-ii]

```java
public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(candidates);
    backtrack(list, new ArrayList<>(), candidates, 0, target);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tmpList, int[] nums, int index, int target) {
    if (target == 0) {
        list.add(new ArrayList<>(tmpList));
        return;
    }

    for (int i = index; i < nums.length; i++) {
        if (i > index && nums[i] == nums[i - 1]) {
            continue;
        }

        if (nums[i] <= target) {
            tmpList.add(nums[i]);
            backtrack(list, tmpList, nums, i + 1, target - nums[i]);
            tmpList.remove(tmpList.size() - 1);
        } 
    }
}
```

[Combination Sum III][combination-sum-iii]

```java
private final int max = 9;

public List<List<Integer>> combinationSum3(int k, int n) {
    List<List<Integer>> list = new ArrayList<>();
    backtrack(list, new ArrayList<>(), 1, k, n);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tmpList, int start, int k, int n) {
    if (k == 0 && n == 0) {
        list.add(new ArrayList<>(tmpList));
        return;
    }

    if (k < 0 || n < 0) {
        return;
    }

    for (int i = start; i <= max; i++) {
        if (i <= n) {
            tmpList.add(i);
            backtrack(list, tmpList, i + 1, k - 1, n - i);
            tmpList.remove(Integer.valueOf(i));
        } 
    }
}
```

[Factor Combinations][factor-combinations]

```java
public List<List<Integer>> getFactors(int n) {
    List<List<Integer>> list = new ArrayList<>();
    backtrack(list, new ArrayList<>(), 2, n);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tmpList, int index, int n) {
    if (n == 1) {
        if (tmpList.size() > 1) {
            list.add(new ArrayList<>(tmpList));
        }
        return;
    }

    for (int i = index; i <= n; i++) {
        if (n % i == 0) {
            tmpList.add(i);
            backtrack(list, tmpList, i, n / i);
            tmpList.remove(tmpList.size() - 1);
        }
    }
}
```

[Palindrome Partitioning][palindrome-partitioning]

```java
public List<List<String>> partition(String s) {
    List<List<String>> list = new ArrayList<>();
    backtrack(list, new ArrayList<>(), s, 0);
    return list;
}

private void backtrack(List<List<String>> list, List<String> tmpList, String s, int index) {
    if (index == s.length()) {
        list.add(new ArrayList<>(tmpList));
    }

    for (int i = index + 1; i <= s.length(); i++) {
        String str = s.substring(index, i);
        if (isPalindrome(str)) {
            tmpList.add(str);
            backtrack(list, tmpList, s, i);
            tmpList.remove(tmpList.size() - 1);
        }
    }
}

private boolean isPalindrome(String s) {
    ...
}
```

[Partition Equal Subset Sum][partition-equal-subset-sum]

```java
public boolean canPartition(int[] nums) {
    int sum = 0;
    for (int i = 0; i < nums.length; i++) {
        sum += nums[i];
    }

    if (sum % 2 == 1) {
        return false;
    }

    Arrays.sort(nums);
    return backtrack(nums, 0, sum / 2);
}

private boolean backtrack(int[] nums, int index, int target) {
    if (target == 0) {
        return true;
    }

    for (int i = index; i < nums.length; i++) {
        // skips duplicates
        if (i > index && nums[i] == nums[i - 1]) {
            continue;
        }

        if (nums[i] <= target && backtrack(nums, i + 1, target - nums[i])) {
            return true;
        }
    }
    return false;
}
```

[Construct the Lexicographically Largest Valid Sequence][construct-the-lexicographically-largest-valid-sequence]

```java
public int[] constructDistancedSequence(int n) {
    int[] seq = new int[n * 2 - 1];
    backtrack(seq, 0, new boolean[n]);
    return seq;
}

private boolean backtrack(int[] seq, int index, boolean[] visited) {
    if (index == seq.length) {
        return true;
    }

    // this index is already assigned
    if (seq[index] != 0) {
        return backtrack(seq, index + 1, visited);
    }

    int n = visited.length;
    // starts from n to find the lexicographically largest sequence
    for (int i = n; i > 0; i--) {
        if (!visited[i - 1]) {
            visited[i - 1] = true;
            seq[index] = i;

            if (i == 1) {
                // early termination
                if (backtrack(seq, index + 1, visited)) {
                    return true;
                }
            } else if (index + i < seq.length && seq[index + i] == 0) {
                seq[i + index] = i;
                if (backtrack(seq, index + 1, visited)) {
                    return true;
                }
                seq[index + i] = 0;
            }

            seq[index] = 0;
            visited[i - 1] = false;
        }
    }

    return false;
}
```

[Generalized Abbreviation][generalized-abbreviation]

```java
private String word;

public List<String> generateAbbreviations(String word) {
    this.word = word;
    List<String> list = new ArrayList<>();
    backtrack(list, new StringBuilder(), 0, 0);
    return list;
}

// k is the count of consecutive abbreviated characters
private void backtrack(List<String> list, StringBuilder sb, int index, int k) {
    int length = sb.length();
    if (index == word.length()) {
        // abbreviates the last k letters
        if (k > 0) {
            sb.append(k);
        }
        list.add(sb.toString());
        sb.setLength(length);
        return;
    }

    // the branch that word.charAt(index) is abbreviated
    backtrack(list, sb, index + 1, k + 1);

    // the branch that word.charAt(index) is kept
    // abbreviates the last k letters
    if (k > 0) {
        sb.append(k);
    }
    // appends word.charAt(index)
    sb.append(word.charAt(index));
    backtrack(list, sb, index + 1, 0);
    sb.setLength(length);
}
```

[Minimum Moves to Spread Stones Over Grid][minimum-moves-to-spread-stones-over-grid]

Neither Greedy nor BFS works.

# Subset Sum Problem

[Subset sum problem](https://en.wikipedia.org/wiki/Subset_sum_problem): NP-complete

[Partition to K Equal Sum Subsets][partition-to-k-equal-sum-subsets]

```java
public boolean canPartitionKSubsets(int[] nums, int k) {
    int sum = 0, max = 0;
    for (int num : nums) {
        sum += num;
        max = Math.max(max, num);
    }

    if (sum % k != 0 || max > sum / k) {
        return false;
    }

    Arrays.sort(nums);
    // searches in reverse order, so that subset sizes decrease faster
    return backtrack(nums, sum / k, nums.length - 1, new int[k]);
}

private boolean backtrack(int[] nums, int target, int index, int[] subsets) {
    // all elements are placed into subsets
    if (index < 0) {
        return true;
    }

    for (int i = 0; i < subsets.length; i++) {
        if (subsets[i] + nums[index] <= target) {
            subsets[i] += nums[index];
            // no need to clone subsets
            if (backtrack(nums, target, index - 1, subsets)) {
                return true;
            }
            subsets[i] -= nums[index];
        }

        // after unwinding, if current subset is empty,
        // we know nums[index] can't be placed in any empty subset.
        // all the subsets following current subset are empty,
        // so we skip all of them.
        if (subsets[i] == 0) {
            break;
        }
    }
    return false;
}
```

```java
public boolean canPartitionKSubsets(int[] nums, int k) {
    int sum = 0, max = 0;
    for (int num : nums) {
        sum += num;
        max = Math.max(max, num);
    }

    if (sum % k != 0 || max > sum / k) {
        return false;
    }

    Arrays.sort(nums);
    // searches in reverse order, so that subset sizes decrease faster
    return backtrack(nums, nums.length - 1, new boolean[nums.length], k, 0, sum / k);
}

private boolean backtrack(int[] nums, int index, boolean[] visited, int k, int sum, int target) {
    if (k == 1) {
        return true;
    }

    if (sum == target) {
        return backtrack(nums, nums.length - 1, visited, k - 1, 0, target);
    }

    for (int i = index; i >= 0; i--) {
        if (!visited[i] && sum + nums[i] <= target) {
            visited[i] = true;
            if (backtrack(nums, i - 1, visited, k, sum + nums[i], target)) {
                return true;
            }  
            visited[i] = false;
        }
    }

    return false;
}
```

[Fair Distribution of Cookies][fair-distribution-of-cookies]

```java
public int distributeCookies(int[] cookies, int k) {
    return backtrack(cookies, new int[k], 0);
}

private int backtrack(int[] cookies, int[] children, int index) {
    if (index == cookies.length) {
        int max = 0;
        for (int num : children) {
            max = Math.max(max, num);
        }
        return max;
    }

    int min = Integer.MAX_VALUE;
    for (int i = 0; i < children.length; i++) {
        children[i] += cookies[index];
        min = Math.min(min, backtrack(cookies, children, index + 1));
        children[i] -= cookies[index];
    }
    return min;
}
```

[Matchsticks to Square][matchsticks-to-square]

[Android Unlock Patterns][android-unlock-patterns]

```java
private int[][] skip;
private int m, n;

public int numberOfPatterns(int m, int n) {
    this.skip = new int[10][10];
    skip[1][3] = skip[3][1] = 2;
    skip[1][7] = skip[7][1] = 4;
    skip[3][9] = skip[9][3] = 6;
    skip[7][9] = skip[9][7] = 8;
    skip[1][9] = skip[9][1] = skip[2][8] = skip[8][2] = skip[3][7] = skip[7][3] = skip[4][6] = skip[6][4] = 5;

    this.m = m;
    this.n = n;

    // symmetry
    boolean visited[] = new boolean[10];
    int count = backtrack(1, 1, visited) * 4;
    count += backtrack(2, 1, visited) * 4;
    count += backtrack(5, 1, visited);
    return count;
}

private int backtrack(int num, int level, boolean[] visited) {
    if (level > n) {
        return 0;
    }

    visited[num] = true;
    int count = 0;
    for (int i = 1; i <= 9; i++) {
        if (visited[i] || (skip[num][i] != 0 && !visited[skip[num][i]])) {
            continue;
        }
        count += backtrack(i, level + 1, visited);
    }
    visited[num] = false;

    // accumulation
    if (level >= m) {
        count++;
    }
    return count;
}
```

[Robot Room Cleaner][robot-room-cleaner]

[Wall follower](https://en.wikipedia.org/wiki/Maze-solving_algorithm#Wall_follower): If the maze is simply connected, that is, all its walls are connected together or to the maze's outer boundary, then by keeping one hand in contact with one wall of the maze the solver is guaranteed not to get lost and will reach a different exit if there is one; otherwise, the algorithm will return to the entrance having traversed every corridor next to that connected section of walls at least once. (DFS)

```java
{% raw %}
private static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
{% endraw %}
private Robot robot;
private Set<String> visited = new HashSet<>();

public void cleanRoom(Robot robot) {
    this.robot = robot;
    backtrack(0, 0, 0);
}

private void backtrack(int row, int col, int d) {
    visited.add(row + "#" + col);
    robot.clean();

    // clockwise : 0: 'up', 1: 'right', 2: 'down', 3: 'left'
    for (int i = 0; i < 4; i++) {
        int newD = (d + i) % 4;
        int newRow = row + DIRECTIONS[newD][0];
        int newCol = col + DIRECTIONS[newD][1];

        // considers visited cells as virtual obstacles
        if (!visited.contains(newRow + "#" + newCol) && robot.move()) {
            backtrack(newRow, newCol, newD);
            goBack();
        }

        // turns the robot following chosen direction : clockwise
        robot.turnRight();
    }
}

// goes back facing the same direction
private void goBack() {
    robot.turnRight();
    robot.turnRight();
    robot.move();
    robot.turnRight();
    robot.turnRight();
}
```

[24 Game][24-game]

```java
private static final double TARGET = 24d;
private static final double EPS = 0.001;

public boolean judgePoint24(int[] cards) {
    return backtrack(IntStream.of(cards).mapToDouble(i -> i).toArray(), cards.length);
}

private boolean backtrack(double[] nums, int length) {
    if (length == 1) {
        if (Math.abs(nums[0] - TARGET) < EPS) {
            return true;
        }
    }

    // picks two cards
    for (int i = 0; i < length - 1; i++) {
        for (int j = i + 1; j < length; j++) {
            double c1 = nums[i], c2 = nums[j];

            // puts the new card to the min of i, j
            // and moves the last card to the max of i,j to shrink the array size bby 1
            int index = Math.min(i, j);
            nums[Math.max(i, j)] = nums[length - 1];

            // iterates through all possible combinations as a new card
            for (double c : new double[]{c1 + c2, c1 - c2, c2 - c1, c1 * c2, c1 / c2, c2 / c1}) {
                nums[index] = c;
                if (backtrack(nums, length - 1)) {
                    return true;
                }
            }

            nums[i] = c1;
            nums[j] = c2;
        }
    }

    return false;
}
```

# Parsing

[Expression Add Operators][expression-add-operators]

```java
private int target;

public List<String> addOperators(String num, int target) {
    this.target = target;

    List<String> list = new ArrayList<>();
    backtrack(list, new StringBuilder(), num, 0, 0, 0);
    return list;
}

// 1 + 2 * 3 * 4
// curr = 4
// eval = 7
// product = 6
public void backtrack(List<String> list, StringBuilder sb, String num, int index, long eval, long product) {
    if (index == num.length()) {
        if (target == eval) {
            list.add(sb.toString());
        }
        return;
    }

    for (int i = index; i < num.length(); i++) {
        // skips consecutive 0's
        if (num.charAt(index) == '0' && i != index) {
            break;
        }

        long curr = Long.parseLong(num.substring(index, i + 1));
        int len = sb.length();
        if (index == 0) {
            backtrack(list, sb.append(curr), num, i + 1, curr, curr);
            sb.setLength(len);
        } else {
            backtrack(list, sb.append("+").append(curr), num, i + 1, eval + curr, curr);
            sb.setLength(len);

            backtrack(list, sb.append("-").append(curr), num, i + 1, eval - curr, -curr);
            sb.setLength(len);

            backtrack(list, sb.append("*").append(curr), num, i + 1, eval - product + product * curr, product * curr);
            sb.setLength(len);
        }
    }
}
```

# NP Complete

[Optimal Account Balancing][optimal-account-balancing]

```java
// NP-complete
public int minTransfers(int[][] transactions) {
    Map<Integer, Integer> g = new HashMap<>();
    for (int[] t : transactions) {
        g.put(t[0], g.getOrDefault(t[0], 0) - t[2]);
        g.put(t[1], g.getOrDefault(t[1], 0) + t[2]);
    }
    return backtrack(0, g.values().stream().mapToInt(Integer::valueOf).toArray());
}

private int backtrack(int index, int[] debt) {
    // skips 0 debt
    while (index < debt.length && debt[index] == 0) {
        index++;
    }

    if (index == debt.length) {
        return 0;
    }

    int min = Integer.MAX_VALUE;
    for (int i = index + 1; i < debt.length; i++) {
        // + & -
        if (debt[index] * debt[i] < 0) {
            debt[i] += debt[index];
            min = Math.min(min, 1 + backtrack(index + 1, debt));
            debt[i] -= debt[index];
        }
    }
    return min;
}
```

[Find Minimum Time to Finish All Jobs][find-minimum-time-to-finish-all-jobs]

```java
private int min = Integer.MAX_VALUE;

public int minimumTimeRequired(int[] jobs, int k) {
    backtrack(jobs, 0, new int[k], 0);
    return min;
}

private void backtrack(int[] jobs, int index, int[] workers, int max) {
    if (index == jobs.length) {
        min = Math.min(min, max);
        return;
    }

    // e.g. [10, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    // with the set, 5 is searched only once
    Set<Integer> used = new HashSet<>();
    for (int i = 0; i < workers.length; i++) {
        if (used.add(workers[i]) && workers[i] + jobs[index] < min) {
            workers[i] += jobs[index];
            backtrack(jobs, index + 1, workers, Math.max(workers[i], max));
            workers[i] -= jobs[index];
        }
    }
}
```

# NP-hard

[Maximum Number of Groups Getting Fresh Donuts][maximum-number-of-groups-getting-fresh-donuts]

```java
private Map<List<Integer>, Integer> memo = new HashMap<>();

// NP-hard
public int maxHappyGroups(int batchSize, int[] groups) {
    // list[i]: count of elements with remainder == i
    List<Integer> list = new ArrayList<>(batchSize);
    while (list.size() < batchSize) {
        list.add(0);
    }

    // greedily combines 2 groups whose remainders sum to 0
    int count = 0;
    for (int g : groups) {
        if (g % batchSize == 0) {
            count++;
        } else if (list.get(batchSize - g % batchSize) > 0) {
            list.set(batchSize - g % batchSize, list.get(batchSize - g % batchSize) - 1);
            count++;
        } else {
            list.set(g % batchSize, list.get(g % batchSize) + 1);
        }
    }

    // k-group combinations (k > 2)
    return backtrack(list, 0) + count;
}

// diff = sum(each element in the current list % n) - sum(each element in the original list % n)
// diff is determined by list, so there's no need to use it as a cache key
private int backtrack(List<Integer> list, int diff) {
    if (memo.containsKey(list)) {
        return memo.get(list);
    }

    int max = 0, batchSize = list.size();
    for (int i = 1; i < batchSize; i++) {
        if (list.get(i) > 0) {
            list.set(i, list.get(i) - 1);
            // diff == 0 means the current list is a happy combination
            // so we increment the number of happy groups by one
            max = Math.max(max, (diff == 0 ? 1 : 0) + backtrack(list, (diff - i + batchSize) % batchSize));
            list.set(i, list.get(i) + 1);
        }
    }

    memo.put(new ArrayList<>(list), max);
    return max;
}
```

# Memoization

[Zuma Game][zuma-game]

```java
public int findMinStep(String board, String hand) {
    int[] freq = new int[26];
    for (char c : hand.toCharArray()) {
        freq[c - 'A']++;
    }

    return backtrack(board, freq, new HashMap<String, Integer>());
}

private int backtrack(String board, int[] freq, Map<String, Integer> memo) {
    if (board.isEmpty()) {
        return 0;
    }

    String key = board + "#" + serialize(freq);
    if (memo.containsKey(key)) {
        return memo.get(key);
    }

    // inserts a ball from hand to every possible position of the board
    int min = Integer.MAX_VALUE;
    for (int i = 0; i <= board.length(); i++) {
        for (int j = 0; j < freq.length; j++) {
            if (freq[j] > 0) {
                freq[j]--;

                String b = updateBoard(board.substring(0, i) + (char)('A' + j) + board.substring(i));
                int steps = backtrack(b, freq, memo);
                if (steps >= 0) {
                    min = Math.min(min, steps + 1);
                }

                freq[j]++;
            }
        }
    }

    if (min == Integer.MAX_VALUE) {
        min = -1;
    }

    memo.put(board + "#" + serialize(freq), min);
    return min;
}

private String updateBoard(String board) {
    for (int i = 0, j = 0; i < board.length(); j++) {
        while (i < board.length() && board.charAt(i) == board.charAt(j)) {
            i++;
        }
        if (i - j >= 3) {
            return updateBoard(board.substring(0, j) + board.substring(i));
        } 
    }
    return board;
}

private String serialize(int[] freq) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < freq.length; i++) {
        if (freq[i] > 0) {
            sb.append((char)('A' + i));
            sb.append(freq[i]);
        }
    }
    return sb.toString();
}
```

# Choices & Decision Space

Backtracking explores all the branches of a solution space.

[Maximum Number of Achievable Transfer Requests][maximum-number-of-achievable-transfer-requests]

```java
private int max = 0;

public int maximumRequests(int n, int[][] requests) {
    helper(requests, 0, new int[n], 0);
    return max;
}

private void helper(int[][] requests, int index, int[] count, int num) {
    // traverses all n buildings to see if they are all 0
    // i.e. balanced
    if (index == requests.length) {
        for (int i : count) {
            if (i > 0) {
                return;
            }
        }
        max = Math.max(max, num);
        return;
    }

    // achieves this request
    count[requests[index][0]]++;
    count[requests[index][1]]--;
    helper(requests, index + 1, count, num + 1);
    count[requests[index][0]]--;
    count[requests[index][1]]++;

    // not achieves this request
    helper(requests, index + 1, count, num);
}
```

[Shopping Offers][shopping-offers]

```java
public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
    return backtrack(price, special, needs);
}

private int backtrack(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
    // direct purchase
    int min = 0;
    for (int i = 0; i < needs.size(); i++) {
        min += price.get(i) * needs.get(i);
    }

    // special offer
    for (List<Integer> offer : special) {
        if (isValid(offer, needs)) {
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < needs.size(); i++) {
                tmp.add(needs.get(i) - offer.get(i));
            }
            min = Math.min(min, backtrack(price, special, tmp) + offer.get(offer.size() - 1));
        }
    }

    return min;
}

private boolean isValid(List<Integer> offer, List<Integer> needs) {
    for (int i = 0; i < needs.size(); i++) {
        if (needs.get(i) < offer.get(i)) {
            return false;
        }
    }
    return true;
}
```

[Verbal Arithmetic Puzzle][verbal-arithmetic-puzzle]

```java
private static final int[] POW_10 = new int[]{1, 10, 100, 1000, 10000, 100000, 1000000};
private boolean[] notZero = new boolean[26];
private int[] weight = new int[26];

public boolean isSolvable(String[] words, String result) {
    Set<Character> charSet = new HashSet<>();

    for (String w : words) {
        int m = w.length();
        if (m > 1) {
            notZero[w.charAt(0) - 'A'] = true;
        }
        for (int i = 0; i < m; i++) {
            charSet.add(w.charAt(i));
            weight[w.charAt(i) - 'A'] += POW_10[m - i - 1];
        }
    }

    // sum(words) - result = 0
    int m = result.length();
    if (m > 1) {
        notZero[result.charAt(0) - 'A'] = true;
    }
    for (int i = 0; i < m; i++) {
        charSet.add(result.charAt(i));
        weight[result.charAt(i) - 'A'] -= POW_10[m - i - 1];
    }

    return backtrack(new boolean[10], new ArrayList<>(charSet), 0, 0);
}

// diff = sum(words) - rsult
// used boolean array implicitly keeps a mapping
private boolean backtrack(boolean[] used, List<Character> charList, int index, int diff) {
    if (index == charList.size()) {
        return diff == 0;
    }

    for (int d = 0; d <= 9; d++) {
        char c = charList.get(index);
        if (!used[d] && (d > 0 || !notZero[c - 'A'])) {
            used[d] = true;
            if (backtrack(used, charList, index + 1, diff + weight[c - 'A'] * d)) {
                return true;
            }
            used[d] = false;
        }
    }
    return false;
}
```

[Remove All Ones With Row and Column Flips II][remove-all-ones-with-row-and-column-flips-ii]

```java
public int removeOnes(int[][] grid) {
    int m = grid.length, n = grid[0].length, min = Integer.MAX_VALUE;
    int[] row = new int[n], col = new int[m];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                for (int r = 0; r < m; r++) {
                    col[r] = grid[r][j];
                    grid[r][j] = 0;
                }
                System.arraycopy(grid[i], 0, row, 0, n);
                Arrays.fill(grid[i], 0);

                min = Math.min(min, removeOnes(grid) + 1);

                System.arraycopy(row, 0, grid[i], 0, n);
                for (int r = 0; r < m; r++) {
                    grid[r][j] = col[r];
                }
            }
        }
    }
    return min == Integer.MAX_VALUE ? 0 : min;
}
```

[Maximum Points in an Archery Competition][maximum-points-in-an-archery-competition]

```java
private int maxScore = 0;
private int[] aliceArrows, bestBobArrows;

public int[] maximumBobPoints(int numArrows, int[] aliceArrows) {
    int n = aliceArrows.length;
    this.aliceArrows = aliceArrows;
    this.bestBobArrows= new int[n];

    backtrack(0, numArrows, 0, new int[n]);

    // if there are remaining arrows, Bob won all sections
    // we simply put all of the remaining arrows to the first section
    bestBobArrows[0] += numArrows - Arrays.stream(bestBobArrows).sum();
    return bestBobArrows;
}

private void backtrack(int k, int remainingArrows, int score, int[] bobArrows) {
    int n = bobArrows.length;
    if (k == n) {
        if (score > maxScore) {
            maxScore = score;
            bestBobArrows = Arrays.copyOf(bobArrows, n);
        }
        return;
    }

    // Bob loses
    backtrack(k + 1, remainingArrows, score, bobArrows);

    // Bob wins
    int arrowsNeeded = aliceArrows[k] + 1;
    if (remainingArrows >= arrowsNeeded) {
        int tmp = bobArrows[k];
        bobArrows[k] = arrowsNeeded;
        backtrack(k + 1, remainingArrows - arrowsNeeded, score + k, bobArrows);
        bobArrows[k] = tmp;
    }
}
```

[Find the K-Sum of an Array][find-the-k-sum-of-an-array]

```java
public long kSum(int[] nums, int k) {
    int n = nums.length;
    long maxSum = 0;
    // subsequence sums that need to be subtracted from the maximum sum
    List<Long> subtrahends = new ArrayList<>();
    for (int i = 0; i < n; i++) {
        if (nums[i] >= 0) {
            maxSum += nums[i];
        } else {
            // we can either subtract the min positive number
            // or add min negative number to get next largest number
            // converts all elements to non-negative so that we only subtract
            nums[i] = -nums[i];
        }
    }

    Arrays.sort(nums);

    // {current min value which needs to be subtracted, index}
    Queue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> a[0]));
    pq.offer(new long[]{nums[0], 0});

    while (!pq.isEmpty() && subtrahends.size() < k - 1) {
        long[] curr = pq.poll();
        long subtrahend = curr[0];
        int index = (int)curr[1];
        subtrahends.add(subtrahend);
        if (index < n - 1) {
            // for a sorted array, the following two operations generate all possible subtrahends
            // similar to backtracking
            pq.offer(new long[]{subtrahend + nums[index + 1], index + 1});
            pq.offer(new long[]{nums[index + 1] + subtrahend - nums[index], index + 1});
        }
    }

    return maxSum - (k == 1 ? 0 : subtrahends.get(k - 2));
}
```

[24-game]: https://leetcode.com/problems/24-game/
[android-unlock-patterns]: https://leetcode.com/problems/android-unlock-patterns/
[beautiful-arrangement]: https://leetcode.com/problems/beautiful-arrangement/
[closest-dessert-cost]: https://leetcode.com/problems/closest-dessert-cost/
[combination-sum]: https://leetcode.com/problems/combination-sum/
[combination-sum-ii]: https://leetcode.com/problems/combination-sum-ii/
[combination-sum-iii]: https://leetcode.com/problems/combination-sum-iii/
[construct-the-lexicographically-largest-valid-sequence]: https://leetcode.com/problems/construct-the-lexicographically-largest-valid-sequence/
[expression-add-operators]: https://leetcode.com/problems/expression-add-operators/
[factor-combinations]: https://leetcode.com/problems/factor-combinations/
[fair-distribution-of-cookies]: https://leetcode.com/problems/fair-distribution-of-cookies/
[find-minimum-time-to-finish-all-jobs]: https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs/
[find-the-k-sum-of-an-array]: https://leetcode.com/problems/find-the-k-sum-of-an-array/
[generalized-abbreviation]: https://leetcode.com/problems/generalized-abbreviation/
[letter-tile-possibilities]: https://leetcode.com/problems/letter-tile-possibilities/
[matchsticks-to-square]: https://leetcode.com/problems/matchsticks-to-square/
[maximum-number-of-achievable-transfer-requests]: https://leetcode.com/problems/maximum-number-of-achievable-transfer-requests/
[maximum-number-of-groups-getting-fresh-donuts]: https://leetcode.com/problems/maximum-number-of-groups-getting-fresh-donuts/
[maximum-points-in-an-archery-competition]: https://leetcode.com/problems/maximum-points-in-an-archery-competition/
[maximum-score-words-formed-by-letters]: https://leetcode.com/problems/maximum-score-words-formed-by-letters/
[minimum-moves-to-spread-stones-over-grid]: https://leetcode.com/problems/minimum-moves-to-spread-stones-over-grid/
[optimal-account-balancing]: https://leetcode.com/problems/optimal-account-balancing/
[palindrome-partitioning]: https://leetcode.com/problems/palindrome-partitioning/
[palindrome-permutation-ii]: https://leetcode.com/problems/palindrome-permutation-ii/
[partition-equal-subset-sum]: https://leetcode.com/problems/partition-equal-subset-sum/
[partition-to-k-equal-sum-subsets]: https://leetcode.com/problems/partition-to-k-equal-sum-subsets/
[permutations]: https://leetcode.com/problems/permutations/
[permutations-ii]: https://leetcode.com/problems/permutations-ii/
[remove-all-ones-with-row-and-column-flips-ii]: https://leetcode.com/problems/remove-all-ones-with-row-and-column-flips-ii/
[robot-room-cleaner]: https://leetcode.com/problems/robot-room-cleaner/
[shopping-offers]: https://leetcode.com/problems/shopping-offers/
[subsets]: https://leetcode.com/problems/subsets/
[subsets-ii]: https://leetcode.com/problems/subsets-ii/
[verbal-arithmetic-puzzle]: https://leetcode.com/problems/verbal-arithmetic-puzzle/
[zuma-game]: https://leetcode.com/problems/zuma-game/
