---
title:  "Recursion"
category: algorithm
tags: recursion
---

[Special Binary String][special-binary-string]

```java
public String makeLargestSpecial(String s) {
    List<String> list = new ArrayList<>();
    int count = 0, i = 0, j = 0;
    // splits s into as many special strings as possible
    while (j < s.length()) {
        // count('1') - count('0')
        if (s.charAt(j) == '1') {
            count++;
        } else {
            count--;
        }

        if (count == 0) {
            // 1M0: M must be another special string.
            // proof:
            // If there is a prefix P of M which has one less 1's than 0's
            // 1P would have been processed as the prefix of a special string in earlier recursions
            list.add('1' + makeLargestSpecial(s.substring(i + 1, j)) + '0');
            i = j + 1;
        }
        j++;
    }
    Collections.sort(list, Collections.reverseOrder());
    return String.join("", list);
}
```

[Strobogrammatic Number II][strobogrammatic-number-ii]

```java
public List<String> findStrobogrammatic(int n) {
    return findStrobogrammatic(n, n);
}

private List<String> findStrobogrammatic(int n, int initialN) {
    if (n == 0) {
        return new ArrayList<>(Arrays.asList(""));
    }

    if (n == 1) {
        return new ArrayList<>(Arrays.asList("0", "1", "8"));
    }

    List<String> list = new ArrayList<>();
    for (String s : findStrobogrammatic(n - 2, initialN)) {
        if (n != initialN) {
            list.add("0" + s + "0");
        }
        list.add("1" + s + "1");
        list.add("8" + s + "8");
        list.add("6" + s + "9");
        list.add("9" + s + "6");
    }
    return list;
}
```

[Strobogrammatic Number III][strobogrammatic-number-iii]

```java
{% raw %}
private static final char[][] PAIRS = {{'0', '0'}, {'1', '1'}, {'6', '9'}, {'8', '8'}, {'9', '6'}};
{% endraw %}
private String low, high;

public int strobogrammaticInRange(String low, String high) {
    this.low = low;
    this.high = high;

    int count = 0;
    for (int len = low.length(); len <= high.length(); len++) {
        count += helper(new char[len], 0, len - 1);
    }
    return count;
}

private int helper(char[] ch, int left, int right) {
    if (left > right) {
        String s = new String(ch);
        return (ch.length == low.length() && s.compareTo(low) < 0) ||
            (ch.length == high.length() && s.compareTo(high) > 0) ? 0 : 1;
    }

    int count = 0;
    for (char[] p : PAIRS) {
        ch[left] = p[0];
        ch[right] = p[1];

        // don't start with 0
        if (ch.length != 1 && ch[0] == '0') {
            continue;
        }

        // don't put 6/9 at the middle of string
        if (left == right && (p[0] == '6' || p[0] == '9')) {
            continue;
        }

        count += helper(ch, left + 1, right - 1);
    }
    return count;
}
```

[Largest Merge Of Two Strings][largest-merge-of-two-strings]

```java
public String largestMerge(String word1, String word2) {
    if (word1.length() == 0  || word2.length() == 0) {
        return word1 + word2;
    }

    return word1.compareTo(word2) > 0 ?
        word1.charAt(0) + largestMerge(word1.substring(1), word2) : word2.charAt(0) + largestMerge(word1, word2.substring(1));
}
```

[Remove Invalid Parentheses][remove-invalid-parentheses]

```java
public List<String> removeInvalidParentheses(String s) {
    List<String> list = new ArrayList<>();
    dfs(s, 0, 0, '(', ')', list);
    return list;
}

public void dfs(String s, int iStart, int jStart, char c1, char c2, List<String> list) {
    // conceptually, c1 is the open parenthesis, c2 is the open parenthesis
    int open = 0, closed = 0;
    for (int i = iStart; i < s.length(); i++) {
        char c = s.charAt(i);
        if (c == c1) {
            open++;
        }
        if (c == c2) {
            closed++;
        }

        // there's an extra closed parenthesis needs to be removed
        if (closed > open) {
            for (int j = jStart; j <= i; j++) {
                // removes the first closed parenthesis in each sequence of consecutive closed parentheses
                // skips duplicates
                if (s.charAt(j) == c2 && (j == jStart || s.charAt(j - 1) != c2)) {
                    // now open == closed until i
                    // j is actually the original j + 1
                    dfs(s.substring(0, j) + s.substring(j + 1), i, j, c1, c2, list);
                }
            }
            return;
        }
    }

    // reverses the String and removes invalid open parentheses
    String reversed = new StringBuilder(s).reverse().toString();
    if (c1 == '(') {
        dfs(reversed, 0, 0, ')','(', list);
    } else {
        list.add(reversed);
    }
}
```

[Minimum Cost to Change the Final Value of Expression][minimum-cost-to-change-the-final-value-of-expression]

```java
// index of matching '('
private Map<Integer, Integer> map = new HashMap<>();

public int minOperationsToFlip(String expression) {
    int n = expression.length();
    Deque<Integer> st = new ArrayDeque<>();
    for (int i = 0; i < n; i++) {
        char ch = expression.charAt(i);
        if (ch == '(') {
            st.push(i);
        } else if (ch == ')') {
            map.put(i, st.pop());
        }
    }
    return evaluate(expression, 0, n - 1)[1];
}

// {value before operations, number of operations}
private int[] evaluate(String expression, int i, int j) {
    if (i == j) {
        return new int[]{expression.charAt(i) - '0', 1};
    }

    char op = 0;
    int[] left, right;
    if (Character.isDigit(expression.charAt(j))) {
        // oj = map.get(j)
        // (0 & 0) & 0
        // i         j
        left = evaluate(expression, i, j - 2);
        right = evaluate(expression, j, j);
        op = expression.charAt(j - 1);
    } else {
        // recursion
        if (map.get(j) == i) {
            return evaluate(expression, i + 1, j - 1);
        }

        // oj = map.get(j)
        // (0 & 0) & (0 & 0 & 0)
        // i         oj        j
        left = evaluate(expression, i, map.get(j) - 2);
        right = evaluate(expression, map.get(j), j);
        op = expression.charAt(map.get(j) - 1);
    }

    if (op == '&') {
        if ((left[0] ^ right[0]) == 1) {
            return new int[]{0, 1};
        }
        if ((left[0] & right[0]) == 1) {
            return new int[]{1, Math.min(left[1], right[1])};
        }
        return new int[]{0, 1 + Math.min(left[1], right[1])};
    } else {
        if ((left[0] ^ right[0]) == 1) {
            return new int[]{1, 1};
        }
        if ((left[0] & right[0]) == 1) {
            return new int[]{1, 1 + Math.min(left[1], right[1])};
        }
        return new int[]{0, Math.min(left[1], right[1])};
    }
}
```

[Self Crossing][self-crossing]

```java
public boolean isSelfCrossing(int[] x) {
    for (int i = 3; i < x.length; i++) {
        //    i-2
        // i-1┌─┐
        //    └─┼─>i
        //     i-3
        if (x[i] >= x[i - 2] && x[i - 1] <= x[i - 3]) {
            return true;
        }

        //      i-2
        // i-1 ┌────┐
        //     └─══>┘i-3
        //     i  i-4      (i overlapped i-4)
        if (i >= 4) {
            if (x[i - 1] == x[i - 3] && x[i] + x[i - 4] >= x[i - 2]) {
                return true;
            }
        }

        //    i-4
        //    ┌──┐
        //    │i<┼─┐
        // i-3│ i-5│i-1
        //    └────┘
        //     i-2
        if (i >= 5) {
            if (x[i - 2] - x[i - 4] >= 0
                && x[i] >= x[i - 2] - x[i - 4]
                && x[i - 1] >= x[i - 3] - x[i - 5]
                && x[i - 1] <= x[i - 3]) {
                return true;  // Sixth line crosses first line and onward
            }
        }
    }
    return false;
}
```

[Least Operators to Express Number][least-operators-to-express-number]

```java
private Map<Integer, Integer> memo = new HashMap<>();

public int leastOpsExpressTarget(int x, int target) {
    if (x == target) {
        return 0;
    }

    if (x > target) {
        // adds `target` times `x / x`
        // or substracts `x - target` times `x / x` from x
        return Math.min(target * 2 - 1, (x - target) * 2);
    }

    if (memo.containsKey(target)) {
        return memo.get(target);
    }

    // greedy
    // multiplies `x` as many as possible
    long expression = x;
    int multiplications = 0;
    while (expression < target) {
        multiplications++;
        expression *= x;
    }

    if (expression == target) {
        return multiplications;
    }

    // expression > target
    int min = Integer.MAX_VALUE;

    // the if condition bounds the recursion to be finite
    // otherwise the new target `expression - target` will become bigger and bigger
    if (expression - target < target) {
        // subtraction:
        // target = x ^ n - (x ^ n - target)
        // operations = multiplications - f(expression - target)
        min = Math.min(min, multiplications + 1 + leastOpsExpressTarget(x, (int)(expression - target)));
    }

    // addition:
    // the multiplications went too far; we need to go one step back
    // target = x ^ (n - 1) + 1 + (target - x ^ n / x)
    // operators = (multiplications - 1) + 1 + f(target - expression / x)
    //   = multiplications + f(target - expression / x);
    min = Math.min(min, multiplications + leastOpsExpressTarget(x, (int)(target - expression / x)));
    memo.put(target, min);
    return min;
}
```

# Parse

[Brace Expansion II][brace-expansion-ii]

```java
public List<String> braceExpansionII(String expression) {
    List<List<String>> groups = new ArrayList<>();
    Set<String> set = new TreeSet<>();
    int level = 0, start = -1;
    for (int i = 0; i < expression.length(); i++) {
        char c = expression.charAt(i);
        if (c == '{') {
            if (level++ == 0) {
                start = i + 1;
            }
        } else if (c == '}') {
            if (--level == 0) {
                groups.add(braceExpansionII(expression.substring(start, i)));
            }
        } else if (c == ',' && level == 0) {
            // processes the groups so far
            set.addAll(combine(groups));
            groups.clear();
        } else if (level == 0) {
            // singleton set at base level
            groups.add(Arrays.asList(String.valueOf(c)));
        }
    }
    set.addAll(combine(groups));
    return new ArrayList<>(set);
}

// {a, b}{c, d} -> {ac, bc, ad, bd}
private List<String> combine(List<List<String>> groups) {
    List<String> prev = Collections.singletonList("");
    for (List<String> group : groups) {
        List<String> curr = new ArrayList<>();
        for (String p : prev) {
            for (String g : group) {
                curr.add(p + g);
            }
        }
        prev = curr;
    }
    return prev;
}
```

# Mutual Recursion

[Elimination Game][elimination-game]

```java
public int lastRemaining(int n) {
    boolean fromLeftToRight = true;
    int remaining = n, step = 1, head = 1;
    while (remaining > 1) {
        // updates head if it's from left to right,
        // or remaining is odd
        if (fromLeftToRight || remaining % 2 == 1) {
            head += step;
        }

        remaining /= 2;
        step *= 2;
        fromLeftToRight = !fromLeftToRight;
    }
    return head;
}
```

```java
public int lastRemaining(int n) {
    return leftToRight(n);
}

private static int leftToRight(int n) {
    if (n <= 2) {
        return n;
    }
    return 2 * rightToLeft(n / 2);
}

private static int rightToLeft(int n) {
    if (n <= 2) {
        return 1;
    }

    return n % 2 == 1 ? 2 * leftToRight(n / 2) : 2 * leftToRight(n / 2) - 1;
}
```

```java
public int lastRemaining(int n) {
    // mirroring:
    // l(n) + r(n) == 1 + n
    return n == 1 ? 1 : 2 * (1 + n / 2 - lastRemaining(n / 2));
}
```

[brace-expansion-ii]: https://leetcode.com/problems/brace-expansion-ii/
[elimination-game]: https://leetcode.com/problems/elimination-game/
[largest-merge-of-two-strings]: https://leetcode.com/problems/largest-merge-of-two-strings/
[least-operators-to-express-number]: https://leetcode.com/problems/least-operators-to-express-number/
[minimum-cost-to-change-the-final-value-of-expression]: https://leetcode.com/problems/minimum-cost-to-change-the-final-value-of-expression/
[remove-invalid-parentheses]: https://leetcode.com/problems/remove-invalid-parentheses/
[self-crossing]: https://leetcode.com/problems/self-crossing/
[special-binary-string]: https://leetcode.com/problems/special-binary-string/
[strobogrammatic-number-ii]: https://leetcode.com/problems/strobogrammatic-number-ii/
[strobogrammatic-number-iii]: https://leetcode.com/problems/strobogrammatic-number-iii/
