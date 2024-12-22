---
title:  "Parentheses"
category: algorithm
---

[Reverse Substrings Between Each Pair of Parentheses][reverse-substrings-between-each-pair-of-parentheses]

```java
public String reverseParentheses(String s) {
    Deque<Integer> st = new ArrayDeque<>();
    int[] pairs = new int[s.length()];
    for (int i = 0; i < s.length(); i++) {
        if (s.charAt(i) == '(')
            st.push(i);
        if (s.charAt(i) == ')') {
            int j = st.pop();
            pairs[i] = j;
            pairs[j] = i;
        }
    }

    StringBuilder sb = new StringBuilder();
    for (int i = 0, d = 1; i < s.length(); i += d) {
        if (s.charAt(i) == '(' || s.charAt(i) == ')') {
            i = pairs[i];
            d = -d;  // changes direction
        } else {
            sb.append(s.charAt(i));
        }
    }

    return sb.toString();
}
```

# Regress to Counter

[Remove Outermost Parentheses][remove-outermost-parentheses]

```java
public string removeouterparentheses(string s) {
    stringbuilder sb = new stringbuilder();
    int open = 0;
    for (char c : s.tochararray()) {
        if ((c == '(' && open++ > 0) || (c == ')' && --open > 0)) {
            sb.append(c);
        }
    }
    return sb.tostring();
}
```

[Minimum Add to Make Parentheses Valid][minimum-add-to-make-parentheses-valid]

```java
public int minAddToMakeValid(String S) {
    int notOpened = 0;  // '(' needed to make the String balanced
    int notClosed = 0;  // ')' needed to make the String balanced
    for (char c : S.toCharArray()) {
        if (c == '(') {
            notClosed++;
        } else if (notClosed == 0) {
            notOpened++;
        } else {
            notClosed--;
        }
    }

    return notOpened + notClosed;
}
```

[Minimum Insertions to Balance a Parentheses String][minimum-insertions-to-balance-a-parentheses-string]

```java
public int minInsertions(String s) {
    int count = 0;
    int notClosed = 0;  // ')' needed to make the String balanced
    for (char c : s.toCharArray()) {
        if (c == '(') {
            if (notClosed % 2 > 0) {
                notClosed--;
                count++;
            }
            notClosed += 2;
        } else {
            notClosed--;
            if (notClosed < 0) {
                notClosed += 2;
                count++;
            }
        }
    }

    return count + notClosed;
}
```

[Maximum Nesting Depth of Two Valid Parentheses Strings][maximum-nesting-depth-of-two-valid-parentheses-strings]

```java
public int[] maxDepthAfterSplit(String seq) {
    int[] result = new int[seq.length()];
    int opened = 0;
    for (int i = 0; i < seq.length(); i++) {
        if (seq.charAt(i) == '(') {
            opened++;
        }

        result[i] = opened % 2;  // split by parity

        if (seq.charAt(i) == ')') {
            opened--;
        }
    }

    return result;
}
```

[Score of Parentheses][score-of-parentheses]

```java
public int scoreOfParentheses(String S) {
    Deque<Integer> st = new ArrayDeque<>();
    int curr = 0;
    for (char c : S.toCharArray()) {
        if (c == '(') {
            st.push(curr);
            curr = 0;
        } else {
            curr = st.pop() + Math.max(curr * 2, 1);
        }
    }
    return curr;
}
```

```java
public int scoreOfParentheses(String s) {
    int score = 0, opened = 0;
    for (int i = 0; i < s.length(); i++) {
        if (s.charAt(i) == '(') {
            opened++;
        } else {
            opened--;
            if (s.charAt(i - 1) == '(') {
                // number of exterior sets of parentheses that contains this core
                score += 1 << opened;
            }
        }
    }
    return score;
}
```

[Longest Valid Parentheses][longest-valid-parentheses]

```java
public int longestValidParentheses(String s) {
    Deque<Integer> st = new ArrayDeque<>();
    st.push(-1);  // imagine there's a ')' at index -1

    int max = 0;
    for (int i = 0; i < s.length(); i++) {
        if (s.charAt(i) == '(') {
            st.push(i);
        } else {
            st.pop();
            if (st.isEmpty()) {
                // uses the current ')' as a checkpoint
                st.push(i);
            } else {
                max = Math.max(max, i - st.peek());
            }
        }
    }
    return max;
}
```

This problem can also be solved by two string scans with better space complexity:

```java
public int longestValidParentheses(String s) {
    return Math.max(longestValidParentheses(s, true), longestValidParentheses(s, false));
}

// scans the string in one direction
private int longestValidParentheses(String s, boolean isForward) {
    int n = s.length(), open = 0, close = 0, max = 0;
    // scans left to right
    for (int i = isForward ? 0 : n - 1; i >= 0 && i < n; i = i + (isForward ? 1 : - 1)) {
        char ch = s.charAt(i);
        if ((isForward && ch == '(') || (!isForward && ch == ')')) {
            open++;
        } else {
            close++;
        }

        if (open == close) {
            max = Math.max(max, 2 * close);
        } else if (open < close) {
            open = close = 0;
        }
    }
    return max;
}
```

[Valid Parenthesis String][valid-parenthesis-string]

```java
public boolean checkValidString(String s) {
    // number of open parenthses is in [min, max];
    int min = 0, max = 0;
    for (char c : s.toCharArray()) {
        if (c == '(') {
            max++;
            min++;
        } else if (c == ')') {
            max--;
            min--;
        } else {
            // '*' -> '('
            max++;
            // '*' -> ')'
            min--; // if `*` become `)` then openCount--
        }

        if (max < 0) {
            return false;
        }
        min = Math.max(min, 0);
    }
    return min == 0;
}
```

Similar: [Check if a Parentheses String Can Be Valid][check-if-a-parentheses-string-can-be-valid]

```java
public boolean canBeValid(String s, String locked) {
    return s.length() % 2 == 0 && checkValidString(s, locked);
}
```

[Minimum Remove to Make Valid Parentheses][minimum-remove-to-make-valid-parentheses]

```c++
string minRemoveToMakeValid(string s) {
    stack<int> st;
    for (int i = 0; i < s.length(); i++) {
        if (s[i] == '(') {
            st.push(i);
        }
        if (s[i] == ')') {
            if (!st.empty()) {
                st.pop();
            } else {
                s[i] = '*';
            }
        }
    }
    while (!st.empty()) {
        s[st.top()] = '*';
        st.pop();
    }
    s.erase(remove(s.begin(), s.end(), '*'), s.end());
    return s;
}
```

```java
public String minRemoveToMakeValid(String s) {
    StringBuilder sb = new StringBuilder(s);
    Deque<Integer> st = new ArrayDeque<>();
    for (int i = 0; i < sb.length(); i++) {
        if (sb.charAt(i) == '(') {
            st.push(i + 1);
        }
        if (sb.charAt(i) == ')') {
            if (!st.isEmpty() && st.peek() >= 0) {
                st.pop();
            } else {
                st.push(-(i + 1));
            }
        }
    }

    while (!st.isEmpty()) {
        sb.deleteCharAt(Math.abs(st.pop()) - 1);
    }
    return sb.toString();
}
```

[check-if-a-parentheses-string-can-be-valid]: https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/
[longest-valid-parentheses]: https://leetcode.com/problems/longest-valid-parentheses/
[maximum-nesting-depth-of-two-valid-parentheses-strings]: https://leetcode.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/
[minimum-add-to-make-parentheses-valid]: https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/
[minimum-insertions-to-balance-a-parentheses-string]: https://leetcode.com/problems/minimum-insertions-to-balance-a-parentheses-string/
[minimum-remove-to-make-valid-parentheses]: https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/
[remove-outermost-parentheses]: https://leetcode.com/problems/remove-outermost-parentheses/
[reverse-substrings-between-each-pair-of-parentheses]: https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/
[score-of-parentheses]: https://leetcode.com/problems/score-of-parentheses/
[valid-parenthesis-string]: https://leetcode.com/problems/valid-parenthesis-string/
