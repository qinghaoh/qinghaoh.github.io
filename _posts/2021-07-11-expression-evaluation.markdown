---
title:  "Expression Evaluation"
category: algorithm
---
# Fundamentals

|    Infix    |  Prefix (RN)  |  Postfix (RPN)  |
|-------------------------------------|
|    a + b    |   + a b   |   a b +   |
|  a + b * c  | + a * b c | a b c * + |
| (a + b) * c | * + a b c | a b + c * |
| | | stack |

* PN: Polish Notation
* RPN: Reverse Polish Notation

# Expression Tree

[Build Binary Expression Tree From Infix Expression][build-binary-expression-tree-from-infix-expression]

```java
public Node expTree(String s) {
    // converts infix to postfix
    StringBuilder postfix = new StringBuilder();
    Deque<Character> opstack = new ArrayDeque<>();
    for (char c : s.toCharArray()) {
        if (Character.isDigit(c)) {
            postfix.append(c);
        } else {
            if (c == '(') {
                opstack.push(c);
            } else if (c == ')') {
                while (opstack.peek() != '(') {
                    postfix.append(opstack.pop());
                }
                opstack.pop();
            } else if (c == '+' || c == '-') {
                while (!opstack.isEmpty() && opstack.peek() != '(') {
                    postfix.append(opstack.pop());
                }
                opstack.push(c);
            } else {
                while (!opstack.isEmpty() && (opstack.peek() == '*' || opstack.peek() == '/')) {
                    postfix.append(opstack.pop());
                }
                opstack.push(c);
            }
        }
    }

    while (!opstack.isEmpty()) {
        postfix.append(opstack.pop());
    }

    // builds inorder expression tree from postfix
    Deque<Node> operandStack = new ArrayDeque<>();
    for (char c : postfix.toString().toCharArray()) {
        if (Character.isDigit(c)) {
            operandStack.push(new Node(c));
        } else {
            Node right = operandStack.pop(), left = operandStack.pop();
            operandStack.push(new Node(c, left, right)); 
        }
    }
    return operandStack.pop();
}
```

[Basic Calculator III][basic-calculator-iii]

```java
private static final Map<Character, Integer> PRECEDENCE = new HashMap<>();

static {
    PRECEDENCE.put('(', -1);
    PRECEDENCE.put('+', 0);
    PRECEDENCE.put('-', 0);
    PRECEDENCE.put('*', 1);
    PRECEDENCE.put('/', 1);
}

public int calculate(String s) {
    // infix -> postfix
    Deque<Integer> operands = new ArrayDeque<>();
    Deque<Character> operators = new ArrayDeque<>();
    int n = s.length();
    for (int i = 0; i < n; i++) {
        char c = s.charAt(i);
        if (Character.isDigit(c)) {
            int val = Character.getNumericValue(s.charAt(i));
            while (i + 1 < n && Character.isDigit(s.charAt(i + 1))) {
                val = val * 10 + Character.getNumericValue(s.charAt(i + 1));
                i++;
            }
            operands.push(val);
        } else if (c == '(') {
            operators.push(c);
        } else if (c == ')') {
            while (operators.peek() != '(') {
                operands.push(operate(operands, operators));
            }
            operators.pop();
        } else {
            while (!operators.isEmpty() && compare(c, operators.peek()) <= 0) {
                operands.push(operate(operands, operators));
            }
            operators.push(c);
        }
    }

    while (!operators.isEmpty()) {
        operands.push(operate(operands, operators));
    }

    return operands.pop();
}

// precedence
private int compare(char a, char b) {
    return PRECEDENCE.get(a) - PRECEDENCE.get(b);
}

private int operate(Deque<Integer> operands, Deque<Character> operators) {
    int a = operands.pop(), b = operands.pop();
    char c = operators.pop();

    switch (c) {
        case '+' : return a + b;
        case '-': return b - a;
        case '*': return a * b;
        case '/': return b / a;
        default: return 0;
    }
}
```

# Dynamic Programming

[The Score of Students Solving Math Expression][the-score-of-students-solving-math-expression]

```java
private static final int MAX_ANSWER = 1000;

public int scoreOfStudents(String s, int[] answers) {
    // number of digits
    int n = s.length() / 2 + 1;
    // dp[i][j]: set of possible results of s[i...j], where the index is of digits only
    // e.g. 1 + 2 * 3, index 2 stands for 3
    Set<Integer>[][] dp = new Set[n][n];
    for (int i = 0; i < n; i++) {
        dp[i][i] = new HashSet<>();
        // 2 * i is the actual index of the digit in the stringg
        dp[i][i].add(s.charAt(2 * i) - '0');
    }

    for (int len = 1; len < n; len++) {
        for (int i = 0; i + len < n; i++) {
            int j = i + len;
            dp[i][j] = new HashSet<>();
            // index of operators
            for (int k = 2 * i + 1; k < 2 * j; k += 2) {
                for (int a : dp[i][k / 2]) {
                    for (int b : dp[k / 2 + 1][j]) {
                        if (s.charAt(k) == '+' && a + b <= MAX_ANSWER) {
                            dp[i][j].add(a + b);
                        } else if (s.charAt(k) == '*' && a * b <= MAX_ANSWER) {
                            dp[i][j].add(a * b);
                        }

                    }
                }
            }
        }
    }

    int val = evaluate(s), points = 0;
    for (int a : answers) {
        if (a == val) {
            points += 5;
        } else if (dp[0][n - 1].contains(a)) {
            points += 2;
        }
    }
    return points;
}

// evaluate
private int evaluate(String s) {
    Deque<Integer> st = new ArrayDeque<>();
    char operator = '+';
    int i = 0, num = 0;
    while (i < s.length()) {
        char ch = s.charAt(i++);
        if (Character.isDigit(ch)) {
            num = ch - '0';
        }
        if (i >= s.length() || ch == '+' || ch == '*') {
            if (operator == '+') {
                st.push(num);
            } else if (operator == '*') {
                st.push(st.pop() * num);
            }
            operator = ch;
        }
    }
    return st.stream().mapToInt(Integer::intValue).sum();
}
```

[basic-calculator-iii]: https://leetcode.com/problems/basic-calculator-iii/
[build-binary-expression-tree-from-infix-expression]: https://leetcode.com/problems/build-binary-expression-tree-from-infix-expression/
[the-score-of-students-solving-math-expression]: https://leetcode.com/problems/the-score-of-students-solving-math-expression/
