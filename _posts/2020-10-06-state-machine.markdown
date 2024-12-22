---
title:  "State Machine"
category: algorithm
mermaid: true
---
[Best Time to Buy and Sell Stock with Transaction Fee][best-time-to-buy-and-sell-stock-with-transaction-fee]

```mermaid
stateDiagram-v2
    S0 --> S1: buy
    S1 --> S0: sell
    S0 --> S0: rest
    S1 --> S1: rest
```

```java
public int maxProfit(int[] prices, int fee) {
    int[] s0 = new int[prices.length];  // cash
    int[] s1 = new int[prices.length];  // hold

    s0[0] = 0;
    s1[0] = -prices[0];

    for (int i = 1; i < prices.length; i++) {
        s0[i] = Math.max(s0[i - 1], s1[i - 1] + prices[i] - fee);
        s1[i] = Math.max(s1[i - 1], s0[i - 1] - prices[i]);
    }

    return s0[prices.length - 1];
}
```

Reduced to 0D:

```java
public int maxProfit(int[] prices, int fee) {
    int hold = -prices[0], cash = 0;
    for (int price : prices) {
        int prevCash = cash;
        cash = Math.max(cash, hold + price - fee);
        hold = Math.max(hold, prevCash - price);
    }
    return Math.max(cash, hold);
}
```

It can be simplified as:

```java
public int maxProfit(int[] prices, int fee) {
    int hold = -prices[0], cash = 0;
    for (int price : prices) {
        cash = Math.max(cash, hold + price - fee);

        // hold2 = max(hold1, cash2 - prices[i]);
        //   = max(hold1, max(cash1, hold1 + prices[i] - fee) - prices[i])
        //   = max(hold1, max(cash1 - prices[i], hold1 - fee))
        //   = max(hold1, cash1 - prices[i], hold1 - fee)
        //   = max(hold1, cash1 - prices[i])
        hold = Math.max(hold, cash - price);
    }

    // max(hold2, cash2)
    //   = max(hold1, cash1 - prices[i], cash1, hold1 + prices[i] - fee)
    //   = max(hold1, cash1, hold1 + prices[i] - fee)
    //   = max(hold1, cash2)
    //   ...
    //   = Math.max(-prices[0], cash2)
    // -prices[0] < 0 = cash0 <= cash2
    //
    // The profit after selling is higher than holding
    return cash;
}
```

[Best Time to Buy and Sell Stock with Cooldown][best-time-to-buy-and-sell-stock-with-cooldown]

```mermaid
stateDiagram-v2
    S0 --> S1: buy
    S1 --> S2: sell
    S0 --> S0: rest
    S1 --> S1: rest
    S2 --> S0: rest
```

```java
public int maxProfit(int[] prices) {
    if (prices.length == 0) {
        return 0;
    }

    int[] s0 = new int[prices.length];  // cash, not immediate after selling
    int[] s1 = new int[prices.length];  // hold
    int[] s2 = new int[prices.length];  // sold, immediate after selling

    s0[0] = 0;
    s1[0] = -prices[0];
    s2[0] = Integer.MIN_VALUE;

    for (int i = 1; i < prices.length; i++) {
        s0[i] = Math.max(s0[i - 1], s2[i - 1]);
        s1[i] = Math.max(s1[i - 1], s0[i - 1] - prices[i]);
        s2[i] = s1[i - 1] + prices[i];
    }

    return Math.max(s0[prices.length - 1], s2[prices.length - 1]);
}
```

Reduced to 0D:

```java
public int maxProfit(int[] prices) {
    int cash = 0, hold = Integer.MIN_VALUE, sold = 0;
    for (int price : prices) {
        int prevSold = sold;
        sold = hold + price;
        hold = Math.max(hold, cash - price);
        cash = Math.max(cash, prevSold);
    }
    return Math.max(sold, cash);
}
```

[Minimum Swaps To Make Sequences Increasing][minimum-swaps-to-make-sequences-increasing]

```java
public int minSwap(int[] A, int[] B) {
    int s1 = 0, s2 = 1;  // same, swap
    for (int i = 1; i < A.length; i++) {
        int tmp = s1;
        if (A[i - 1] < A[i] && B[i - 1] < B[i]) {
            if (A[i - 1] < B[i] && B[i - 1] < A[i]) {
                s1 = Math.min(s1, s2);
                s2 = Math.min(tmp, s2) + 1;
            } else {
                s2++;
            }
        } else {
            s1 = s2;
            s2 = tmp + 1;
        }
    }
    return Math.min(s1, s2);
}
```

# Deterministic Finite Automation

[Deterministic finite automaton (DFA)](https://en.wikipedia.org/wiki/Deterministic_finite_automaton): a finite-state machine that accepts or rejects a given string of symbols, by running through a state sequence uniquely determined by the string.

[Valid Number][valid-number]

```mermaid
stateDiagram-v2
    direction LR
    0 --> 1
    0 --> 2
    2 --> 3
    0 --> 3
    1 --> 1
    2 --> 1
    1 --> 4
    3 --> 4
    4 --> 4
    1 --> 5
    4 --> 5
    5 --> 6
    5 --> 7
    6 --> 7
    7 --> 7
```

```java
private static final List<Map<String, Integer>> dfa = List.of(
    Map.of("digit", 1, "sign", 2, "dot", 3),
    Map.of("digit", 1, "dot", 4, "exponent", 5),
    Map.of("digit", 1, "dot", 3),
    Map.of("digit", 4),
    Map.of("digit", 4, "exponent", 5),
    Map.of("sign", 6, "digit", 7),
    Map.of("digit", 7),
    Map.of("digit", 7)
);

private static final Set<Integer> validFinalStates = Set.of(1, 4, 7);

public boolean isNumber(String s) {
    int currState = 0;
    String group = null;

    for (char ch : s.toCharArray()) {
        if (Character.isDigit(ch)) {
            group = "digit";
        } else if (ch == '+' || ch == '-') {
            group = "sign";
        } else if (ch == 'e' || ch == 'E') {
            group = "exponent";
        } else if (ch == '.') {
            group = "dot";
        } else {
            return false;
        }

        if (!dfa.get(currState).containsKey(group)) {
            return false;
        }

        currState = dfa.get(currState).get(group);
    }

    return validFinalStates.contains(currState);
}
```

[best-time-to-buy-and-sell-stock-with-cooldown]: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
[best-time-to-buy-and-sell-stock-with-transaction-fee]: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
[minimum-swaps-to-make-sequences-increasing]: https://leetcode.com/problems/minimum-swaps-to-make-sequences-increasing/
[valid-number]: https://leetcode.com/problems/valid-number/
