---
title:  "Probability Theory"
category: algorithm
tags: math
---
[Guess the Word][guess-the-word]

```java
public void findSecretWord(String[] wordlist, Master master) {
    int n = wordlist.length;
    // 10 guesses are allowed at most
    for (int g = 0, matches = 0; g < 10 && matches < 6; g++) {
        // count[i][j]: frequency of char j at the i-th position of all words
        int[][] count = new int[6][26];
        for (String w : wordlist) {
            for (int i = 0; i < w.length(); i++) {
                count[i][w.charAt(i) - 'a']++;
            }
        }

        // the possiblity that a word has 0 match with the secret words is (25 / 26) ^ 6 ~= 80%
        // say we have a group of candidate words from the wordlist,
        // and each word in the group has at least one match with the guess
        // then we want to maximize the group size so if the guess is wrong, we eliminate most words
        int max = 0;
        String guess = wordlist[0];
        for (String w : wordlist) {
            // score is the sum of overall frequency of all chars in this word
            // the higher the score is, the larger the group is
            int score = 0;
            for (int i = 0; i < w.length(); i++) {
                score += count[i][w.charAt(i) - 'a'];
            }

            if (score > max) {
                guess = w;
                max = score;
            }
        }

        // now guess has the highest score
        matches = master.guess(guess);
        List<String> tmp = new ArrayList<>();
        for (String w : wordlist) {
            if (getMatches(guess, w) == matches) {
                tmp.add(w);
            }
        }
        wordlist = tmp.toArray(new String[0]);
    }
}

private int getMatches(String a, String b) {
    int matches = 0;
    for (int i = 0; i < a.length(); i++) {
        if (a.charAt(i) == b.charAt(i)) {
            matches++;
        }
    }
    return matches;
}
```

# Dynamic Programming 

[New 21 Game][new-21-game]

```java
private static final double MAX_ERROR = 1e-5;

public double new21Game(int n, int k, int maxPts) {
    // k <= score <= n
    if (k == 0 || k + maxPts <= n) {
        return 1d;
    }

    // dp[i]: the probability to get score i
    double[] dp = new double[n + 1];
    dp[0] = 1;

    // sum is the sum of dp[j] where j + maxPts >= i
    double sum = 1, p = 0;
    for (int i = 1; i <= n; i++) {
        // uniform distribution
        dp[i] = sum / maxPts;

        if (i < k) {
            sum += dp[i];
        } else {
            p += dp[i];
        }

        // sliding window size is maxPts
        if (i - maxPts >= 0) {
            sum -= dp[i - maxPts];
        }
    }
    return p;
}
```

[guess-the-word]: https://leetcode.com/problems/guess-the-word/
[new-21-game]: https://leetcode.com/problems/new-21-game/
