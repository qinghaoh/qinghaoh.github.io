---
title:  "Corner Cases"
category: algorithm
---
[Reverse Integer][reverse-integer]

```java
public int reverse(int x) {
    int res = 0;
    while (x != 0) {
        int d = x % 10;
        int tmpRes = 10 * res + d;
        // avoid overflow
        if ((tmpRes - d) / 10 != res) {
            return 0;
        }
        res = tmpRes;
        x /= 10;
    }
    return res;
}
```

[Minimum Time Difference][minimum-time-difference]

```java
public int findMinDifference(List<String> timePoints) {
    int[] minutes = timePoints.stream()
        .mapToInt(this::convert)
        .sorted()
        .toArray();

    int d = 24 * 60;
    for (int i = 0; i < minutes.length - 1; i++) {
        d = Math.min(d, minutes[i + 1] - minutes[i]);
    }

    // comparison of circluar array element distances
    d = Math.min(d, minutes[0] + 24 * 60 - minutes[minutes.length - 1]);
    return d;
}

private int convert(String t) {
    String[] s = t.split(":");
    return Integer.valueOf(s[0]) * 60 + Integer.valueOf(s[1]);
}
```

[Most Visited Sector in a Circular Track][most-visited-sector-in-a-circular-track]

```java
public List<Integer> mostVisited(int n, int[] rounds) {
    List<Integer> result = new ArrayList<>();

    // start <= end
    if (rounds[0] <= rounds[rounds.length - 1]) {
        for (int i = rounds[0]; i <= rounds[rounds.length - 1]; i++) {
            result.add(i);
        }
    } else {
        // start > end
        for (int i = 1; i <= rounds[rounds.length - 1]; i++) {
            result.add(i);
        }   
        for (int i = rounds[0]; i <= n; i++) {
            result.add(i);
        } 
    }

    return result;
}
```

[Rabbits in Forest][rabbits-in-forest]

```java
public int numRabbits(int[] answers) {
    int[] c = new int[1000];
    int count = 0;
    for (int a : answers) {
        // If c[a] % (a + 1) == 0, there are c[a] / (a + 1) groups of (a + 1) rabbits
        // If c[a] % (a + 1) != 0, there are c[a] / (a + 1) + 1 groups of (a + 1) rabbits
        if (c[a]++ % (a + 1) == 0) {
            count += a + 1;
        }
    }

    return count;
}
```

[Heaters][heaters]

```java
public int findRadius(int[] houses, int[] heaters) {
    Arrays.sort(houses);
    Arrays.sort(heaters);

    int i = 0, radius = 0;
    for (int house : houses) {
        while (i < heaters.length - 1 && heaters[i] + heaters[i + 1] <= house * 2) {
            i++;
        }
        radius = Math.max(radius, Math.abs(heaters[i] - house));
    }
    return radius;
}
```

![Heaters](/assets/img/algorithm/heaters.png)

[Sum of All Odd Length Subarrays][sum-of-all-odd-length-subarrays]

```java
public int sumOddLengthSubarrays(int[] arr) {
    int sum = 0;
    for (int i = 0; i < arr.length; i++) {
        // number of subarrays containing A[i]:
        // left: i + 1
        // right: n - i
        sum += ((i + 1) * (n - i) + 1) / 2 * A[i];
    }
    return sum;
}
```

[Sum of Mutated Array Closest to Target][sum-of-mutated-array-closest-to-target]

```java
public int findBestValue(int[] arr, int target) {
    Arrays.sort(arr);

    // after the loop, value is in (arr[i - 1], arr[i]]
    // arr.length - i is the number of remaining elements in the array
    int i = 0;
    while (i < arr.length && target > arr[i] * (arr.length - i)) {
        target -= arr[i++];
    }

    // value > arr[arr.length - 1]
    // i.e. target > sum(arr)
    if (i == arr.length) {
        return arr[arr.length - 1];
    }

    // ceiling function
    int value = target / (arr.length - i);
    // 3 * 5 < 19 < 4 * 5
    // 3 * 5 <= 15 < 4 * 5
    if (target - value * (arr.length - i) > (value + 1) * (arr.length - i) - target) {
        value++;
    }

    return value;
}
```

[Swap for Longest Repeated Character Substring][swap-for-longest-repeated-character-substring]

```java
public int maxRepOpt1(String text) {
    // char : list of indexes
    Map<Character, List<Integer>> map = new HashMap<>();
    for (int i = 0; i < text.length(); i++) {
        map.computeIfAbsent(text.charAt(i), k -> new ArrayList<>()).add(i);
    }

    int result = 0;
    for (List<Integer> list : map.values()) {
        // count of chars in previous and current block
        int prev = 0, curr = 1, sum = 1;
        for (int i = 1; i < list.size(); i++) {
            if (list.get(i) == list.get(i - 1) + 1) {
                curr++;
            } else {
                // if previous block is more than 1 char away, clears it
                prev = list.get(i) == list.get(i - 1) + 2 ? curr : 0;
                curr = 1;
            }
            sum = Math.max(sum, curr + prev);
        }
        // if sum < list.size(), there are more of that char somewhere in the string 
        result = Math.max(result, sum + (sum < list.size() ? 1 : 0));
    }
    return result;
}
```

[Number of Steps to Reduce a Number in Binary Representation to One][number-of-steps-to-reduce-a-number-in-binary-representation-to-one]

```java
public int numSteps(String s) {
    int step = 0, carry = 0;
    for (int i = s.length() - 1; i > 0; i--) {
        step++;
        // additional step:
        // first encounter of '1'
        // or, encounter of '0' after the first '1' is encountered
        if (s.charAt(i) - '0' + carry == 1) {
            // carry is always 1 after the first '1' is encountered
            carry = 1;
            step++;
        }
    }
    return step + carry;
}
```

[Count Number of Teams][count-number-of-teams]

```java
public int numTeams(int[] rating) {
    int count = 0;
    // i is the middle soldier
    for (int i = 1; i < rating.length - 1; i++) {
        int[] less = new int[2], greater = new int[2];
        for (int j = 0; j < rating.length; j++) {
            // 0: left, 1: right
            int index = j > i ? 1 : 0;
            if (rating[i] < rating[j]) {
                less[index]++;
            }
            if (rating[i] > rating[j]) {
                greater[index]++;
            }
        }
        count += less[0] * greater[1] + greater[0] * less[1];
    }
    return count;
}
```

[Count of Matches in Tournaments][count-of-matches-in-tournament]

```java
public int numberOfMatches(int n) {
    // one champion; each of the other teams lost one game
    return n - 1;
}
```

[Similar RGB Color][similar-rgb-color]

```java
public String similarRGB(String color) {
    return "#" + getClosest(color.substring(1, 3)) + getClosest(color.substring(3, 5)) + getClosest(color.substring(5));
}

private String getClosest(String s) {
    int q = Integer.parseInt(s, 16);
    // #AB -> #CC
    // 16 * A + B = 16 * C + C = 17 * C
    // C = q / 17.0
    q = q / 17 + (q % 17 > 8 ? 1 : 0);
    return String.format("%02x", 17 * q);
}
```

[Number of Students Unable to Eat Lunch][number-of-students-unable-to-eat-lunch]

```java
public int countStudents(int[] students, int[] sandwiches) {
    int[] count = {0, 0};
    for (int s: students) {
        count[s]++;
    }

    int eaten = 0;
    while (eaten < sandwiches.length && count[sandwiches[eaten]] > 0) {
        count[sandwiches[eaten++]]--;
    }

    return students.length - eaten;
}
```

[Maximum Score From Removing Stones][maximum-score-from-removing-stones]

```java
public int maximumScore(int a, int b, int c) {
    // in the end, 3 0's or 2 0's
    return Math.min((a + b + c) / 2, a + b + c - Math.max(a, Math.max(b, c)));
}
```

[Nth Digit][nth-digit]

```java
public int findNthDigit(int n) {
    long count = 9;
    int len = 1;

    while (n > len * count) {
        n -= len * count;
        len++;
        count *= 10;
    }

    return Character.getNumericValue(Integer.toString((int)(count / 9) + (n - 1) / len).charAt((n - 1) % len));
}
```

[Convert Integer to the Sum of Two No-Zero Integers][convert-integer-to-the-sum-of-two-no-zero-integers]

```java
public int[] getNoZeroIntegers(int n) {
    int a = 0, b = 0, step = 1;

    // from LSB to MSB
    while (n > 0) {
        int d = n % 10;
        n /= 10;

        // views 0 as 10, 1 as 11 (with carry)
        // if n == 0, there was only one digit left,
        // so we can't assume carry,
        // and that's handled by the else statement
        if ((d == 0 || d == 1) && n > 0) {
            a += step * (1 + d);
            b += step * 9;
            n--;  // handles carry
        } else {
            a += step * 1;
            b += step * (d - 1);
        }
        step *= 10;
    }

    return new int[]{a, b};
}
```

[Remove Comments][remove-comments]

```java
public List<String> removeComments(String[] source) {
    List<String> list = new ArrayList<>();
    StringBuilder sb = new StringBuilder();     
    boolean isBlockComments = false;
    for (String s : source) {
        // processes each char
        for (int i = 0; i < s.length(); i++) {
            if (isBlockComments) {
                // "*/"
                if (s.charAt(i) == '*' && i < s.length() - 1 && s.charAt(i + 1) == '/') {
                    isBlockComments = false;
                    i++;
                }
            } else {
                // "//"
                if (s.charAt(i) == '/' && i < s.length() - 1 && s.charAt(i + 1) == '/') {
                    break;
                }

                // "/*"
                if (s.charAt(i) == '/' && i < s.length() - 1 && s.charAt(i + 1) == '*') {
                    isBlockComments = true;
                    i++;
                } else {
                    sb.append(s.charAt(i));
                }
            }
        }

        // to reach this point,
        // either it's in a line comment, or all chars in this line are processed
        // adds the chars in the String builder only if it's not in a block comment
        if (!isBlockComments && sb.length() > 0) {
            list.add(sb.toString());
            sb.setLength(0);
        }
    }
    return list;
}
```

[convert-integer-to-the-sum-of-two-no-zero-integers]: https://leetcode.com/problems/convert-integer-to-the-sum-of-two-no-zero-integers/
[count-number-of-teams]: https://leetcode.com/problems/count-number-of-teams/
[count-of-matches-in-tournament]: https://leetcode.com/problems/count-of-matches-in-tournament/
[heaters]: https://leetcode.com/problems/heaters/
[maximum-score-from-removing-stones]: https://leetcode.com/problems/maximum-score-from-removing-stones/
[minimum-time-difference]: https://leetcode.com/problems/minimum-time-difference/
[most-visited-sector-in-a-circular-track]: https://leetcode.com/problems/most-visited-sector-in-a-circular-track/
[nth-digit]: https://leetcode.com/problems/nth-digit/
[number-of-steps-to-reduce-a-number-in-binary-representation-to-one]: https://leetcode.com/problems/number-of-steps-to-reduce-a-number-in-binary-representation-to-one/
[number-of-students-unable-to-eat-lunch]: https://leetcode.com/problems/number-of-students-unable-to-eat-lunch/
[rabbits-in-forest]: https://leetcode.com/problems/rabbits-in-forest/
[remove-comments]: https://leetcode.com/problems/remove-comments/
[reverse-integer]: https://leetcode.com/problems/reverse-integer/
[similar-rgb-color]: https://leetcode.com/problems/similar-rgb-color/
[sum-of-all-odd-length-subarrays]: https://leetcode.com/problems/sum-of-all-odd-length-subarrays/
[sum-of-mutated-array-closest-to-target]: https://leetcode.com/problems/sum-of-mutated-array-closest-to-target/
[swap-for-longest-repeated-character-substring]: https://leetcode.com/problems/swap-for-longest-repeated-character-substring/
