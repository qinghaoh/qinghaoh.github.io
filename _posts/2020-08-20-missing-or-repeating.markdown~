---
title:  "Missing or Repeating"
tags: array
---
[Find All Numbers Disappeared in an Array][find-all-numbers-disappeared-in-an-array]

## Sort

```java
public List<Integer> findDisappearedNumbers(int[] nums) {
    for (int i = 0; i < nums.length; i++) {
        int index = nums[i] - 1;
        while (nums[index] != index + 1) {
            int curr = nums[index];
            nums[index] = index + 1;
            index = curr - 1;
        }
    }

    List<Integer> result = new ArrayList<>();
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] != i + 1) {
            result.add(i + 1);
        }
    }
    return result;
}
```

## Mark

```java
public List<Integer> findDisappearedNumbers(int[] nums) {
    for (int i = 0; i < nums.length; i++) {
        int index = Math.abs(nums[i]) - 1;
        if (nums[index] > 0) {
            nums[index] = -nums[index];
        }
    }

    List<Integer> result = new ArrayList<>();
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] > 0) {
            result.add(i + 1);
        }
    }
    return result;
}
```

```
[4,3,2,7,8,2,3,1]
[4,3,2,-7,8,2,3,1]
[4,3,-2,-7,8,2,3,1]
[4,-3,-2,-7,8,2,3,1]
[4,-3,-2,-7,8,2,-3,1]
[4,-3,-2,-7,8,2,-3,-1]
[4,-3,-2,-7,8,2,-3,-1]
[4,-3,-2,-7,8,2,-3,-1]
[-4,-3,-2,-7,8,2,-3,-1]
```

[find-all-numbers-disappeared-in-an-array]: https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/
