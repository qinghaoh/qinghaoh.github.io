---
title:  "Selection"
category: algorithm
tags: selection
---
[Selection algorithm](https://en.wikipedia.org/wiki/Selection_algorithm)

# Heap Sort

[Kth Largest Element in an Array][kth-largest-element-in-an-array]

```java
public int findKthLargest(int[] nums, int k) {
    // min heap
    Queue<Integer> pq = new PriorityQueue<>();
    for (int num : nums) {
        pq.offer(num);

        if (pq.size() > k) {
            pq.poll();    
        }
    }
    return pq.peek();
}
```

# Bucket Sort

[Top K Frequent Elements][top-k-frequent-elements]

```java
public int[] topKFrequent(int[] nums, int k) {
    Map<Integer, Integer> count = new HashMap<>();
    for (int num : nums) {
        count.put(num, count.getOrDefault(num, 0) + 1);
    }

    List<Integer>[] bucket = new List[nums.length + 1];
    for (var e : count.entrySet()) {
        if (bucket[e.getValue()] == null) {
            bucket[e.getValue()] = new ArrayList<>();
        }
        bucket[e.getValue()].add(e.getKey());
    }

    int[] result = new int[k];
    int index = 0;
    for (int i = bucket.length - 1; i >= 0; i--) {
        if (bucket[i] != null) {
            for (int num : bucket[i]) {
                result[index++] = num;
            }
            if (index == k) {
                break;
            }
        }
    }
    return result;
}
```

# Quickselect

[Quickselect](https://en.wikipedia.org/wiki/Quickselect)
[Partial sorting](https://en.wikipedia.org/wiki/Partial_sorting)

Time complexity: 
* Average: `O(n)`
* Worst: `O(n^2)`

[Kth Largest Element in an Array][kth-largest-element-in-an-array]

```java
public int findKthLargest(int[] nums, int k) {
    return quickSelect(nums, 0, nums.length - 1, k);
}

private int quickSelect(int[] nums, int low, int high, int k) {
    int p = partition(nums, low, high);

    // count of nums greater than or equal to nums[p]
    int count = high - p + 1;
    if (count == k) {
        return nums[p];
    }

    return count > k ? quickSelect(nums, p + 1, high, k) : quickSelect(nums, low, p - 1, k - count);
}

private int partition(int[] nums, int low, int high) {
    int pivot = nums[high];
    int i = low;
    for (int j = low; j < high; j++) {
        if (nums[j] < pivot) {
            swap(nums, i, j);
            i++;
        }
    }
    swap(nums, i, high);
    return i;
}
```

[Kth Largest Element in a Stream][kth-largest-element-in-a-stream]

Iterative:

[K Closest Points to Origin][k-closest-points-to-origin]

```java
public int[][] kClosest(int[][] points, int k) {
    int low = 0, high = points.length - 1;
    while (low <= high) {
        int mid = partition(points, low, high);
        if (mid == k) {
            break;
        }
        if (mid < k) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return Arrays.copyOfRange(points, 0, k);
}

private int partition(int[][] points, int low, int high) {
    int[] pivot = points[low];
    while (low < high) {
        while (low < high && compare(points[high], pivot) >= 0) {
            high--;
        }
        points[low] = points[high];
        while (low < high && compare(points[low], pivot) <= 0) {
            low++;
        }
        points[high] = points[low];
    }
    points[low] = pivot;
    return low;
}

private int compare(int[] p1, int[] p2) {
    return p1[0] * p1[0] + p1[1] * p1[1] - p2[0] * p2[0] - p2[1] * p2[1];
}
```

# Median of Medians

[Median of medians](https://en.wikipedia.org/wiki/Median_of_medians)

Time complexity:
* Best: `O(n)`
* Worst: `O(n)`

# Binary Search

[Kth Smallest Product of Two Sorted Arrays][kth-smallest-product-of-two-sorted-arrays]

```java
public long kthSmallestProduct(int[] nums1, int[] nums2, long k) {
    long low = (long)-10e10, high = (long)10e10;
    while (low < high) {
        long mid = low + (high - low) / 2;;
        if (condition(nums1, nums2, mid, k)) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

// finds the count of num * nums2 <= target
private boolean condition(int[] nums1, int[] nums2, long target, long k) {
    long count = 0;
    for (int num : nums1) {
        if (num == 0) {
            count += target >= 0 ? nums2.length : 0;
        } else {
            count += countPairs(nums2, num, target);
        }
    }
    return count >= k;
}

private int countPairs(int[] nums2, long num1, long target) {
    int low = 0, high = nums2.length;
    while (low < high) {
        int mid = (low + high) >>> 1;
        long product = num1 * nums2[mid];

        // if num1 >= 0, product is ascending
        // if num1 < 0, product is descending
        if (num1 < 0 ? product <= target : product > target) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return num1 < 0 ? nums2.length - low : low;
}
```

[k-closest-points-to-origin]: https://leetcode.com/problems/k-closest-points-to-origin/
[kth-largest-element-in-a-stream]: https://leetcode.com/problems/kth-largest-element-in-a-stream/
[kth-largest-element-in-an-array]: https://leetcode.com/problems/kth-largest-element-in-an-array/
[kth-smallest-product-of-two-sorted-arrays]: https://leetcode.com/problems/kth-smallest-product-of-two-sorted-arrays/
[top-k-frequent-elements]: https://leetcode.com/problems/top-k-frequent-elements/
