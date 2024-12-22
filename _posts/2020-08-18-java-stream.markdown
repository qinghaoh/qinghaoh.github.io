---
title:  "Java Stream"
category: java
tags: java
---
# Ordering

https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/stream/package-summary.html

Streams may or may not have a defined encounter order. Whether or not a stream has an encounter order depends on the source and the intermediate operations. Certain stream sources (such as List or arrays) are intrinsically ordered, whereas others (such as HashSet) are not. Some intermediate operations, such as sorted(), may impose an encounter order on an otherwise unordered stream, and others may render an ordered stream unordered, such as BaseStream.unordered(). Further, some terminal operations may ignore encounter order, such as forEach().

[Stream\<T\> sorted(Comparator\<? super T\> comparator)](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/stream/Stream.html#sorted(java.util.Comparator)): For ordered streams, the sort is stable. For unordered streams, no stability guarantees are made.

# Summing numbers

## Stream.reduce()

[T reduce(T identity, BinaryOperator\<T\> accumulator)](https://docs.oracle.com/javase/8/docs/api/java/util/stream/Stream.html#reduce-T-java.util.function.BinaryOperator-)

```java
List<Integer> integers = Arrays.asList(0, 1, 2);
Integer sum = integers.stream()
  .reduce(0, (a, b) -> a + b);
```

```java
List<Integer> integers = Arrays.asList(0, 1, 2);
Integer sum = integers.stream()
  .reduce(0, Integer::sum);
```

## Stream.collect() 

```java
List<Integer> integers = Arrays.asList(0, 1, 2);
Integer sum = integers.stream()
  .collect(Collectors.summingInt(Integer::intValue));
```

## IntStream.sum()

```java
List<Integer> integers = Arrays.asList(0, 1, 2);
Integer sum = integers.stream()
  .mapToInt(Integer::intValue)
  .sum();
```

## Range

```java
List<Integer> integers = Arrays.asList(0, 1, 2);
Integer sum = integers.stream()
  .skip(1)
  .limit(1)
  .mapToInt(Integer::intValue)
  .sum();
```

# Frequency Map

```java
Map<Integer, Long> count = Arrays.stream(nums).boxed()
    .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
```

# Max

```java
int[] arr = {0, 1, 2};
Arrays.stream(arr).max().getAsInt()
```
