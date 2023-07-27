---
title:  "Java Conversions"
category: java
tags: java
---
# int[] -> Integer[]
```java
int[] a = {0, 1, 2};
Integer[] b = Arrays.stream(a).boxed().toArray(Integer[]::new);
```

# Integer[] -> int[]
```java
Integer[] a = {0, 1, 2};
int[] b = Arrays.stream(a).mapToInt(Integer::intValue).toArray();
b = Arrays.stream(a).mapToInt(i -> i).toArray();
```

# List\<Integer\> -> int[]
```java
int[] b = list.stream().mapToInt(Integer::valueOf).toArray();
b = list.stream().mapToInt(i -> i).toArray();
```

# int[] -> List\<Integer\>
```java
int[] a = {0, 1, 2};
List<Integer> b = Arrays.stream(a).boxed().collect(Collectors.toList());
```

# Integer[] -> Set\<Integer\>
```java
Integer[] a = {0, 1, 2};
Set<Integer> b = Arrays.stream(a).collect(Collectors.toSet());
```

# int[] -> Iterable\<Integer\>
```java
int[] a = {0, 1, 2};
Iterable<Integer> b = IntStream.of(a).boxed().iterator();
```

# int[] -> double[]
```java
int[] a = {0, 1, 2};
double[] b = IntStream.of(a).mapToDouble(i -> i).toArray();
```

# List -> Array:
```java
List<String> list = new ArrayList<>();
String[] array = list.toArray(new String[0])
```

# Array -> Stream
[public static \<T\> Stream\<T\> stream(T\[\] array, int startInclusive, int endExclusive)](https://docs.oracle.com/javase/8/docs/api/java/util/Arrays.html#stream-T:A-int-int-)

# char + int
```java
char c = (char)('a' + i);
```

# char number -> int
```java
char c = '9';
int a = c - '0';
```

# String -> int

[public static int parseInt(String s, int radix) throws NumberFormatException](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/Integer.html#parseInt(java.lang.String,int))
