---
title:  "Java Cheatsheet"
category: java
tags: java
---
# Array

Empty array:

```java
int[] arr1 = new int[0];
int[][] arr2 = new int[0][0];
```

# ArrayDeque

Null elements are prohibited.

# Arrays
* [public static \<T\> List\<T\> asList(T... a)](https://docs.oracle.com/javase/8/docs/api/java/util/Arrays.html#asList-T...-): fixed-size, mutable
* [public static \<T\> int compare(T\[\] a, int aFromIndex, int aToIndex, T\[\] b, int bFromIndex, int bToIndex, Comparator\<? super T\> cmp)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/util/Arrays.html#compare(T%5B%5D,int,int,T%5B%5D,int,int,java.util.Comparator))
* [public static \<T\> T\[\] copyOf(T\[\] original, int newLength)](https://docs.oracle.com/javase/8/docs/api/java/util/Arrays.html#copyOf-T:A-int-)
* [public static boolean equals(Object\[\] a, Objecti\[\] a2)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/util/Arrays.html#equals(java.lang.Object%5B%5D,java.lang.Object%5B%5D))
* [public static void fill(Object\[\] a, int fromIndex, int toIndex, Object val)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/util/Arrays.html#fill(java.lang.Object%5B%5D,int,int,java.lang.Object)): Assigns the specified Object *reference* to each element
* [static \<E\> List\<E\> of(E... elements)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/List.html#of%28E...%29): immutable
* [public static \<T\> void sort(T\[\] a, int fromIndex, int toIndex, Comparator\<? super T\> c)](https://docs.oracle.com/javase/8/docs/api/java/util/Arrays.html#sort-T:A-int-int-java.util.Comparator-): stable
* [public static String toString(Object\[\] a)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/util/Arrays.html#toString(java.lang.Object%5B%5D))

# Collection
* [void clear()](https://docs.oracle.com/javase/8/docs/api/java/util/Collection.html#clear--)
* [boolean contains(Object o)](https://docs.oracle.com/javase/8/docs/api/java/util/Collection.html#contains-java.lang.Object-)

# Collections
* [public static boolean disjoint(Collection\<?\> c1, Collection\<?\> c2)](https://docs.oracle.com/en/java/javase/15/docs/api/java.base/java/util/Collections.html#disjoint(java.util.Collection,java.util.Collection))
* [public static final \<T\> List\<T\> emptyList()](https://docs.oracle.com/javase/8/docs/api/java/util/Collections.html#emptyList--): immutable
* [public static \<T\> void fill(List\<? super T\> list, T obj)](https://docs.oracle.com/javase/8/docs/api/java/util/Collections.html#fill-java.util.List-T-)
* [public static int frequency(Collection\<?\> c, Object o)](https://docs.oracle.com/javase/8/docs/api/java/util/Collections.html#frequency-java.util.Collection-java.lang.Object-)
* [public static \<T\> T max(Collection\<? extends T\> coll, Comparator\<? super T\> comp)](https://docs.oracle.com/javase/8/docs/api/java/util/Collections.html#max-java.util.Collection-java.util.Comparator-)
* [public static \<T\> List\<T\> nCopies(int n, T o)](https://docs.oracle.com/javase/8/docs/api/java/util/Collections.html#nCopies-int-T-): immutable
* [public static void reverse(List\<?\> list)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/util/Collections.html#reverse(java.util.List))
* [public static \<T\> Comparator\<T\> reverseOrder()](https://docs.oracle.com/javase/8/docs/api/java/util/Collections.html#reverseOrder--)

```java
Integer[] arr = {1, 2, 3};

// Sorts arr[] in descending order
Arrays.sort(arr, Collections.reverseOrder());
```

* [public static \<T\> void sort(List\<T\> list, Comparator\<? super T\> c)](https://docs.oracle.com/javase/8/docs/api/java/util/Collections.html#sort-java.util.List-java.util.Comparator-): stable
* [public static void swap(List\<?\> list, int i, int j)](https://docs.oracle.com/javase/8/docs/api/java/util/Collections.html#swap-java.util.List-int-int-)

# Comparable
* [int compareTo(T o)](https://docs.oracle.com/javase/8/docs/api/java/lang/Comparable.html#compareTo-T-)

# Comparator
[Comparator](https://docs.oracle.com/javase/8/docs/api/java/util/Comparator.html)

Compare Strings by length in descending order, then by alphabetical order:
```java
Comparator.comparing(String::length)
    .reversed()
    .thenComparing(Comparator.<String>naturalOrder())
```

* [static \<T\> Comparator\<T\> comparingInt(ToIntFunction\<? super T\> keyExtractor)](https://docs.oracle.com/javase/8/docs/api/java/util/Comparator.html#comparingInt-java.util.function.ToIntFunction-)
* [static \<T extends Comparable\<? super T\>\> Comparator\<T\> naturalOrder()](https://docs.oracle.com/javase/8/docs/api/java/util/Comparator.html#naturalOrder--)
* [default Comparator\<T\> reversed()](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/util/Comparator.html#reversed())
* [static \<T extends Comparable\<? super T\>\> Comparator\<T\> reverseOrder()](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/util/Comparator.html#reverseOrder())

`Comparator.reverseOrder()` is preferable over `Collections.reverseOrder()` because it enforces the type argument to be a subtype of `Comparable`, and thus type safe.

# Decimal

https://docs.oracle.com/en/java/javase/18/docs/api/java.base/java/text/DecimalFormat.html

```java
DecimalFormat df = new DecimalFormat("0.00");
df.format("1.2345");  // "1.23"
```

# Deque
[Deque](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Deque.html)
Deques can also be used as LIFO (Last-In-First-Out) stacks. This interface should be used in preference to the legacy `Stack` class. When a deque is used as a stack, elements are pushed and popped from the beginning of the deque.

```java
Deque<Integer> stack = new ArrayDeque<>();
```

* [Iterator\<E\> descendingIterator()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Deque.html#descendingIterator())

# Enum
[Class Enum\<E extends Enum\<E\>\>](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/Enum.html)

* ordinal()
* toString()
* valueOf()

# Integer
* [MAX_VALUE](https://docs.oracle.com/javase/8/docs/api/java/lang/Integer.html#MAX_VALUE): 2^31-1 = 2147483647 ~= 2e9 (10 digits)
* [MIN_VALUE](https://docs.oracle.com/javase/8/docs/api/java/lang/Integer.html#MIN_VALUE): -2^31 = -2147483648 ~= -2e9 (10 digits)
* [public static int bitCount(int i)](https://docs.oracle.com/javase/8/docs/api/java/lang/Integer.html#bitCount-int-)
* [public int intValue()](https://docs.oracle.com/javase/8/docs/api/java/lang/Integer.html#intValue--)
* [public static int highestOneBit(int i)](https://docs.oracle.com/en/java/javase/15/docs/api/java.base/java/lang/Integer.html#highestOneBit(int))
* [public static int parseUnsignedInt(CharSequence s, int beginIndex, int endIndex, int radix) throws NumberFormatException](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/Integer.html#parseUnsignedInt(java.lang.CharSequence,int,int,int))
* [public static String toBinaryString(int i)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/Integer.html#toBinaryString(int))
* [public static String toString(int i,int radix)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/Integer.html#toString(int,int))

```java
MAX_VALUE + 1 == MIN_VALUE
```

# IntStream
* [int sum()](https://docs.oracle.com/javase/8/docs/api/java/util/stream/IntStream.html#sum--)

# Iterator
* [default void forEachRemaining(Consumer\<? super E\> action)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Iterator.html#forEachRemaining(java.util.function.Consumer))

# LinkedList
* [LinkedList](https://docs.oracle.com/javase/8/docs/api/java/util/LinkedList.html)
Doubly-linked list implementation of the `List` and `Deque` interfaces. Implements all optional list operations, and permits all elements (including `null`).

# List
* [void add(int index, E element)](https://docs.oracle.com/javase/8/docs/api/java/util/List.html#add-int-E-)
* [List\<E\> subList(int fromIndex, int toIndex)](https://docs.oracle.com/javase/8/docs/api/java/util/List.html#subList-int-int-)

# Math
* [public static double random()](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/Math.html#random()): it internally uses `Random.nextDouble()`

```java
// number of digits in n
int k = (int) (Math.log10(n) + 1);
```

# Object
* [protected Object clone() throws CloneNotSupportedException](https://docs.oracle.com/javase/8/docs/api/java/lang/Object.html#clone--)

# Priority Queue

The Iterator provided in method `iterator()` and the Spliterator provided in method `spliterator()` are ***not*** guaranteed to traverse the elements of the priority queue in any particular order. If you need ordered traversal, consider using Arrays.sort(pq.toArray()).

# Scanner

https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/util/Scanner.html

[Fraction Addition and Subtraction][fraction-addition-and-subtraction]

```java
public String fractionAddition(String expression) {
    Scanner sc = new Scanner(expression).useDelimiter("/|(?=[-+])");
    int num = 0, den = 1;
    while (sc.hasNext()) {
        int a = sc.nextInt(), b = sc.nextInt();
        num = num * b + a * den;
        den *= b;
        int g = gcd(num, den);
        num /= g;
        den /= g;
    }
    return num + "/" + den;
}

private int gcd(int a, int b) {
    a = Math.abs(a);
    while (b != 0) {
        int tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}
```

# Set
* [boolean containsAll(Collection\<?\> c)](https://docs.oracle.com/javase/8/docs/api/java/util/Set.html#containsAll-java.util.Collection-)
* [boolean retainAll(Collection\<?\> c)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/util/Set.html#retainAll(java.util.Collection))

# Stream
* [\<A\> A\[\] toArray(IntFunction\<A\[\]\> generator)](https://docs.oracle.com/javase/8/docs/api/java/util/stream/Stream.html#toArray-java.util.function.IntFunction-)

# String
* [public String(char\[\]  value, int offset, int count)](https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#String-char:A-int-int-)
* [public boolean contains(CharSequence s)](https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#contains-java.lang.CharSequence-)
* [public int indexOf(String str, int fromIndex)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/String.html#indexOf(java.lang.String,int))
* [public static String join(CharSequence delimiter, CharSequence... elements)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/String.html#join(java.lang.CharSequence,java.lang.CharSequence...))
* [public int lastIndexOf(int ch, int fromIndex)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/String.html#lastIndexOf(int,int)): searching *backward* starting at the specified index.
* [public String repeat(int count)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#repeat(int))
* [public String replace(char oldChar, char newChar)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/String.html#replace(char,char))
* [public String replaceAll(String regex, String replacement)](https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#replaceAll-java.lang.String-java.lang.String-)
* [public boolean startsWith(String prefix, int toffset)](https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#startsWith-java.lang.String-int-)
* [public String strip()](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/String.html#strip())
* [public String stripLeading()](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/String.html#stripLeading())
* substring(int beginIndex): `0 <= beginIndex <= s.length()`
* substring(int beginIndex, int endIndex): `0 <= beginIndex <= endIndex <= s.length()`

# StringBuilder
* [public StringBuilder delete(int start, int end)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/StringBuilder.html#delete(int,int))
* [public StringBuilder deleteCharAt(int index)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/StringBuilder.html#deleteCharAt(int))
* [public StringBuilder replace(int start, int end, String str)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/StringBuilder.html#replace(int,int,java.lang.String))
* [public StringBuilder reverse()](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/StringBuilder.html#reverse())
* [public void setLength​(int newLength)](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/StringBuilder.html#setLength(int))

# System
* [public static void arraycopy(Object src, int srcPos, Object dest, int destPos, int length)](https://docs.oracle.com/javase/8/docs/api/java/lang/System.html#arraycopy-java.lang.Object-int-java.lang.Object-int-int-)

# TreeSet
* [public NavigableSet\<E\> subSet(E fromElement, boolean fromInclusive, E toElement, boolean toInclusive)](https://docs.oracle.com/javase/8/docs/api/java/util/TreeSet.html#subSet-E-boolean-E-boolean-)
* [public SortedSet\<E\> subSet(E fromElement, E toElement)](https://docs.oracle.com/javase/8/docs/api/java/util/TreeSet.html#subSet-E-E-)

[fraction-addition-and-subtraction]: https://leetcode.com/problems/fraction-addition-and-subtraction/
