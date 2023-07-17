---
title:  "Java Conversions"
tags: java
---
# int[] -> Integer[]
{% highlight java %}
int[] a = {0, 1, 2};
Integer[] b = Arrays.stream(a).boxed().toArray(Integer[]::new);
{% endhighlight %}

# Integer[] -> int[]
{% highlight java %}
Integer[] a = {0, 1, 2};
int[] b = Arrays.stream(a).mapToInt(Integer::intValue).toArray();
b = Arrays.stream(a).mapToInt(i -> i).toArray();
{% endhighlight %}

# List\<Integer\> -> int[]
{% highlight java %}
int[] b = list.stream().mapToInt(Integer::valueOf).toArray();
b = list.stream().mapToInt(i -> i).toArray();
{% endhighlight %}

# int[] -> List\<Integer\>
{% highlight java %}
int[] a = {0, 1, 2};
List<Integer> b = Arrays.stream(a).boxed().collect(Collectors.toList());
{% endhighlight %}

# Integer[] -> Set\<Integer\>
{% highlight java %}
Integer[] a = {0, 1, 2};
Set<Integer> b = Arrays.stream(a).collect(Collectors.toSet());
{% endhighlight %}

# int[] -> Iterable\<Integer\>
{% highlight java %}
int[] a = {0, 1, 2};
Iterable<Integer> b = IntStream.of(a).boxed().iterator();
{% endhighlight %}

# int[] -> double[]
{% highlight java %}
int[] a = {0, 1, 2};
double[] b = IntStream.of(a).mapToDouble(i -> i).toArray();
{% endhighlight %}

# List -> Array:
{% highlight java %}
List<String> list = new ArrayList<>();
String[] array = list.toArray(new String[0])
{% endhighlight %}

# Array -> Stream
[public static \<T\> Stream\<T\> stream(T\[\] array, int startInclusive, int endExclusive)](https://docs.oracle.com/javase/8/docs/api/java/util/Arrays.html#stream-T:A-int-int-)

# char + int
{% highlight java %}
char c = (char)('a' + i);
{% endhighlight %}

# char number -> int
{% highlight java %}
char c = '9';
int a = c - '0';
{% endhighlight %}

# String -> int

[public static int parseInt(String s, int radix) throws NumberFormatException](https://docs.oracle.com/en/java/javase/14/docs/api/java.base/java/lang/Integer.html#parseInt(java.lang.String,int))
