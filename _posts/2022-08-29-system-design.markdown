---
layout: post
title:  "System Design"
usemathjax: true
---

## Redis

<img src="/assets/redis_logo.svg" width="150">

* [Documentation](https://redis.io/docs/)
* [Interview questions](https://gist.github.com/aershov24/16f4e369a93182de3f235a9a154a6b4a)

Redis (Remote Dictionary Server)

In-memory key-value data store. Stores cache data into physical storage if needed.

[Scaling](https://redis.io/docs/manual/scaling/): Redis Cluster

**Sharding**

hash value = CRC16(key) / #hash_slots (=16384)

NOT consistent hashing!

[Replication](https://redis.io/docs/manual/replication/): Master-replica

Asynchronous replication by default. Can be forced to synchronous by `WAIT` command when absolutely needed.

After *node timeout* has elapsed:
* Unresponsive master node is considered to be failing and can be replaced by one of its replicas
* If a master node cannot sense the majority of the other masters, it enters error state

**Node Communication**

TCP ports:
* Redis TCP port: node to clients
* Cluster bus port: node to node
 * Binary protocol: Redis Cluster Bus (Gossip)

Nodes:
* hold the data
* take the state of the cluster, including mapping keys to the right nodes
* auto-discover other nodes
* detect non-working nodes
* promote replica nodes to master (failover)

**Consistency**

NOT strong consistency.

Datatypes:
* string: C dynamic string library
* hash: namespace/ group for several key/value pairs (??) Hash table
* list: durable, atomic queues. Linked list
* set: Hash table
* sorted set: Skip list
* bitmap
* hyperlog
* zip list
* int set

Pub-sub model

Executes ultra-fast LUA scripts.

Atomicity

Item eviction policies

Blocking queues

Data persistence: snapshots (automatic, BGSAVE or at shutdown) (RDB?)
AOF (Append Only Files): optional

journaling??

Single threaded: an individual command is always atomic
Provide concurrency at the I/O level by I/O multiplexing + even loop. Atomicity is at no extra cost (doesn't require synchronization between threads)
CPU is usually not the bottleneck. IT's either memory or network bound.
Redis 4.0 more threaded: deleting objects in the background, blocking commands implemented via Redis modules

Pipelining (vs batching?) Redis commands

Multi/exec sequence ensures no other clients are executing commands in between.

Transactions: MULTI, EXEC, DISCARD, WATCH
Rollback is not supported

Sharding + Replication

Leader follower (master-replica) replication

mater sends a stream of commands to the replica to 

# Distributed Messaging System

## Apache Kafka

<img src="/assets/apache_kafka.png" width="150">

[Documentation](https://kafka.apache.org/documentation/)
[Interview questions](https://www.interviewbit.com/kafka-interview-questions/)

Records -> Topic
* Topics are separated into partitions
* Record log: offset
* A single topic can contain multiple partition logs (parallel processing)

Replication
* A replica is the redundant element of a topic partition
* Each partition contains one or more replicas across brokers

Brokers -> Cluster
* Managed by Apache ZooKeeper
* GB R/W /s
* Leader election

Consumers -> Consumer Group

* Fault-tolerant storage
* Pub/sub

## RabbitMQ

# Database

## Apache Cassandra

## MongoDB

## Apache HBase
Wide column. Time series data

## InfluxDB
Time series data

# Distributed File System

## Apache Hadoop File System (HDFS)

Availability
Scalability
Performance

CAP Theorem: Consistency, Availability and Parition Tolerance

Distributed cache:
* Dedicated cache cluster
* Co-located cache

MemCacheD

Shards: consistent hashing (cache client, server (Redis) or cache proxy (Twemproxy))
e.g. MurmurHash
Drawbacks:
* Domino effect
* Uneven server distribution

Solution: 
* add each server on the circle multiple times
* Jump Hash algorithm (Google)
* Proportional Hash (Yahoo!)
Possible problem: hot shard

Hot partition solution:
* include event time to the parition key
* Split hot parition into more partitions
* Dedicated parition for popular items

Configuration management tool
* [Chef](https://docs.chef.io/)
* Puppet

Configuration Service:
* Apache ZooKeeper

Data replication: availability
Protocols:
* Probabilistic: gossip, epidemic broadcast tree, bimodal multicase
 * Eventual consistency
* Consensus: 2 or 3 phase commit, Paxos, Raft, chain replication
 * Strong consistency

Leader-Follower replication:
* Leader: put, get
* Follower: get (deals with hot shards problem)

Leader election:
* Configuration service (Apache ZooKeeper, Redis Sentinel)
* Implement in cluster

Data replication is asynchronous, which may cause failures or inconsistency

Source of inconsistency:
* Asynchronous data replication
* Inconsistent server list

Expired items:
* Passive expire: remove when read them
* Active expire: a thread runs periodically to clean. If dataset is too big, test a several items use probablistic algorithms at every run

Firwall to protect cache server ports
Cache elements can be encrypted

Databases ca  handle millions of requests per second (source?)

MapReduce

# Distributed Coordination Service

* Synchronization
* Configuration maintenance
* Groups and naming

## Apache ZooKeeper

![Apache ZooKeeper](/assets/zookeeper_small.gif)

[Apache ZooKeeper](https://zookeeper.apache.org/): high performace, high availability, strictly ordered access.

* Shared hierarchical namespace (file system)
* Data registers - znodes (files and directories)
* Data are in-memory (high throughput, low latency)

Replication:
Ensemble: servers must all know about each other

* In-memory image: state
* Persistent storage: transaction logs and snapshots

ZooKeeper transactions (clients update data on znodes):

* Sequential consistency: transactions are stamped with a number that reflects the order
* Atomicity: Znode data reads/writes are atomic.
* Single System Image: a client's view of the service keeps the same regardless of the server that it connects to
* Reliability: an update persists from being applied until a client overwrites it
* Timeliness: the clients view of the system is guaranteed to be up-to-date within a certain time bound

Works best for read-dominant workload: (r/w ration = 10:1)

Znode stat structure includes:
* Version numbers for data changes
* ACL changes
* Timestamps

Ephemeral nodes are znodes that live only when the session that created the znode is alive

Client maintains a TCP connection to a znode. It can set a watch on a zonde that will be triggered when the znode changes.

Read requests are serviced from the local replica of each server database. Requests that change the state of the service, write requests, are processed by an agreement protocol.

All write requests from clients are forwarded to a single server, called the leader. The rest of the ZooKeeper servers, called followers, receive message proposals from the leader and agree upon message delivery. The messaging layer takes care of replacing leaders on failures and syncing followers with leaders.

custom atomic messaging protocol


Counter-based algorithms
* Count-min sketch
* Lossy counting
* Sapce saving
* Sticky sampling

Lambda Architecture
Nathan Marz, Apache Storm,
Jay Kreps, Apache Kafka

# Stream Processing Framework

## Apache Spark

## Apache Flink

Front-end
* Request valiation
* Authentication/Authorization
* TLS termination
* Server-side encryption
* Caching
* Rate limiting
* Request dispatchign
* Request deduplication
* Usage data collection

* Users/Customers
 * Who
 * How
* Scale
 * Requests per second
 * Traffic spikes
* Performace
 * Latency
* Cost
 * Development cost
 * Maintenance cost

SQL:
Normalization
Sharding, Cluster proxy (configuration service),
shard proxy
* Cache
* Monitor health
* Publish metris
* Terminate long queries 
Vitess (YouTube)

SQL
* ACID transactions
* Complex dynamic queries
* Data analytics
* Data warehousing

NoSQL
* Easy scaling for both writes and reads
* Highly available
* Tolerate network partitions

Some keywords:
* Scalable: partitioning
* Reliable: replication and checkpointing
* Fast: in-memory

Data enrichment
Embedded database (LinkedIn): RocksDB

Client
* blocking: create one thread for each new connection, easy to debug
* non-blocking I/O

Batching:
* increases throughput
* saves on cost
* request compression

Timeouts
* connection timeout: tens of ms
* request timeout: exponential backoff and jitter
 * Circuite Breaker: prevents repeat retries

Resource dispatching
* Bulkhead pattern: isolates elements of an application into pools.

Load balancers
* Round robin
* Least connections
* Least response time
* Hash-based

Service discovery:
* Server-side: load balancer
* Client-side
 * Service registry (e.g. Apache ZooKeeper, Netflix Eureka)
 * Gossip protocol

Replication
* Single leader: SQL scaling
* Multi leader: (TBD)
* Leaderless: Apache Cassandra

Binary formats:
* Thrift: tag
* Protocol buffers: tag
* Avro

Storage strategy: Data rollup
Hot/Cold storage
Data federation

Clients:
* Netty: Non-blocking I/O
* Netflix Hystrix
* Polly

Load balancer:
* NetScaler: hardware
* NGINX: software

Performace testing
* Load testing
* Stress testing: find break point
* Soak testing: find leaking resources
Apache JMeter to generate load

Monitoring:
* Latency
* Traffic
* Errors
* Saturation

Audit System:
* Weak: Canary
* Strong: Different path, Lambda Architecture

Queue message deletion:
* Offset (Apache Kafka)
* Mark as invisible so other cosumers won't see it. The consumer who retrieved the message deletes it explicitly, otherwise it becomes visible again (AWS SQS)

Message delivery:
* At most once
* At least once
* Exactly once (hard to achieve)

Message sharing
* Broadcasting (full mesh)
* Gossip protocol (< several thousands)
* Redis
* Coordination service

Service + Daemon

maxmemory: write commands starts to fail or evict keys

[Number of Distinct Roll Sequences][number-of-distinct-roll-sequences]

{% highlight java %}
{% endhighlight %}

[number-of-distinct-roll-sequences]: https://leetcode.com/problems/number-of-distinct-roll-sequences/
