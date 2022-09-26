---
layout: post
title:  "System Design"
usemathjax: true
---

Redis:

Leader follower (master-replica) replication

mater sends a stream of commands to the replica to 

# Distributed Messaging System

## Apache Kafka

Topic

Apache Cassandra

## MongoDB

## HBase

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
Drawbacks:
* Domino effect
* Uneven server distribution

Solution: 
* add each server on the circle multiple times
* Jump Hash algorithm (Google)
* Proportional Hash (Yahoo!)
Possible problem: hot shard

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

* Scalable: partitioning
* Reliable: replication and checkpointing
* Fast: in-memory

Data enrichment
Embedded database (LinkedIn)

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

[Number of Distinct Roll Sequences][number-of-distinct-roll-sequences]

{% highlight java %}
{% endhighlight %}

[number-of-distinct-roll-sequences]: https://leetcode.com/problems/number-of-distinct-roll-sequences/
