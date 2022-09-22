---
layout: post
title:  "System Design"
usemathjax: true
---

Redis:

Leader follower (master-replica) replication

mater sends a stream of commands to the replica to 

Apache Kafka

Apache Cassandra

MongoDB

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
* Chef
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

Ordered:
All transactions are stamped with a number that reflects the order

Works best for read-dominant workload: (r/w ration = 10:1)

Znode stat structure includes:
* Version numbers for data changes
* ACL changes
* Timestamps

Znode data read/write are atomic.

Ephemeral nodes are znodes that live only when the session that created the znode is alive

Client maintains a TCP connection to a znode. It can set a watch on a zonde that will be triggered when the znode changes.




Count-min sketch


[Number of Distinct Roll Sequences][number-of-distinct-roll-sequences]

{% highlight java %}
{% endhighlight %}

[number-of-distinct-roll-sequences]: https://leetcode.com/problems/number-of-distinct-roll-sequences/
