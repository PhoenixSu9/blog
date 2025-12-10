---
title: Zookeeper 观察者模式架构分析
date: 2018-04-23 16:47:38
tags: [zookeeper, 分布式, 观察者模式, 架构设计]
categories: [大数据技术, Zookeeper]
description: Zookeeper的设计概念和实际应用。
---

# Zookeeper 从观察者模式角度的理解

## 核心概念

Zookeeper 是一个基于观察者模式设计的分布式服务管理框架，它负责存储和管理大家都关心的数据，然后接受观察者的注册，一旦这些数据的状态发生变化，Zookeeper 就将负责通知已经在 Zookeeper 上注册的那些观察者做出相应的反应。

## 观察者模式在 Zookeeper 中的体现

### 1. 架构组件

![Zookeeper观察者模式架构图](images/zookeeper/arti/a1.png)


### 2. 观察者模式的工作流程
![观察者模式的工作流程](images/zookeeper/arti/a2.png)


### 3. 核心特性

#### 主题 (Subject) - Zookeeper 服务器
- **数据存储**: 维护 ZNode 数据树
- **观察者管理**: 维护每个 ZNode 的观察者列表
- **状态变更通知**: 当 ZNode 数据发生变化时，通知所有注册的观察者

#### 观察者 (Observer) - 客户端应用
- **注册 Watch**: 向 Zookeeper 注册对特定 ZNode 的监听
- **接收通知**: 接收来自 Zookeeper 的变更通知
- **响应处理**: 根据通知执行相应的业务逻辑

## 实际应用场景

### 1. 配置管理
```java
// 伪代码示例
public class ConfigManager implements Watcher {
    private ZooKeeper zk;
    
    public void watchConfig(String configPath) {
        // 注册观察者
        zk.getData(configPath, this, null);
    }
    
    @Override
    public void process(WatchedEvent event) {
        if (event.getType() == Event.EventType.NodeDataChanged) {
            // 配置发生变化，重新加载
            reloadConfig(event.getPath());
            // 重新注册观察者（一次性触发）
            zk.getData(event.getPath(), this, null);
        }
    }
}
```

### 2. 服务发现
```java
public class ServiceDiscovery implements Watcher {
    private ZooKeeper zk;
    private String servicePath = "/services";
    
    public void watchServices() {
        // 监听服务节点变化
        zk.getChildren(servicePath, this);
    }
    
    @Override
    public void process(WatchedEvent event) {
        if (event.getType() == Event.EventType.NodeChildrenChanged) {
            // 服务列表发生变化
            updateServiceList();
            // 重新注册观察者
            zk.getChildren(servicePath, this);
        }
    }
}
```

## 架构优势

### 1. 解耦合
- 数据提供者（Zookeeper）和数据消费者（客户端）解耦
- 客户端不需要主动轮询，减少网络开销

### 2. 实时性
- 数据变更时立即通知所有观察者
- 支持多种事件类型（数据变更、子节点变更等）

### 3. 可扩展性
- 支持多个观察者同时监听同一个 ZNode
- 支持层次化的监听（父节点、子节点）

## 注意事项

### 1. 一次性触发
- Zookeeper 的 Watch 是一次性的，触发后需要重新注册
- 避免在事件处理中遗漏重新注册

### 2. 顺序保证
- 同一个客户端的事件通知是有序的
- 不同客户端之间的事件顺序不保证

### 3. 性能考虑
- 大量的 Watch 注册会影响性能
- 合理设计监听粒度，避免过度监听

## 集群架构与特性

### 集群组成与容错
- **主从架构**：由1个Leader和多个Follower组成
- **容错机制**：采用过半存活原则，集群需要半数以上节点存活才能正常工作
- **部署建议**：推荐部署奇数台服务器（如3、5、7台）以确保选举稳定性

### 数据一致性保证
- **全局一致性**：所有Server维护相同数据副本，客户端连接任意节点获取一致数据
- **顺序性保证**：
  - 同一客户端的更新请求按发送顺序执行
  - 不同客户端间的顺序不保证
- **原子性操作**：每次数据更新要么完全成功，要么完全失败

### 实时性特点
客户端能在可接受时间范围内读取到最新数据

## 核心数据结构

### ZNode 设计
- **树形结构**：类似Unix文件系统的层次化结构
- **节点特性**：
  - 每个ZNode默认存储上限1MB
  - 通过完整路径唯一标识
  - 支持临时节点和持久节点两种类型

## 典型应用场景

### 1. 统一命名服务
- **解决痛点**：分布式环境下服务标识难题
- **实现方式**：将IP等难记标识映射为易记名称
- **应用示例**：域名解析服务

### 2. 统一配置管理
- **核心价值**：实现分布式配置集中管理和动态更新
- **工作流程**：
  1. 将配置写入指定ZNode
  2. 客户端注册Watcher监听配置节点
  3. 配置变更时自动通知所有监听客户端

### 3. 集群状态管理
- **监控需求**：实时掌握集群节点状态
- **实现方案**：
  1. 节点信息注册到ZNode
  2. 监听节点状态变化
  3. 根据状态变化动态调整集群

## 动态感知设计

![服务器状态转换图](images/zookeeper/arti/zookeeper服务器上下线状态转换图.png)

## 非常重要的Zookeeper的选举机制过程图

![选举机制](images/zookeeper/arti/zookeeper选举机制.png)