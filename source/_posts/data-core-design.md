---
title: data core design
date: 2025-07-31 20:26:20
tags: [大数据]
categories: [大数据技术]
description: 数据中台设计
---

以下是一个支持1MB/s实时数据流的综合数据中台设计方案，包含架构设计、硬件配置及性能分析，使用Mermaid绘制架构图：

![数据中台设计](images/数据中台/data_core.png)

## 一、核心架构设计

### 1. 数据摄入层
- **Kafka集群（3节点）**
  - 吞吐量：单节点>50MB/s，总吞吐150MB/s（满足1MB/s 10倍冗余）
  - Topic分区：按业务划分（e.g. device_data, user_behavior）
  - 保留策略：热数据3天（满足重处理需求）

### 2. 实时处理层
- **Flink引擎（分布式部署）**
  - 窗口计算：Tumbling Window 1min做微批聚合
  - 状态管理：RocksDB做Checkpoint存储
  - 输出：实时指标入Redis，原始数据入湖

### 3. 存储层
- **数据湖架构（Apache Iceberg）**
  ![数据湖架构](images/数据中台/2fa1f89c96fde8.png)
  - 存储格式：Parquet + ZSTD压缩（压缩比≈4:1）
  - 分区策略：按天分区+业务键哈希分桶

### 4. 治理层
- 元数据管理：Apache Atlas
- 数据血缘：自动追踪Kafka→Flink→Iceberg链路
- 质量监控：Great Expectations验证规则

### 5. 服务层
- 实时API：基于Redis的毫秒级响应
- 批处理API：Presto/Trino支持SQL查询
- 流批一体：同一SQL语法访问实时/历史数据

## 二、硬件配置方案（最小集群）

| 组件 | 配置规格 | 节点数 | 说明 |
|------|---------|-------|------|
| Kafka节点 | 8核32GB + 2TB NVMe SSD | 3 | 高吞吐需SSD保障IOPS |
| Flink计算节点 | 16核64GB + 1TB SSD | 3 | 独立部署避免资源争用 |
| Iceberg存储 | 4核16GB + 24TB HDD | 3 | 计算存储分离架构 |
| Redis集群 | 8核16GB + 512GB SSD | 3主3从 | 纯内存+持久化备份 |
| Trino查询节点 | 32核128GB + 2TB NVMe | 2 | 并行计算内存需求大 |

**容量计算：**
- 原始数据：1MB/s * 86400 = 86.4GB/日
- 压缩存储：Parquet+ZSTD ≈ 21.6GB/日
- 3年存储：21.6GB * 1095 ≈ 23.6TB （3副本=70.8TB）
- Redis容量：实时数据保留7天，约605MB

## 三、关键设计原理

1. **流批一体架构**
   - 优势：同一份Iceberg数据，Flink处理实时流，Spark分析历史数据
   - 数据一致性：通过Watermark机制确保

2. **弹性扩展策略**
   - Kafka分区动态扩容：单Topic从12分区→48分区
   - Flink自动扩缩容：基于K8s/Yarn的资源响应

3. **容错机制**
   ![容错机制](images/数据中台/error_back.png)
   - Kafka：副本因子=3（容忍2节点故障）
   - Flink：Checkpoint间隔=1分钟

4. **治理集成点**
   - 数据质量：在Flink处理管道嵌入规则引擎
   - 敏感数据：在存储层自动识别PII字段（身份证/手机号）

## 四、性能保障措施

1. **实时链路：Kafka→Flink→Redis < 500ms**
   - Kafka P99延迟：<10ms（SSD保障）
   - Flink反压检测：通过Metrics监控提前扩容

2. **查询性能：**
   ```sql
   -- Iceberg查询优化示例
   SELECT * FROM user_behavior
   WHERE event_date = '2023-10-01'
     AND user_id IN (SELECT user_id FROM vip_users)  -- 自动谓词下推
   ```
   - 分区裁剪减少90% I/O
   - 列式存储提升扫描效率

3. **压测指标：**

| 场景 | 要求 | 设计能力 |
|------|-----|---------|
| 数据摄入 | 1MB/s | 50MB/s |
| 实时查询QPS | 100 | 3000+ |
| 复杂分析时长 | 1亿条/5min | 2min(32核) |

## 五、高可用方案

![233735c16e94b](images/数据中台/233735c16e94b.png)

- Kafka Leader自动选举
- Flink JobManager主备模式
- Redis Cluster分片容错
- 多可用区部署（生产环境建议）

**成本优化提示：** 初期可采用4台物理机（128核/512GB/40TB SSD），通过Docker/K8s混部降低硬件成本，随业务增长逐步扩展独立集群。

该设计通过流批一体架构平衡实时性与分析需求，结合多层次存储策略降低成本，硬件配置预留5倍冗余以应对业务峰值，治理模块贯穿全链路确保数据可信度。