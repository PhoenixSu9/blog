---
title: docker 常用命令和效果
date: 2026-01-22 10:49:00
tags: [后端开发, Docker]
categories: [后端开发]
description: Docker 日常开发运维常用命令和功能。
---

```
# 最常用：显示最后 50 行并实时跟踪
docker logs -f --tail 50 data_transform_prod

# 查看最近 10 分钟的日志
docker logs -f --since 10m data_transform_prod

# 查看错误日志
docker logs -f --tail 200 data_transform_prod | grep -i error
```