---
title: Sourcegraph部署安装和使用完整指南
date: 2024-03-12 09:30:00
tags: [Sourcegraph, 代码搜索, 部署, Docker, 私有化部署, 开发工具]
categories: [开发工具, 代码管理]
description: 详细介绍Sourcegraph的部署安装方法，包括浏览器扩展、Docker单容器、Docker Compose集群部署，以及配置本地代码仓库的完整指南
---

## 什么是Sourcegraph？

Sourcegraph是一个强大的代码搜索和导航平台，让您能够：

- **智能代码搜索**：在海量代码库中快速定位函数、变量、类等
- **跨仓库跳转**：轻松查看函数定义、引用关系
- **团队协作**：支持代码审查、批量修改等功能

![Sourcegraph功能演示](images/sourcegraph/03dd91401436089ff0cc26f4698f7525.gif)

## 快速开始：浏览器扩展

### 最简单的安装方式

对于个人用户，最便捷的方式是安装浏览器扩展：

![应用商店安装](images/sourcegraph/s1.png)

安装后，您的GitHub项目页面会出现一个Sourcegraph按钮：

![GitHub上的Sourcegraph按钮](images/sourcegraph/s2.png)

点击按钮后，项目会在Sourcegraph平台上打开：

![Sourcegraph界面](images/sourcegraph/s3.png)

### 数据隐私考虑

如果您的项目涉及：
- 公司内部代码
- 私有仓库
- 敏感数据

建议部署私有化实例来确保**数据安全**。

## 私有化部署方案

### 方案一：Docker单容器部署（快速体验）

适合快速体验和小规模使用：

```bash
docker run --publish 7080:7080 \
  --publish 127.0.0.1:3370:3370 --rm \
  --volume ~/.sourcegraph/config:/etc/sourcegraph \
  --volume ~/.sourcegraph/data:/var/opt/sourcegraph \
  sourcegraph/server:6.5.2654
```

启动后访问：`http://hostname:7080/`

**注意**：此方式有功能限制，官方不推荐生产环境使用：

![官网关于单容器部署的说明](images/sourcegraph/s4.png)

### 方案二：Docker Compose集群部署（推荐）

适合生产环境和完整功能需求。

#### 环境准备

由于官方docker-compose配置存在兼容性问题，需要使用特定版本：

```bash
# 安装兼容版本的docker-compose
curl -L https://github.com/docker/compose/releases/download/1.25.0-rc4/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

#### 部署步骤

```bash
# 克隆项目
git clone https://github.com/sourcegraph/deploy-sourcegraph-docker.git
cd deploy-sourcegraph-docker

# 创建配置分支
git checkout master -b release
cd docker-compose/

# 复制配置文件
cp docker-compose.yaml docker-compose.override.yaml

# 启动服务
docker-compose -f docker-compose.override.yaml up
```

#### 常见问题解决

**问题1：服务依赖错误**
```bash
# 错误信息
services.sourcegraph-frontend-internal.depends_on.migrator.condition contains "service_completed_successfully"
```

**解决方案**：使用上述指定的docker-compose版本

**问题2：端口冲突**
```bash
# 修改docker-compose.override.yaml中的端口映射
ports:
  - "8080:7080"  # 改为其他端口
```

**问题3：资源不足**
如果遇到CPU或内存不足错误，需要：
- 增加系统资源配置
- 调整docker-compose.yaml中的资源限制

#### 成功启动

部署成功后，您将看到：

![部署成功界面](images/sourcegraph/se5.png)

服务默认占用80端口，可通过IP或域名直接访问。

## 初始配置

### 用户设置

首次访问需要创建管理员账户：

![登录页面](images/sourcegraph/login.png)

### 配置代码仓库

登录后配置Git仓库地址：

![代码仓库配置](images/sourcegraph/sl1.png)

支持多种代码托管平台：
- **GitHub**
- **GitLab**
- **Generic Git host**
- **Bitbucket**
- **其他Git服务**

#### 配置示例

![仓库配置示例](images/sourcegraph/r5.png)

成功添加5个仓库后的效果：

![仓库列表](images/sourcegraph/demo.png)

## 本地代码仓库配置

### 创建本地Git服务

![创建本地仓库](images/sourcegraph/start.png)

#### 安装src命令行工具

```bash
# 下载src CLI工具
curl -L https://sourcegraph.com/.api/src-cli/src_linux_amd64 -o /usr/local/bin/src
chmod +x /usr/local/bin/src
```

#### 启动本地Git服务

```bash
# 在代码目录启动服务
cd /home/user/projects
src serve-git

# 输出示例
serve-git: 2022/10/20 02:07:41 listening on http://[::]:3434
serve-git: 2022/10/20 02:07:41 serving git repositories from /home/user/projects
```

#### 在Sourcegraph中配置本地仓库

![本地仓库配置](images/sourcegraph/image.png)

配置完成后，您可以：
- 搜索本地代码
- 查看函数定义和引用
- 使用代码导航功能

## 高级功能探索

Sourcegraph提供了丰富的功能等待探索：

### 代码搜索技巧
- 使用正则表达式搜索
- 按文件类型过滤
- 跨仓库搜索

### 代码导航
- 函数定义跳转
- 引用查找
- 符号搜索

### 批量操作
- 批量代码修改
- 代码重构
- 依赖分析

## 版本信息

- **当前版本**：6.5.2654（2024年12月）
- **Docker镜像**：sourcegraph/server:6.5.2654
- **支持的Git版本**：2.x+
- **推荐系统**：Linux/macOS/Windows

## 故障排查

### 常见问题

1. **服务启动失败**
   - 检查端口占用情况
   - 确认Docker版本兼容性
   - 查看系统资源是否充足

2. **无法访问仓库**
   - 验证Git仓库URL
   - 检查网络连接
   - 确认认证信息

3. **性能问题**
   - 调整Docker资源限制
   - 优化索引配置
   - 考虑使用SSD存储

### 日志查看

```bash
# 查看容器日志
docker-compose logs -f sourcegraph-frontend

# 查看特定服务日志
docker-compose logs -f gitserver
```

## 总结

Sourcegraph是一个功能强大的代码搜索平台，通过本文的部署指南，您可以：

1. **快速体验**：使用浏览器扩展
2. **私有部署**：Docker单容器或集群部署
3. **本地集成**：配置本地代码仓库
4. **功能探索**：利用高级搜索和导航功能

无论是个人开发者还是企业团队，Sourcegraph都能显著提升代码阅读和理解效率。

---

**相关资源**：
- [Sourcegraph官方文档](https://docs.sourcegraph.com/)
- [GitHub项目地址](https://github.com/sourcegraph/sourcegraph)
- [社区支持论坛](https://community.sourcegraph.com/)

如果您在使用过程中发现更多有趣的功能，欢迎分享交流！让我们一起提升代码开发效率！


