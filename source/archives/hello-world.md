---
title: Hello World
---
Welcome to [Hexo](https://hexo.io/)! This is your very first post. Check [documentation](https://hexo.io/docs/) for more info. If you get any problems when using Hexo, you can find the answer in [troubleshooting](https://hexo.io/docs/troubleshooting.html) or you can ask me on [GitHub](https://github.com/hexojs/hexo/issues).

## Quick Start

### Create a new post

``` bash
$ hexo new "My New Post"
```

More info: [Writing](https://hexo.io/docs/writing.html)

### Run server

``` bash
$ hexo server
```

More info: [Server](https://hexo.io/docs/server.html)

### Generate static files

``` bash
$ hexo generate
```

More info: [Generating](https://hexo.io/docs/generating.html)

### Deploy to remote sites

``` bash
$ hexo deploy
```

More info: [Deployment](https://hexo.io/docs/one-command-deployment.html)


工作原理
分类页面 (/categories/) - 显示所有分类的概览，用户可以点击进入具体分类
标签页面 (/tags/) - 显示所有标签的概览，用户可以点击进入具体标签
自动生成的分类页面：
/categories/人工智能/ - 显示AI分类下的所有文章
/categories/人工智能/机器学习/ - 显示机器学习子分类的文章
/categories/人工智能/深度学习/ - 显示深度学习子分类的文章
/categories/大数据/ - 显示大数据分类下的所有文章
/categories/大数据/物联网/ - 显示物联网子分类的文章
用户体验流程
用户点击"分类" → 看到所有分类的概览页面
用户点击具体分类 → 看到该分类下的所有文章
用户点击子分类 → 看到子分类下的文章
优势
简洁明了：只有4个主要菜单项，不会让用户感到混乱
功能完整：用户可以通过分类页面访问所有分类和子分类
自动维护：当您添加新文章时，分类页面会自动更新
层级清晰：支持多层级分类结构


所以这里面新建文章就非常的清晰明了
# 创建新文章
npx hexo new post "文章标题"

# 编辑文章，设置分类
categories:
  - bigdata    # 主分类
  - iot        # 子分类
tags:
  - 深度学习
  - 图像识别
  - CNN
  - 计算机视觉

至于这些分类和标签定义位置在: _config.yml

如果说觉得首页比较冗杂，可以使用  
```
<div style="text-align: center; margin: 30px 0;">
  <a href="/bigdata/iot/index.html" style="display: inline-block; padding: 15px 30px; font-size: 18px; font-weight: bold; color: #fff; background-color: #0066cc; border-radius: 5px; text-decoration: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    点击查看完整解决方案 →
  </a>
</div>

```
然后将静态页面单独定义在一个专有的目录下面， 这样就能实现简洁的效果。


其他的命令
```
运行 hexo clean 清理缓存
运行 hexo generate 生成站点
运行 hexo server 启动服务器
访问 http://localhost:4000 查看效果
```