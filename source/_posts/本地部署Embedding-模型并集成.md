---
title: 本地部署Embedding 模型并集成
date: 2025-12-19 14:55:02
tags:
---

# 本地部署 Embedding 模型指南（CPU 版本）

本文档介绍如何在 CentOS 服务器上部署本地 Embedding 模型，适用于 CPU 环境。
现在网络上的Embedding模型 都是

---

## 推荐模型

针对你的环境（128GB 内存，仅 CPU），推荐以下模型：

| 模型 | 维度 | 大小 | 语言 | 推荐场景 |
|------|------|------|------|---------|
| **BAAI/bge-large-zh-v1.5** ⭐ | 1024 | ~1.3GB | 中文 | 中文知识库（推荐） |
| BAAI/bge-m3 | 1024 | ~2.2GB | 多语言 | 中英混合场景 |
| shibing624/text2vec-base-chinese | 768 | ~400MB | 中文 | 资源受限场景 |
| moka-ai/m3e-large | 768 | ~650MB | 中文 | 轻量级方案 |

**推荐使用 `bge-large-zh-v1.5`**：性能优秀，中文支持好，资源占用适中。

---

## 部署方案

### 方案一：使用 FastAPI + Sentence-Transformers（推荐）

这是最简单直接的方案，将模型封装为 REST API 服务。

#### 1. 环境准备

```bash
# 创建工作目录
mkdir -p /opt/embedding-service
cd /opt/embedding-service

# 安装 Python 3.10+（如果没有）
sudo yum install python3.10 python3.10-venv -y

# 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 安装依赖
pip install fastapi uvicorn sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu

https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 2. 下载模型

```bash
# 方式一：使用 HuggingFace（需要网络通畅）
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-large-zh-v1.5')"

# 方式二：使用魔搭（ModelScope，国内更快）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-large-zh-v1.5', cache_dir='./models')"
```

#### 3. 创建 API 服务

创建文件 `/opt/embedding-service/main.py`：

```python
"""
Embedding API 服务
兼容 OpenAI Embeddings API 格式
"""

import time
from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# 初始化模型（启动时加载）
print("正在加载 Embedding 模型...")
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
print("模型加载完成！")

app = FastAPI(title="Embedding API", version="1.0.0")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "bge-large-zh-v1.5"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """
    创建文本向量
    兼容 OpenAI API 格式
    """
    try:
        # 处理输入
        texts = request.input if isinstance(request.input, list) else [request.input]
        
        # 生成向量
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        # 构造响应
        data = [
            EmbeddingData(
                embedding=emb.tolist(),
                index=i
            )
            for i, emb in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": sum(len(t) for t in texts),
                "total_tokens": sum(len(t) for t in texts)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "model": "bge-large-zh-v1.5"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

#### 4. 启动服务

```bash
# 开发测试
cd /opt/embedding-service
source venv/bin/activate
python main.py

# 生产环境（使用 gunicorn）
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080
```

#### 5. 配置 Systemd 服务（可选）

创建文件 `/etc/systemd/system/embedding.service`：

```ini
[Unit]
Description=Embedding API Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/embedding-service
Environment="PATH=/opt/embedding-service/venv/bin"
ExecStart=/opt/embedding-service/venv/bin/gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable embedding
sudo systemctl start embedding
sudo systemctl status embedding
```

---

## 在 RAG 项目中使用

修改 `backend/core/embeddings.py`：

```python
from langchain_openai import OpenAIEmbeddings

def get_embedding_client():
    return OpenAIEmbeddings(
        api_key="not-needed",  # 本地服务不需要 API Key
        base_url="http://你的服务器IP:8080/v1",  # 指向本地服务
        model="bge-large-zh-v1.5"
    )
```
上面这种方式不行
本地模型需要修正为http形式的访问。
出现的问题:
```
获取查询向量失败: Error code: 422 - {'detail': [{'type': 'string_type', 'loc': ['body', 'input', 'str'], 'msg': 'Input should be a valid string', 'input': [[36827, 14276, 108, 74318, 9554, 29391, 77413, 82302, 39607, 1811]]}, {'type': 'string_type', 'loc': ['body', 'input', 'list[str]', 0], 'msg': 'Input should be a valid string', 'input': [36827, 14276, 108, 74318, 9554, 29391, 77413, 82302, 39607, 1811]}]} 
```
分析: LangChain 的 OpenAIEmbeddings 默认会先用 tiktoken 对文本进行分词（tokenize），然后发送 token ID 数组（如 [36827, 14276, ...]）而不是原始文本字符串给 API。本地 Embedding 服务期望接收 文本字符串，所以报错。

```
EMBEDDING_API_URL = "http://172.16.60.26:8884/v1/embeddings"
EMBEDDING_MODEL = "bge-large-zh-v1.5"


def get_embedding(text: str) -> List[float]:
    """
    获取文本的向量表示
    
    参数：
    - text: 输入文本
    
    返回：
    - 向量列表（1024 维）
    """
    try:
        response = requests.post(
            EMBEDDING_API_URL,
            json={
                "input": text,
                "model": EMBEDDING_MODEL
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        # OpenAI 格式的响应：data.data[0].embedding
        return data["data"][0]["embedding"]
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Embedding API 请求失败: {e}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Embedding API 响应格式错误: {e}")
```


修改 `backend/config/settings.py`：

```python
RAG_CONFIG = {
    'chunk_size': 500,
    'chunk_overlap': 50,
    'top_k': 3,
    'vector_dim': 1024,  # bge-large-zh-v1.5 输出 1024 维
}
```

---

## 测试验证

```bash
# 测试 API
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "model": "bge-large-zh-v1.5"}'

# 预期响应
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],  # 1024 维向量
      "index": 0
    }
  ],
  "model": "bge-large-zh-v1.5",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

---

## 性能优化

### CPU 优化

```bash
# 安装优化版 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 设置线程数（根据 CPU 核心数调整）
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### 批处理优化

修改 API 服务支持更大批量：

```python
# 在 main.py 中调整
embeddings = model.encode(
    texts, 
    normalize_embeddings=True,
    batch_size=32,  # 增加批处理大小
    show_progress_bar=False
)
```

---

## 常见问题

### Q: 模型下载很慢？

使用国内镜像源：

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用 ModelScope
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-large-zh-v1.5', cache_dir='./models')"
```

### Q: 内存占用过高？

尝试使用更小的模型：

```python
# 切换到更小的模型
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')  # ~400MB
```

### Q: CPU 推理太慢？

1. 使用 ONNX 优化：
```bash
pip install onnxruntime
# 导出并使用 ONNX 模型
```

2. 使用量化版本（int8）

---
