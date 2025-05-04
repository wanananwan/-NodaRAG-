# 地球观测数据RAG检索系统

这是一个基于Ollama的RAG（检索增强生成）系统，用于检索地球观测数据共享平台的数据条目。

## 系统架构

- **向量索引构建**：`build_index.py` 用于从Excel文件中提取数据，进行分句，生成嵌入，并存储到ChromaDB向量数据库
- **Web服务**：`app.py` 提供检索服务，使用混合检索策略(BM25+向量检索)，通过Ollama API调用大模型生成回答

## 依赖说明

系统主要依赖：
- **sentence-transformers**: 用于文本嵌入
- **chromadb**: 向量数据库
- **langchain**: 用于检索增强生成框架
- **Flask**: Web服务框架
- **Ollama**: 本地大模型服务

## 快速开始

### 1. 环境准备

推荐使用Conda创建虚拟环境：

```bash
# 创建新环境
conda create -n rag_new python=3.10
# 激活环境
conda activate rag_new
```

### 2. 安装依赖

使用提供的安装脚本安装所有依赖：

```bash
python install.py
```

或者手动安装：

```bash
pip install -r requirements.txt
```

### 3. 准备数据

将Excel数据文件放置在`docs`目录下：

```bash
mkdir -p docs
# 将Excel文件复制到docs目录
```

### 4. 确保Ollama服务已启动

Ollama需要单独安装和启动，详见[Ollama官方文档](https://github.com/ollama/ollama)

```bash
# 启动Ollama服务
ollama serve

# 在另一个终端拉取模型（例如：deepseek-r1:14b 或 qwen:7b）
ollama pull deepseek-r1:14b
```

### 5. 构建索引

```bash
python build_index.py
```

索引将被存储在`chroma_db`目录下。

### 6. 启动Web服务

```bash
python app.py
```

服务默认运行在 http://localhost:5000

## 技术说明

### 版本兼容性

系统使用以下组件版本以确保兼容性：

- chromadb==0.4.18
- langchain==0.1.0
- langchain-community==0.0.13
- langchain-core==0.1.10
- sentence-transformers==2.2.2
- numpy==1.24.3

为了解决依赖冲突，我们：
1. 直接使用chromadb API而不是langchain-chroma
2. 自定义实现与LangChain兼容的检索器接口
3. 通过requests库直接调用Ollama API

### 嵌入模型

默认使用`BAAI/bge-large-zh`作为嵌入模型，适合中文文本。

### 混合检索策略

系统使用BM25和向量检索的组合策略：
- BM25: 基于关键词匹配
- 向量检索: 基于语义相似度

## 常见问题

1. **依赖包冲突**：如遇包版本冲突，请使用`install.py`脚本分步安装
2. **GPU加速**：系统会自动检测并使用可用的GPU资源
3. **Ollama连接问题**：确保Ollama服务在`localhost:11434`运行
4. **索引构建失败**：检查docs目录中是否有有效的Excel文件 
