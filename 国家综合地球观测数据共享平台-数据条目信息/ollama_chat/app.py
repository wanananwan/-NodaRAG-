from flask import Flask, request, jsonify
from langchain.docstore.document import Document
# 使用原生chromadb而不是langchain_chroma
import chromadb
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import time
import json
import requests  # 用于API调用
from sentence_transformers import SentenceTransformer  # 用于嵌入
import torch
import numpy as np  # 用于向量运算

# ---- 配置区 ----
DB_DIR = "./chroma_db"
EMBED_MODEL = "BAAI/bge-large-zh"  # 使用bge-large-zh模型
TOP_K = 20  # 增加到20，提高召回率
OLLAMA_API = "http://localhost:11434/api"  # Ollama API地址

print("正在加载嵌入模型和向量库...")
start_time = time.time()

# 加载SentenceTransformer模型
embedding_model = SentenceTransformer(EMBED_MODEL)

# 检查是否有GPU
if torch.cuda.is_available():
    embedding_model = embedding_model.to(torch.device("cuda"))
    print("成功将模型加载到GPU")
else:
    print("未检测到GPU，使用CPU运行")

# 创建ChromaDB客户端并获取集合
chroma_client = chromadb.PersistentClient(path=DB_DIR)
try:
    collection = chroma_client.get_collection("earth_observation_data")
    print(f"已加载集合，共有 {collection.count()} 条文档")
except Exception as e:
    print(f"加载集合失败: {str(e)}")
    print("请先运行 build_index.py 构建索引")
    collection = None

print(f"向量库加载完成，用时 {time.time() - start_time:.2f} 秒")

# 自定义检索器，封装chromadb
class ChromaRetriever:
    def __init__(self, collection, embedding_model, top_k=10):
        self.collection = collection
        self.embedding_model = embedding_model
        self.top_k = top_k
        
    def get_relevant_documents(self, query):
        # 计算查询的嵌入向量
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        
        # 确保转换为Python列表，而不是NumPy数组
        query_embedding_list = query_embedding.tolist()
        
        # 测试打印嵌入值
        print(f"生成的嵌入向量类型: {type(query_embedding_list)}, 长度: {len(query_embedding_list)}")
        
        # 使用ChromaDB进行查询
        try:
            # 基本查询 - 只使用嵌入
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=self.top_k
            )
        except Exception as e:
            print(f"向量检索错误: {str(e)}")
            # 回退方法 - 使用文本查询
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=self.top_k
                )
                print("已成功回退到文本查询")
            except Exception as e2:
                print(f"文本检索也失败: {str(e2)}")
                # 如果两种方法都失败，返回空列表
                return []
        
        # 转换为Document格式
        documents = []
        if results and 'ids' in results and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                doc_text = results['documents'][0][i]
                doc_metadata = results['metadatas'][0][i]
                documents.append(Document(page_content=doc_text, metadata=doc_metadata))
        
        return documents

# 获取所有文档以创建BM25检索器
print("正在创建检索器...")
retriever_start = time.time()

if collection:
    # 从ChromaDB获取所有文档
    all_docs = collection.get(include=["documents", "metadatas"])
    
    documents = [Document(page_content=content, metadata=meta)
                for content, meta in zip(all_docs['documents'], all_docs['metadatas'])]
    
    # 创建BM25检索器
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = TOP_K * 2  # 增加检索数量以提高召回率
    
    # 对于短查询特别设置更高的BM25相关性要求
    def get_bm25_docs(query, top_k=TOP_K * 2):
        # 获取所有匹配文档
        all_bm25_docs = bm25_retriever.get_relevant_documents(query)
        
        # 对于简短查询（如GF-1），确保必须包含关键词
        if len(query.split()) <= 2:
            # 查找包含完整关键词的文档
            exact_matches = []
            close_matches = []
            
            # 准备几种不同格式的查询
            query_lower = query.lower()  # 小写: gf-1
            query_upper = query.upper()  # 大写: GF-1
            query_no_dash = query.replace('-', '')  # 无破折号: GF1
            query_no_dash_lower = query_no_dash.lower()  # 小写无破折号: gf1
            query_camel = ''.join(word.capitalize() for word in query_lower.replace('-', ' ').split())  # 驼峰: Gf1
            query_space = query.replace('-', ' ')  # 空格替代破折号: GF 1
            
            # 可能的变体组合
            variants = [query, query_lower, query_upper, query_no_dash, 
                        query_no_dash_lower, query_camel, query_space]
            
            # 输出调试信息
            print(f"检索变体: {variants}")
            
            for doc in all_bm25_docs:
                doc_content_lower = doc.page_content.lower()
                
                # 检查是否有精确匹配
                if any(variant in doc.page_content for variant in variants):
                    exact_matches.append(doc)
                # 检查小写内容是否包含关键词变体
                elif any(variant.lower() in doc_content_lower for variant in variants):
                    close_matches.append(doc)
            
            # 如果找到精确匹配，优先返回这些文档
            if exact_matches:
                print(f"找到 {len(exact_matches)} 条精确匹配结果，包含关键词变体 '{query}'")
                return exact_matches + [doc for doc in all_bm25_docs if doc not in exact_matches][:top_k]
            
            # 如果找到近似匹配，也优先返回
            if close_matches:
                print(f"找到 {len(close_matches)} 条近似匹配结果，匹配关键词 '{query}' 的变体")
                return close_matches + [doc for doc in all_bm25_docs if doc not in close_matches][:top_k]
        
        # 默认返回所有BM25结果
        return all_bm25_docs[:top_k]
    
    # 创建ChromaDB检索器
    chroma_retriever = ChromaRetriever(collection, embedding_model, TOP_K * 2)  # 增加检索数量以提高召回率
    
    # 创建自定义组合检索器，而不是使用EnsembleRetriever
    class CustomEnsembleRetriever:
        def __init__(self, retrievers, weights=None):
            self.retrievers = retrievers
            self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)
            
        def invoke(self, query):
            # 使用BM25检索器 - 对精确匹配效果好，使用优化的BM25检索
            bm25_docs = get_bm25_docs(query)
            print(f"BM25检索到 {len(bm25_docs)} 条结果")
            
            # 使用Chroma检索器 - 对语义相似性效果好
            try:
                chroma_docs = self.retrievers[1].get_relevant_documents(query)
                print(f"向量检索到 {len(chroma_docs)} 条结果")
            except Exception as e:
                print(f"向量检索出错，将只使用BM25结果: {str(e)}")
                chroma_docs = []
            
            # 如果两种检索都没结果，直接返回空列表
            if not bm25_docs and not chroma_docs:
                print("警告: 两种检索方法都没有找到匹配文档")
                return []
            
            # 优化合并策略：提高相关性排序
            merged_docs = []
            seen_content = set()  # 用于记录已添加的内容，避免重复
            
            # 首先添加前3个BM25结果 (如果有)
            for i, doc in enumerate(bm25_docs[:3]):
                if doc.page_content not in seen_content:
                    merged_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            # 然后添加前3个向量结果
            for i, doc in enumerate(chroma_docs[:3]):
                if doc.page_content not in seen_content:
                    merged_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            # 然后交替添加剩余结果
            remaining_bm25 = [doc for doc in bm25_docs[3:] if doc.page_content not in seen_content]
            remaining_chroma = [doc for doc in chroma_docs[3:] if doc.page_content not in seen_content]
            
            for i in range(max(len(remaining_bm25), len(remaining_chroma))):
                if i < len(remaining_bm25):
                    if remaining_bm25[i].page_content not in seen_content:
                        merged_docs.append(remaining_bm25[i])
                        seen_content.add(remaining_bm25[i].page_content)
                if i < len(remaining_chroma):
                    if remaining_chroma[i].page_content not in seen_content:
                        merged_docs.append(remaining_chroma[i])
                        seen_content.add(remaining_chroma[i].page_content)
            
            return merged_docs

    # 创建自定义组合检索器
    ensemble_retriever = CustomEnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )
    
    print(f"检索器创建完成，用时 {time.time() - retriever_start:.2f} 秒")
else:
    print("警告: 未能创建检索器，请先运行 build_index.py")
    ensemble_retriever = None

# ---- Flask 配置 ----
app = Flask(__name__, 
            static_folder="templates",
            static_url_path="")

print("正在配置Ollama API...")
# 用requests替代ChatOllama
def query_ollama(prompt, context=None, model="deepseek-r1:14b"):
    """通过REST API调用Ollama"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }
    
    if context:
        payload["context"] = context
    
    try:
        print(f"发送请求到Ollama API，使用模型: {model}")
        print(f"请求大小: {len(prompt)} 字符")
        
        # 尝试检查Ollama服务是否在运行
        try:
            service_check = requests.get(f"{OLLAMA_API}/tags", timeout=5)
            if service_check.status_code != 200:
                return f"Ollama服务未正常响应，请检查Ollama服务是否已启动。错误码：{service_check.status_code}"
        except requests.exceptions.ConnectionError:
            return "Ollama服务连接失败，请确保已启动Ollama服务。可通过运行Ollama桌面应用程序或命令行启动服务。"
        except requests.exceptions.Timeout:
            return "Ollama服务响应超时，请检查服务状态。"
        
        # 尝试请求模型信息
        try:
            # 注意：直接检查模型是否在运行的ollama列表中
            model_info_response = requests.get(f"{OLLAMA_API}/tags", timeout=5)
            if model_info_response.status_code == 200:
                models_data = model_info_response.json()
                model_available = any(m.get('name', '').startswith(model) for m in models_data.get('models', []))
                if not model_available:
                    print(f"警告: 模型 {model} 不在已安装列表中，但将继续尝试使用")
                else:
                    print(f"模型信息: {model} 在已安装列表中")
            else:
                print(f"警告: 无法获取模型列表: {model_info_response.status_code}")
                # 即使无法获取模型列表，也继续尝试，而不是直接返回错误
        except Exception as e:
            print(f"警告: 检查模型信息时出错: {str(e)}")
            # 继续执行，不要因为模型信息检查失败而中断整个流程
        
        # 发起生成请求
        response = requests.post(f"{OLLAMA_API}/generate", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(f"Ollama API响应成功，响应大小: {len(result.get('response', ''))} 字符")
        return result["response"]
    except requests.exceptions.HTTPError as e:
        print(f"调用Ollama API HTTP错误: {str(e)}")
        print(f"错误码: {e.response.status_code}")
        print(f"响应内容: {e.response.text}")
        
        if e.response.status_code == 500:
            return f"Ollama服务内部错误，可能是以下原因：\n1. 模型 {model} 可能不支持所提供的参数\n2. 提示词太长\n3. 系统资源不足\n\n建议检查模型是否正确安装，或尝试重启Ollama服务。"
        
        # 其他错误情况
        return f"调用Ollama API发生错误: {str(e)}"
    except Exception as e:
        print(f"调用Ollama API错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"发生错误: {str(e)}"

print(f"Ollama API 配置完成")

@app.route("/chat", methods=["POST"])
def chat():
    start_time = time.time()
    
    try:
        # 获取用户输入
        q = request.json["prompt"]
        
        # 检查检索器是否可用
        if not ensemble_retriever:
            return jsonify({"response": "系统尚未准备好，请先运行 build_index.py 构建索引", "refs": []})
        
        # 直接使用原始查询进行检索
        retrieval_start = time.time()
        docs = ensemble_retriever.invoke(q)
        
        # 去重 - 改进的去重逻辑，基于文档ID
        seen_ids = set()
        docs_filtered = []
        print("\n==== 检索去重详情 ====")
        for i, doc in enumerate(docs):
            # 尝试从内容或元数据中提取文档ID
            doc_id = None
            
            # 从JSON内容中提取ID
            if doc.page_content.startswith('{"_id":'):
                try:
                    content_json = json.loads(doc.page_content)
                    if "_id" in content_json:
                        doc_id = content_json["_id"]
                except:
                    pass
            
            # 也可以从元数据中提取
            if not doc_id and "source_file" in doc.metadata and "row_index" in doc.metadata:
                # 使用文件名+行号作为唯一标识
                doc_id = f"{doc.metadata['source_file']}:{doc.metadata['row_index']}"
            
            # 如果无法提取ID，则使用内容哈希作为ID
            if not doc_id:
                doc_id = hash(doc.page_content)
            
            doc_source = doc.metadata.get('source_file', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
            
            # 只添加未见过的文档
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                docs_filtered.append(doc)
                print(f"✓ 保留文档 #{i+1}: ID={doc_id}, 来源={doc_source}")
                if len(docs_filtered) >= TOP_K:  # 限制最大文档数
                    print(f"已达到最大文档数 {TOP_K}，停止添加")
                    break
            else:
                print(f"✗ 丢弃重复文档 #{i+1}: ID={doc_id}, 来源={doc_source}")
        
        print(f"去重前: {len(docs)} 条文档, 去重后: {len(docs_filtered)} 条文档")
        print("==== 检索去重详情结束 ====\n")
        
        retrieval_time = time.time() - retrieval_start
        print(f"检索耗时: {retrieval_time:.2f}秒，找到{len(docs_filtered)}条结果")
        
        # 打印检索结果详情
        print("\n==== 检索结果详情 ====")
        for i, doc in enumerate(docs_filtered):
            print(f"\n[文档 {i+1}]")
            print(f"内容: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else f"内容: {doc.page_content}")
            print(f"元数据: {doc.metadata}")
        print("==== 检索结果详情结束 ====\n")
        
        # 限制上下文大小，只取前10个文档
        docs_filtered = docs_filtered[:10]  # 确保不会有太多文档
        
        # 限制每个文档的长度
        max_doc_length = 500
        limited_docs = []
        print("\n==== 文档内容处理详情 ====")
        for i, doc in enumerate(docs_filtered):
            # 尝试解析JSON内容并提取关键信息
            processed_content = ""
            try:
                content = doc.page_content
                original_length = len(content)
                if content.startswith("{") and content.endswith("}"):
                    data = json.loads(content)
                    
                    # 构建更加简洁的内容格式
                    important_fields = [ "type", "title", "keyword", "description", "url", 
                                       "timeInfo"]
                    
                    # 先添加最重要的字段
                    processed_lines = []
                    extracted_fields = []
                    for field in important_fields:
                        if field in data:
                            value = data[field]
                            # 对于长文本，进行截断
                            if isinstance(value, str) and len(value) > 200:
                                value = value[:200] + "..."
                            processed_lines.append(f"{field}: {value}")
                            extracted_fields.append(field)
                    
                    # 添加其他有用字段
                    for key, value in data.items():
                        if key not in important_fields:
                            # 对于URL相关字段，单独处理
                            if any(url_term in key.lower() for url_term in ["url", "uri", "download", "ftp"]):
                                processed_lines.append(f"{key}: {value}")
                                extracted_fields.append(key)
                    
                    processed_content = "\n".join(processed_lines)
                    print(f"文档 #{i+1}: 从JSON提取字段 {', '.join(extracted_fields)}")
                else:
                    # 非JSON内容直接使用
                    processed_content = content
                    print(f"文档 #{i+1}: 非JSON内容，保持原样")
            except Exception as e:
                # 解析失败，使用原始内容
                print(f"文档 #{i+1}: JSON解析错误: {str(e)}")
                processed_content = doc.page_content
            
            # 限制文档长度
            final_length = len(processed_content)
            if final_length > max_doc_length:
                processed_content = processed_content[:max_doc_length] + "..."
                print(f"文档 #{i+1}: 内容被截断，从 {final_length} 字符到 {max_doc_length} 字符")
            else:
                print(f"文档 #{i+1}: 内容长度 {final_length} 字符")
                
            # 创建新文档
            limited_doc = Document(page_content=processed_content, metadata=doc.metadata)
            limited_docs.append(limited_doc)
        
        print("==== 文档内容处理详情结束 ====\n")
        
        # 检查是否有限制后的文档
        if not limited_docs:
            print("警告: 没有找到任何相关文档")
            return jsonify({"response": f"抱歉，在现有数据库中没有找到与\"{q}\"相关的数据条目。请尝试使用其他关键词，如完整的卫星名称、数据类型或研究领域。", "refs": []})
        
        # 限制总文档数量为10条
        if len(limited_docs) > 10:
            print(f"限制文档数量从 {len(limited_docs)} 到 10 条")
            limited_docs = limited_docs[:10]
        
        # 合并文档内容，使用明确的分隔符
        ctx = "\n\n---文档分隔线---\n\n".join(d.page_content for d in limited_docs)
        print(f"上下文总长度: {len(ctx)} 字符")
        
        # 输出最终给LLM的上下文概要
        print("\n==== 最终传递给LLM的文档概要 ====")
        for i, doc in enumerate(limited_docs):
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"文档 #{i+1}: {len(doc.page_content)} 字符\n预览: {content_preview}\n")
        print("==== 文档概要结束 ====\n")
        
        print("\n==== 输入给LLM的上下文内容 ====\n" + ctx + "\n==== 上下文内容结束 ====\n")
        
        # 生成回答
        generation_start = time.time()
        prompt = f"""你是一名专业的数据检索助手，请基于以下参考资料回答用户问题。

分析步骤：
1. 先仔细审核所有参考资料，识别与用户查询"{q}"真正相关的数据条目
2. 过滤掉不相关或关联度低的数据条目
3. 对剩余的相关数据条目进行分析和整理
4. 使用Markdown格式输出最终答案

回答要求：
1. 只基于提供的参考资料中的信息回答
2. 使用Markdown格式，保持清晰整洁
3. 对于每个相关数据条目，列出：
   - 数据集的标题（使用Markdown标题格式）
   - 关键词
   - 描述信息
   - URL链接（如果有）
4. 对于简短查询（如只有一个缩写或术语），确保全面分析可能的相关性
5. 不要分析检索过程，不要用类似"根据参考资料"的引导语
6. 直接展示筛选后的相关信息，而不是所有检索结果

用户问题是：{q}

参考资料：
{ctx}

请直接回答："""

        # 使用API调用Ollama而不是ChatOllama
        answer = query_ollama(prompt)
        
        generation_time = time.time() - generation_start
        print(f"生成耗时: {generation_time:.2f}秒")
        
        # 检查回答是否有意义（应该包含一些与提供的相关）
        if answer and len(answer.strip()) > 10:
            refs = []
            # 提取引用的文档
            for i, doc in enumerate(limited_docs):
                if hasattr(doc, 'metadata') and 'source_file' in doc.metadata:
                    refs.append({
                        "text": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                        "file": doc.metadata['source_file'] if 'source_file' in doc.metadata else "未知"
                    })
            print("回答生成完成")
            return jsonify({"response": answer, "refs": refs})
        else:
            # 如果回答很短或没有意义，使用备用提示
            print("原始回答太短或无意义，使用备用提示...")
            backup_prompt = f"""你是一名专业的数据检索助手。请根据以下参考资料，分析关于"{q}"的相关数据条目。

执行以下步骤：
1. 仔细审核所有参考资料，找出与"{q}"最相关的数据条目
2. 过滤掉不相关或相关性较低的内容
3. 使用Markdown格式整理筛选后的内容

对于相关数据条目，请包含：
- 数据集标题（使用Markdown标题格式）
- 关键词
- 描述信息
- URL链接（如果有）

保持回答简洁、清晰，直接呈现筛选后的内容，而不是所有检索结果。
如果确实没有找到相关数据条目，请简明说明。

参考资料:
{ctx}"""
            
            answer = query_ollama(backup_prompt)
            
            return jsonify({
                "response": answer,
                "refs": []
            })
            
    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"response": f"处理请求时出错: {str(e)}", "refs": []})

@app.route("/")
def index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
