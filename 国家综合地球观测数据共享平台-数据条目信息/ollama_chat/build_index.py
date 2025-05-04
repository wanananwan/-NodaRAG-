import os, glob, json, re, time
import pandas as pd
from langchain.docstore.document import Document
# 直接使用chromadb而不是langchain_chroma
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import logging
import torch

# 设置日志级别为WARNING或更高，减少输出
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)

# ---- 配置区 ----
DATA_DIR    = "./docs"                   # 存放所有 .xls/.xlsx 文件的目录
DB_DIR      = "./chroma_db"              # 向量库存放目录
EMBED_MODEL = "BAAI/bge-large-zh"        # 嵌入模型名称
BATCH_SIZE  = 200                        # 每批处理的文档数

# 只处理这些关键字段
KEY_FIELDS = ['timeInfo', 'keyword', 'description', 'title']
MAIN_FIELDS = ['description', 'title']  # 需要分句的主要字段

# ---- 打印目录信息，方便调试 ----
print("当前工作目录:", os.getcwd())
print("DATA_DIR 绝对路径:", os.path.abspath(DATA_DIR))
files = glob.glob(os.path.join(DATA_DIR, "*.xls*"))
print("匹配到的 Excel 文件:", files)

def split_text(text):
    """分句函数：按中文句号、英文句号、分号、问号、感叹号、换行符等切分"""
    if not isinstance(text, str):
        return []
    # 按标点符号和换行符切分
    sentences = re.split(r'[。；;.!?！？\n]', text)
    # 去除空白字符，过滤空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def split_text_for_field(field_name, text, record=None):
    """根据字段特点选择不同的分句策略"""
    if not isinstance(text, str) or not text.strip():
        return []
        
    # 字段特定处理
    if field_name in ['description', 'title']:
        # 这些字段包含长文本描述，适合详细分句
        return split_long_text(text)
    elif field_name == 'timeInfo':
        # 这些字段是JSON结构，需要提取并重组
        return extract_json_values(text)
    else:
        # 其他字段采用基本分句
        return basic_split(text)

def basic_split(text):
    """基本分句：按句号等标点分句"""
    if not text or not isinstance(text, str):
        return []
    sentences = re.split(r'([。；;.!?！？\n])', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            sent = sentences[i] + sentences[i+1]
            if sent.strip():
                result.append(sent.strip())
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    return result or [text] if text.strip() else []

def split_long_text(text, max_length=150):
    """长文本分句：先按标点分，再按长度控制"""
    sentences = basic_split(text)
    result = []
    
    for sent in sentences:
        if len(sent) <= max_length:
            result.append(sent)
        else:
            # 先尝试在逗号等次要标点处分句
            subsents = re.split(r'([,，、])', sent)
            current = ""
            for i in range(0, len(subsents)-1, 2):
                if i+1 < len(subsents):
                    part = subsents[i] + subsents[i+1]
                    if len(current) + len(part) <= max_length:
                        current += part
                    else:
                        if current:
                            result.append(current)
                        current = part
            if current:
                result.append(current)
                
            # 如果仍有句子超长，强制切分
            for item in result[:]:
                if len(item) > max_length:
                    result.remove(item)
                    for j in range(0, len(item), max_length):
                        chunk = item[j:j+max_length]
                        if chunk:
                            result.append(chunk)
    
    return result

def extract_json_values(text):
    """从JSON结构中提取值并转为句子"""
    try:
        # 尝试解析JSON
        if text.startswith('{'):
            data = json.loads(text)
            result = []
            # 递归提取所有值
            extract_values(data, result)
            return [item for item in result if item and isinstance(item, str) and item.strip()]
        else:
            return basic_split(text)
    except:
        return basic_split(text)

def extract_values(obj, result):
    """递归提取JSON中的所有值"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                extract_values(value, result)
            elif value and isinstance(value, str) and value.strip():
                result.append(f"{key}: {value}")
    elif isinstance(obj, list):
        for item in obj:
            extract_values(item, result)
    elif obj and isinstance(obj, str) and obj.strip():
        result.append(obj)

# ---- 读取每个 Excel 的每个工作表，只处理关键字段 ----
docs = []
for path in tqdm(files, desc="处理文件"):
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)  # 读所有 sheet
        for sheet_name, df in sheets.items():
            df = df.fillna("")  # 缺失值填空串
            
            # 只保留关键字段
            available_fields = [f for f in KEY_FIELDS if f in df.columns]
            if not available_fields:
                print(f"警告: 文件 {path}, 工作表 {sheet_name} 没有找到任何关键字段")
                continue
                
            reduced_df = df[available_fields].copy()
            
            for idx, row in reduced_df.iterrows():
                # 只处理关键字段
                record = row.to_dict()
                
                # 处理每个字段，得到一个混合分句结果
                all_sentences = []
                field_sentences = {}
                
                # 1. 先处理主要文本字段
                for field in MAIN_FIELDS:
                    if field in record and record[field]:
                        field_text = record[field]
                        sentences = split_text_for_field(field, field_text, record)
                        if sentences:
                            field_sentences[field] = sentences
                            all_sentences.extend(sentences)
                
                # 如果没有任何分句结果，使用整行
                if not all_sentences:
                    content = json.dumps(record, ensure_ascii=False)
                    metadata = {
                        "source_file": os.path.basename(path),
                        "sheet": sheet_name,
                        "row_index": int(idx) + 2,
                        "sentence_index": 0,
                        "field": "all",
                        "is_whole_row": True
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
                    continue
                
                # 处理分句结果
                for field, sentences in field_sentences.items():
                    for sent_idx, sent in enumerate(sentences, 1):
                        # 创建新的记录，只替换当前字段
                        record_copy = record.copy()
                        # 更新字段内容为当前句子
                        if field in record_copy:
                            record_copy[field] = sent
                        
                        content = json.dumps(record_copy, ensure_ascii=False)
                        metadata = {
                            "source_file": os.path.basename(path),
                            "sheet": sheet_name,
                            "row_index": int(idx) + 2,
                            "sentence_index": sent_idx,
                            "field": field,
                            "is_whole_row": False
                        }
                        docs.append(Document(page_content=content, metadata=metadata))
    except Exception as e:
        print(f"处理文件 {path} 时出错: {str(e)}")
        continue

print(f"准备索引 {len(docs)} 条记录（包含分句）……")

# 手动分批处理文档
total_docs = len(docs)
start_time = time.time()

# 使用sentence-transformers加载模型
embedding_model = SentenceTransformer(EMBED_MODEL)

# GPU加速
if torch.cuda.is_available():
    embedding_model = embedding_model.to(torch.device("cuda"))
    print("成功将模型加载到GPU")
else:
    print("未检测到GPU，使用CPU运行")

# 检查DB_DIR是否已存在，如果存在则提示可能会覆盖
if os.path.exists(DB_DIR):
    print(f"警告: 向量库目录 {DB_DIR} 已存在，将会更新其中的内容")

# 创建ChromaDB客户端和集合
chroma_client = chromadb.PersistentClient(path=DB_DIR)
try:
    # 尝试获取已有集合
    collection = chroma_client.get_collection("earth_observation_data")
    print("已找到现有集合 'earth_observation_data'")
except:
    # 创建新集合
    collection = chroma_client.create_collection(
        name="earth_observation_data",
        embedding_function=None  # 我们会手动提供嵌入
    )
    print("创建新集合 'earth_observation_data'")

for batch_idx in range(0, total_docs, BATCH_SIZE):
    print(f"处理批次 {batch_idx // BATCH_SIZE + 1}/{(total_docs + BATCH_SIZE - 1) // BATCH_SIZE}：{batch_idx} 到 {min(batch_idx + BATCH_SIZE - 1, total_docs - 1)}")
    
    # 获取当前批次的文档
    batch_docs = docs[batch_idx:batch_idx + BATCH_SIZE]
    
    # 准备批量添加数据
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    # 处理每个文档
    for i, doc in enumerate(batch_docs):
        doc_id = f"doc_{batch_idx + i}"
        ids.append(doc_id)
        documents.append(doc.page_content)
        metadatas.append(doc.metadata)
        
    # 为所有文档计算嵌入向量
    batch_embeddings = embedding_model.encode([doc.page_content for doc in batch_docs], convert_to_tensor=False)
    embeddings.extend(batch_embeddings.tolist() if hasattr(batch_embeddings, 'tolist') else batch_embeddings)
    
    # 添加到ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    # 打印进度信息
    processed = min(batch_idx + BATCH_SIZE, total_docs)
    elapsed = time.time() - start_time
    docs_per_sec = processed / elapsed if elapsed > 0 else 0
    print(f"已处理 {processed}/{total_docs} 文档 ({processed/total_docs*100:.1f}%), "
          f"耗时 {elapsed:.1f}秒, 速度 {docs_per_sec:.1f} 文档/秒")
    
    # 打印一些批次文档的示例信息，方便调试
    if batch_idx == 0:
        for i, doc in enumerate(batch_docs[:2]):
            try:
                print(f"\n示例文档 {i+1}:")
                print(f"内容: {doc.page_content[:150]}...")
                print(f"元数据: {doc.metadata}")
            except Exception as e:
                print(f"打印示例文档时出错: {str(e)}")

print(f"\n索引构建完成！共处理了 {total_docs} 文档，耗时 {time.time() - start_time:.1f}秒")
print(f"向量库存储在: {os.path.abspath(DB_DIR)}")
print(f"集合信息: {collection.count()} 条文档")