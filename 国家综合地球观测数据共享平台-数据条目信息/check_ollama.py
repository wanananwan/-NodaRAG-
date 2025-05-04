import requests
import sys

OLLAMA_API = "http://localhost:11434/api"

def check_ollama_service():
    """检查Ollama服务是否可用"""
    try:
        print("正在检查Ollama服务状态...")
        response = requests.get(f"{OLLAMA_API}/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama服务正常运行")
            return True
        else:
            print(f"❌ Ollama服务状态异常，状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Ollama服务，请确保服务已启动")
        return False
    except requests.exceptions.Timeout:
        print("❌ 连接Ollama服务超时")
        return False
    except Exception as e:
        print(f"❌ 检查Ollama服务时发生错误: {str(e)}")
        return False

def check_model(model_name):
    """检查模型是否可用"""
    try:
        print(f"正在检查模型 {model_name} 是否可用...")
        response = requests.get(f"{OLLAMA_API}/show", params={"name": model_name}, timeout=10)
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ 模型 {model_name} 可用:")
            print(f"   - 大小: {model_info.get('size', '未知')}")
            print(f"   - 参数量: {model_info.get('parameter_size', '未知')}")
            print(f"   - 量化: {model_info.get('quantization_level', '未知')}")
            return True
        else:
            print(f"❌ 模型 {model_name} 不可用，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 检查模型 {model_name} 时发生错误: {str(e)}")
        return False

def list_models():
    """列出所有可用模型"""
    try:
        print("正在获取所有可用模型...")
        response = requests.get(f"{OLLAMA_API}/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print("✅ 已安装的模型:")
                for model in models:
                    print(f"   - {model.get('name')}")
            else:
                print("⚠️ 未找到任何已安装的模型")
        else:
            print(f"❌ 获取模型列表失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 获取模型列表时发生错误: {str(e)}")

def test_generation(model_name="deepseek-r1:14b"):
    """测试模型生成能力"""
    try:
        print(f"正在测试模型 {model_name} 的生成能力...")
        payload = {
            "model": model_name,
            "prompt": "用一句话介绍自己",
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        }
        response = requests.post(f"{OLLAMA_API}/generate", json=payload, timeout=20)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 生成测试成功:")
            print(f"   响应: {result.get('response', '无响应')}")
            return True
        else:
            print(f"❌ 生成测试失败，状态码: {response.status_code}")
            print(f"   响应: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 测试生成时发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    print("====== Ollama 状态检查工具 ======")
    
    # 检查服务状态
    service_ok = check_ollama_service()
    if not service_ok:
        print("\n❌ Ollama服务不可用，请确保已启动Ollama服务")
        print("   Windows: 运行Ollama桌面应用")
        print("   Linux/Mac: 运行 'ollama serve' 命令")
        sys.exit(1)
    
    # 列出所有模型
    print("\n----- 已安装的模型 -----")
    list_models()
    
    # 检查指定模型
    model_to_check = "deepseek-r1:14b"
    print(f"\n----- 检查模型: {model_to_check} -----")
    model_ok = check_model(model_to_check)
    
    # 测试生成
    if model_ok:
        print(f"\n----- 测试模型生成能力 -----")
        test_generation(model_to_check)
    
    print("\n====== 检查完成 ======") 