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
        model_info_response = requests.get(f"{OLLAMA_API}/show", params={"name": model})
        if model_info_response.status_code != 200:
            print(f"警告: 无法获取模型 {model} 信息: {model_info_response.status_code}")
            return f"模型 {model} 可能不可用，请检查是否已安装该模型。可使用'ollama list'命令查看已安装的模型，或使用'ollama pull {model}'安装模型。"
        else:
            print(f"模型信息: {model} 可用")
        
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