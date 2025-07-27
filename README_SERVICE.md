# 视频分析服务使用指南

本文档介绍如何使用环境变量配置视频分析服务以及解决常见问题。

## 快速开始

1. 复制环境配置文件：
   ```bash
   cp .env.local .env
   ```

2. 编辑 `.env` 文件中的配置项

3. 运行服务：
   ```bash
   python test_service.py /path/to/your/video.mp4
   ```

## 环境变量配置说明

### 客户端配置
- `CLIENT_TYPE`: 选择使用的客户端类型，可选 `ollama` 或 `openai_api`
- `OLLAMA_URL`: Ollama 服务地址，默认 `http://localhost:11434`
- `OLLAMA_MODEL`: Ollama 使用的模型名称
- `OPENAI_API_KEY`: OpenAI 兼容 API 的密钥
- `OPENAI_API_URL`: OpenAI 兼容 API 的地址
- `OPENAI_MODEL`: OpenAI 兼容 API 使用的模型名称

### 通用配置
- `TEMPERATURE`: 控制生成文本的随机性 (0.0-1.0)
- `OUTPUT_DIR`: 输出目录路径
- `KEEP_FRAMES`: 是否保留帧文件 (true/false)

### 音频处理配置
- `WHISPER_MODEL`: Whisper 模型大小 (tiny, base, small, medium, large) 或本地路径
- `AUDIO_DEVICE`: 音频处理设备 (cpu, cuda, auto)
- `AUDIO_LANGUAGE`: 音频语言代码，如 `en`, `zh` 等，留空则自动检测

### 帧处理配置
- `FRAMES_PER_MINUTE`: 每分钟提取的帧数
- `MAX_FRAMES`: 最大处理帧数

### 自定义提示
- `CUSTOM_PROMPT`: 自定义分析提示

## 常见问题及解决方案

### 1. SSL 连接错误或无法下载模型

错误信息示例：
```
WARNING: An error occured while synchronizing the model from the Hugging Face Hub
(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded
```

**解决方案：**
1. 预先下载模型到本地：
   ```bash
   # 使用 transformers CLI 下载模型
   pip install huggingface-hub
   huggingface-cli download Systran/faster-whisper-medium
   ```

2. 在 `.env` 文件中设置本地模型路径：
   ```bash
   WHISPER_MODEL=/path/to/downloaded/faster-whisper-medium
   ```

3. 设置网络代理（如果适用）：
   ```bash
   # 在 .env 文件中添加
   HTTP_PROXY=http://your.proxy:port
   HTTPS_PROXY=http://your.proxy:port
   ```

### 2. 数学计算警告

警告信息示例：
```
RuntimeWarning: divide by zero encountered in matmul
RuntimeWarning: overflow encountered in matmul
RuntimeWarning: invalid value encountered in matmul
```

**说明：**
这些警告通常不会影响转录结果，是由于音频特征提取过程中的数值计算问题导致的。

**解决方案：**
1. 降低音频处理的精度要求（通常不需要）
2. 更新 faster-whisper 版本：
   ```bash
   pip install --upgrade faster-whisper
   ```

### 3. CUDA 相关问题

如果遇到 CUDA 相关错误，可以尝试：
1. 强制使用 CPU 进行音频处理：
   ```bash
   # 在 .env 文件中设置
   AUDIO_DEVICE=cpu
   ```

2. 检查 CUDA 和 cuDNN 安装是否正确

## 集成到其他项目

要将视频分析功能集成到其他项目中，可以参考以下示例：

```python
from test_service import VideoAnalyzerService

# 创建服务实例
service = VideoAnalyzerService()

# 分析视频
results = service.analyze_video("/path/to/video.mp4")

# 处理结果
print(results["video_description"]["response"])
```

## 输出文件说明

分析完成后，会在输出目录生成以下文件：
- `analysis.json`: 完整的分析结果，包含元数据、转录、帧分析和视频描述
- `frames/`: 提取的帧图片（如果设置了 KEEP_FRAMES=true）
- `audio.wav`: 提取的音频文件（临时文件，分析完成后会自动删除）