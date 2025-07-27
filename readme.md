# Video Analysis using vision models like Llama3.2 Vision and OpenAI's Whisper Models

A video analysis tool that combines vision models like Llama's 11B vision model and Whisper to create a description by taking key frames, feeding them to the vision model to get details. It uses the details from each frame and the transcript, if available, to describe what's happening in the video. 

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [Ollama Setup](#ollama-setup)
  - [OpenAI-compatible API Setup](#openai-compatible-api-setup-optional)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Sample Output](#sample-output)
  - [Complete Usage Guide](docs/USAGES.md)
- [Design](#design)
  - [Detailed Design Documentation](docs/DESIGN.md)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Output](#output)
- [Uninstallation](#uninstallation)
- [License](#license)
- [Contributing](#contributing)

## Features
- 💻 Can run completely locally - no cloud services or API keys needed
- ☁️  Or, leverage any OpenAI API compatible LLM service (openrouter, openai, etc) for speed and scale
- 🎬 Intelligent key frame extraction from videos
- 🔊 High-quality audio transcription using OpenAI's Whisper
- 👁️ Frame analysis using Ollama and Llama3.2 11B Vision Model
- 📝 Natural language descriptions of video content
- 🔄 Automatic handling of poor quality audio
- 📊 Detailed JSON output of analysis results
- ⚙️ Highly configurable through command line arguments or config file

## Design
The system operates in three stages:

1. Frame Extraction & Audio Processing
   - Uses OpenCV to extract key frames
   - Processes audio using Whisper for transcription
   - Handles poor quality audio with confidence checks

2. Frame Analysis
   - Analyzes each frame using vision LLM
   - Each analysis includes context from previous frames
   - Maintains chronological progression
   - Uses frame_analysis.txt prompt template

3. Video Reconstruction
   - Combines frame analyses chronologically
   - Integrates audio transcript
   - Uses first frame to set the scene
   - Creates comprehensive video description

![Design](docs/design.png)

## Requirements

### System Requirements
- Python 3.11 or higher
- FFmpeg (required for audio processing)
- When running LLMs locally (not necessary when using openrouter)
  - At least 16GB RAM (32GB recommended)
  - GPU at least 12GB of VRAM or Apple M Series with at least 32GB

### Installation

1. Clone the repository:
```bash
git clone https://github.com/byjlw/video-analyzer.git
cd video-analyzer
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
pip install .  # For regular installation
# OR
pip install -e .  # For development installation
```

4. Install FFmpeg:
- Ubuntu/Debian:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg
  ```
- macOS:
  ```bash
  brew install ffmpeg
  ```
- Windows:
  ```bash
  choco install ffmpeg
  ```

### Ollama Setup

1. Install Ollama following the instructions at [ollama.ai](https://ollama.ai)

2. Pull the default vision model:
```bash
ollama pull gemma3:4b
```

3. Start the Ollama service:
```bash
ollama serve
```

### OpenAI-compatible API Setup (Optional)

If you want to use OpenAI-compatible APIs (like OpenRouter or OpenAI) instead of Ollama:

1. Get an API key from your provider:
   - [OpenRouter](https://openrouter.ai)
   - [OpenAI](https://platform.openai.com)

2. Configure via command line:
   ```bash
   # For OpenRouter
   video-analyzer video.mp4 --client openai_api --api-key your-key --api-url https://openrouter.ai/api/v1 --model gpt-4o

   # For OpenAI
   video-analyzer video.mp4 --client openai_api --api-key your-key --api-url https://api.openai.com/v1 --model gpt-4o
   ```

   Or add to config/config.json:
   ```json
   {
     "clients": {
       "default": "openai_api",
       "openai_api": {
         "api_key": "your-api-key",
         "api_url": "https://openrouter.ai/api/v1"  # or https://api.openai.com/v1
       }
     }
   }
   ```

Note: With OpenRouter, you can use llama 3.2 11b vision for free by adding :free to the model name

## Design
For detailed information about the project's design and implementation, including how to make changes, see [docs/DESIGN.md](docs/DESIGN.md).

## Usage

For detailed usage instructions and all available options, see [docs/USAGES.md](docs/USAGES.md).

### Quick Start

```bash
# Local analysis with Ollama (default)
video-analyzer video.mp4

# Cloud analysis with OpenRouter
video-analyzer video.mp4 \
    --client openai_api \
    --api-key your-key \
    --api-url https://openrouter.ai/api/v1 \
    --model meta-llama/llama-3.2-11b-vision-instruct:free

# Analysis with custom prompt
video-analyzer video.mp4 \
    --prompt "What activities are happening in this video?" \
    --whisper-model large
```

## Output

The tool generates a JSON file (`output\analysis.json`) containing:
- Metadata about the analysis
- Audio transcript (if available)
- Frame-by-frame analysis
- Final video description

## API服务

项目还提供基于FastAPI的REST API服务：

```bash
# 启动API服务
video-analyzer-api
```

API服务将在 `http://localhost:8000` 上运行，提供以下端点：

- `POST /upload` - 上传视频文件
- `POST /analyze/{session_id}` - 开始分析指定会话的视频
- `GET /status/{session_id}` - 获取分析状态
- `GET /results/{session_id}` - 下载分析结果
- `DELETE /session/{session_id}` - 删除会话
- `GET /health` - 健康检查

### API使用示例

```bash
# 上传视频
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@video.mp4"

# 开始分析（使用默认配置）
curl -X POST "http://localhost:8000/analyze/{session_id}" \
     -H "accept: application/json"

# 获取状态
curl -X GET "http://localhost:8000/status/{session_id}" \
     -H "accept: application/json"

# 下载结果
curl -X GET "http://localhost:8000/results/{session_id}" \
     -H "accept: application/json"
```

### Sample Output
```
The video begins with a person with long blonde hair, wearing a pink t-shirt and yellow shorts, standing in front of a black plastic tub or container on wheels. The ground appears to be covered in wood chips.\n\nAs the video progresses, the person remains facing away from the camera, looking down at something inside the tub. ........
```
full sample output in `docs/sample_analysis.json`
## Configuration

The tool uses a cascading configuration system with command line arguments taking highest priority, followed by user config (config/config.json), and finally the default config. See [docs/USAGES.md](docs/USAGES.md) for detailed configuration options.


## Uninstallation

To uninstall the package:
```bash
pip uninstall video-analyzer
```

## License

Apache License

## Contributing

We welcome contributions! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines on how to:
- Review the project design
- Propose changes through GitHub Discussions
- Submit pull requests
