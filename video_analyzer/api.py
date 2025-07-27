#!/usr/bin/env python3
"""
FastAPI服务，提供视频分析的HTTP接口
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from video_analyzer.frame import VideoProcessor
from video_analyzer.prompt import PromptLoader
from video_analyzer.analyzer import VideoAnalyzer
from video_analyzer.audio_processor import AudioProcessor
from video_analyzer.clients.ollama import OllamaClient
from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Analyzer API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储分析状态
analysis_status: Dict[str, dict] = {}

class AnalysisRequest(BaseModel):
    client_type: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llava:latest"
    openai_api_key: Optional[str] = None
    openai_api_url: str = "https://openrouter.ai/api/v1"
    openai_model: str = "meta-llama/llama-3.2-11b-vision-instruct:free"
    temperature: float = 0.2
    output_dir: str = "output"
    keep_frames: bool = False
    whisper_model: str = "medium"
    audio_device: str = "cpu"
    audio_language: Optional[str] = None
    frames_per_minute: int = 30
    max_frames: int = 100
    custom_prompt: str = ""

def get_env_config(analysis_request: AnalysisRequest = None):
    """
    获取配置
    """
    if analysis_request:
        config = {
            # 客户端配置
            "client_type": analysis_request.client_type,
            "ollama_url": analysis_request.ollama_url,
            "ollama_model": analysis_request.ollama_model,
            "openai_api_key": analysis_request.openai_api_key,
            "openai_api_url": analysis_request.openai_api_url,
            "openai_model": analysis_request.openai_model,

            # 通用配置
            "temperature": analysis_request.temperature,
            "output_dir": analysis_request.output_dir,
            "keep_frames": analysis_request.keep_frames,

            # 音频处理配置
            "whisper_model": analysis_request.whisper_model,
            "audio_device": analysis_request.audio_device,
            "audio_language": analysis_request.audio_language,

            # 帧处理配置
            "frames_per_minute": analysis_request.frames_per_minute,
            "max_frames": analysis_request.max_frames,

            # 自定义提示
            "custom_prompt": analysis_request.custom_prompt,
        }
    else:
        config = {
            # 客户端配置
            "client_type": os.getenv("CLIENT_TYPE", "ollama"),
            "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "llava:latest"),
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_api_url": os.getenv("OPENAI_API_URL", "https://openrouter.ai/api/v1"),
            "openai_model": os.getenv("OPENAI_MODEL", "meta-llama/llama-3.2-11b-vision-instruct:free"),

            # 通用配置
            "temperature": float(os.getenv("TEMPERATURE", "0.2")),
            "output_dir": os.getenv("OUTPUT_DIR", "output"),
            "keep_frames": os.getenv("KEEP_FRAMES", "false").lower() == "true",

            # 音频处理配置
            "whisper_model": os.getenv("WHISPER_MODEL", "medium"),
            "audio_device": os.getenv("AUDIO_DEVICE", "cpu"),
            "audio_language": os.getenv("AUDIO_LANGUAGE", None),

            # 帧处理配置
            "frames_per_minute": int(os.getenv("FRAMES_PER_MINUTE", "30")),
            "max_frames": int(os.getenv("MAX_FRAMES", "100")),

            # 自定义提示
            "custom_prompt": os.getenv("CUSTOM_PROMPT", ""),
        }

    return config


def cleanup_files(output_dir: Path):
    """
    清理临时文件
    """
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            import shutil
            shutil.rmtree(frames_dir)
            logger.debug(f"Cleaned up frames directory: {frames_dir}")
        
        audio_file = output_dir / "audio.wav"
        if audio_file.exists():
            audio_file.unlink()
            logger.debug(f"Cleaned up audio file: {audio_file}")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@app.post("/analyze/{file_path:path}")
async def analyze_video(file_path: str, analysis_request: AnalysisRequest = None):
    """
    分析视频文件
    """
    # 检查文件是否存在
    video_path = Path(file_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {file_path}")
    
    # 设置初始状态
    analysis_status[file_path] = {
        "status": "processing",
        "results": None,
        "error": None
    }
    
    try:
        # 获取配置
        env_config = get_env_config(analysis_request)
        
        # 创建输出目录
        output_dir = Path(env_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化客户端
        client_type = env_config["client_type"]
        if client_type == "ollama":
            client = OllamaClient(env_config["ollama_url"])
            model = env_config["ollama_model"]
        elif client_type == "openai_api":
            if not env_config["openai_api_key"]:
                raise ValueError("OpenAI API key is required when using openai_api client")
            client = GenericOpenAIAPIClient(
                env_config["openai_api_key"],
                env_config["openai_api_url"]
            )
            model = env_config["openai_model"]
        else:
            raise ValueError(f"Unknown client type: {client_type}")
        
        logger.info(f"Using client: {client_type}, model: {model}")
        
        # 初始化其他组件
        prompt_loader = PromptLoader("video_analyzer/prompts", [
            {
                "name": "Frame Analysis",
                "path": "frame_analysis/frame_analysis.txt"
            },
            {
                "name": "Video Reconstruction",
                "path": "frame_analysis/describe.txt"
            }
        ])
        
        transcript = None
        frames = []
        frame_analyses = []
        video_description = None
        
        # 阶段1: 音频和帧处理
        logger.info("Stage 1: Processing audio and extracting frames...")
        
        # 初始化音频处理器
        audio_processor = AudioProcessor(
            language=env_config["audio_language"],
            model_size_or_path=env_config["whisper_model"],
            device=env_config["audio_device"]
        )
        
        # 提取音频
        logger.info("Extracting audio from video...")
        try:
            audio_path = audio_processor.extract_audio(video_path, output_dir)
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            audio_path = None
        
        # 转录音频
        if audio_path is None:
            logger.debug("No audio found in video - skipping transcription")
        else:
            logger.info("Transcribing audio...")
            transcript = audio_processor.transcribe(audio_path)
            if transcript is None:
                logger.warning("Could not generate reliable transcript. Proceeding with video analysis only.")
        
        # 提取帧
        logger.info(f"Extracting frames from video...")
        processor = VideoProcessor(
            video_path,
            output_dir / "frames",
            ""  # 不再传递模型参数
        )
        frames = processor.extract_keyframes(
            frames_per_minute=env_config["frames_per_minute"],
            max_frames=env_config["max_frames"]
        )
        
        # 阶段2: 帧分析
        logger.info("Stage 2: Analyzing frames...")
        analyzer = VideoAnalyzer(
            client,
            model,
            prompt_loader,
            env_config["temperature"],
            env_config["custom_prompt"]
        )
        
        for frame in frames:
            analysis = analyzer.analyze_frame(frame)
            frame_analyses.append(analysis)
            logger.info(f"Analyzed frame {frame.number}")
        
        # 阶段3: 视频重建
        logger.info("Stage 3: Reconstructing video description...")
        video_description = analyzer.reconstruct_video(frame_analyses, frames, transcript)
        
        # 保存结果
        results = {
            "metadata": {
                "client": client_type,
                "model": model,
                "whisper_model": env_config["whisper_model"],
                "frames_per_minute": env_config["frames_per_minute"],
                "frames_extracted": len(frames),
                "audio_language": transcript.language if transcript else None,
                "transcription_successful": transcript is not None
            },
            "transcript": {
                "text": transcript.text if transcript else None,
                "segments": transcript.segments if transcript else None
            } if transcript else None,
            "frame_analyses": frame_analyses,
            "video_description": video_description
        }
        
        results_file = output_dir / "analysis.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis complete. Results saved to {results_file}")
        
        # 更新状态
        analysis_status[file_path] = {
            "status": "completed",
            "results": results,
            "results_file": str(results_file),
            "error": None
        }
        
        # 清理临时文件
        if not env_config["keep_frames"]:
            cleanup_files(output_dir)
            
        return {"message": "Analysis completed", "results_file": str(results_file)}
        
    except Exception as e:
        logger.error(f"Error during video analysis: {e}")
        analysis_status[file_path] = {
            "status": "failed",
            "results": None,
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{file_path:path}")
async def get_status(file_path: str):
    """
    获取分析状态
    """
    if file_path not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_status[file_path]

@app.get("/results/{file_path:path}")
async def get_results(file_path: str):
    """
    获取分析结果
    """
    if file_path not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    status_info = analysis_status[file_path]
    if status_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Analysis not completed. Current status: {status_info['status']}")
    
    results_file = status_info.get("results_file")
    if not results_file or not Path(results_file).exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return status_info["results"]

@app.get("/")
async def root():
    """
    根路径
    """
    return {"message": "Video Analyzer API"}

@app.get("/health")
async def health_check():
    """
    健康检查
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)