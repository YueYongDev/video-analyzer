#!/usr/bin/env python3
"""
简化版测试脚本，用于演示如何使用环境变量配置视频分析服务
"""

import sys
import os
from pathlib import Path
import json
import logging

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入视频分析器模块
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


class VideoAnalyzerService:
    """
    视频分析服务类，使用环境变量进行配置
    """
    
    def __init__(self):
        """初始化服务，从环境变量加载配置"""
        self._load_config()
        self._init_client()
    
    def _load_config(self):
        """从环境变量加载配置"""
        self.config = {
            # 客户端配置
            "client_type": os.getenv("CLIENT_TYPE", "ollama"),
            "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "gemma2:9b"),
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_api_url": os.getenv("OPENAI_API_URL", "https://openrouter.ai/api/v1"),
            "openai_model": os.getenv("OPENAI_MODEL", "meta-llama/llama-3.2-11b-vision-instruct"),
            
            # 通用配置
            "temperature": float(os.getenv("TEMPERATURE", "0.2")),
            "output_dir": Path(os.getenv("OUTPUT_DIR", "output")),
            "keep_frames": os.getenv("KEEP_FRAMES", "false").lower() == "true",
            
            # 音频处理配置
            "whisper_model": os.getenv("WHISPER_MODEL", "medium"),
            "audio_device": os.getenv("AUDIO_DEVICE", "cpu"),
            "audio_language": os.getenv("AUDIO_LANGUAGE"),
            
            # 帧处理配置
            "frames_per_minute": int(os.getenv("FRAMES_PER_MINUTE", "30")),
            "max_frames": int(os.getenv("MAX_FRAMES", "100")),
            
            # 自定义提示
            "custom_prompt": os.getenv("CUSTOM_PROMPT", ""),
        }
    
    def _init_client(self):
        """初始化LLM客户端"""
        client_type = self.config["client_type"]
        if client_type == "ollama":
            self.client = OllamaClient(self.config["ollama_url"])
            self.model = self.config["ollama_model"]
        elif client_type == "openai_api":
            if not self.config["openai_api_key"]:
                raise ValueError("OpenAI API key is required when using openai_api client")
            self.client = GenericOpenAIAPIClient(
                self.config["openai_api_key"],
                self.config["openai_api_url"]
            )
            self.model = self.config["openai_model"]
        else:
            raise ValueError(f"Unknown client type: {client_type}")
        
        logger.info(f"Initialized client: {client_type}, model: {self.model}")
    
    def analyze_video(self, video_path: str):
        """
        分析视频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            dict: 分析结果
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # 创建输出目录
        output_dir = self.config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
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
        
        try:
            # 阶段1: 音频和帧处理
            logger.info("Stage 1: Processing audio and extracting frames...")
            
            # 初始化音频处理器
            audio_processor = AudioProcessor(
                language=self.config["audio_language"],
                model_size_or_path=self.config["whisper_model"],
                device=self.config["audio_device"]
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
                else:
                    logger.info(f"Audio transcription completed. Detected language: {transcript.language}")
            
            # 提取帧
            logger.info("Extracting frames from video...")
            processor = VideoProcessor(
                video_path, 
                output_dir / "frames", 
                ""  # 根据经验，帧提取不使用模型
            )
            frames = processor.extract_keyframes(
                frames_per_minute=self.config["frames_per_minute"],
                max_frames=self.config["max_frames"]
            )
            logger.info(f"Extracted {len(frames)} frames from video")
            
            # 阶段2: 帧分析
            logger.info("Stage 2: Analyzing frames...")
            analyzer = VideoAnalyzer(
                self.client, 
                self.model, 
                prompt_loader,
                self.config["temperature"],
                self.config["custom_prompt"]
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
                    "client": self.config["client_type"],
                    "model": self.model,
                    "whisper_model": self.config["whisper_model"],
                    "frames_per_minute": self.config["frames_per_minute"],
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
            
            # 输出结果摘要
            if transcript and transcript.text:
                logger.info("\nTranscript:")
                logger.info(transcript.text[:200] + "..." if len(transcript.text) > 200 else transcript.text)
            
            if video_description:
                logger.info("\nVideo Description:")
                description_text = video_description.get("response", "No description generated")
                logger.info(description_text[:500] + "..." if len(description_text) > 500 else description_text)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during video analysis: {e}")
            raise
        finally:
            # 清理临时文件
            if not self.config["keep_frames"]:
                self._cleanup_files(output_dir)
    
    def _cleanup_files(self, output_dir: Path):
        """
        清理临时文件
        
        Args:
            output_dir: 输出目录路径
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


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python test_service.py <video_path>")
        print("Make sure to set up your .env file with the required configuration.")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    try:
        # 创建服务实例
        service = VideoAnalyzerService()
        
        # 分析视频
        results = service.analyze_video(video_path)
        
        print("Video analysis completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Failed to analyze video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()