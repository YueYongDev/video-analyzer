from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="video-analyzer",
    version="0.1.1",
    author="Jesse White",
    author_email="jesse.white@example.com",
    description="A tool for analyzing videos using Vision models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jessewhite/video-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_data={
        'video_analyzer': [
            'config/*.json',
            'prompts/**/*',
            'prompts/**/*.txt',
        ],
    },
    install_requires=requirements + [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
    ],
    entry_points={
        "console_scripts": [
            "video-analyzer=video_analyzer.cli:main",
            "video-analyzer-api=video_analyzer.api:main",
        ],
    },
    python_requires=">=3.8",
    include_package_data=True
)
