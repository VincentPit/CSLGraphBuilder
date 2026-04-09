"""
GraphBuilder Setup Configuration

Enterprise-grade knowledge graph builder with advanced AI capabilities.
"""

from setuptools import setup, find_packages

setup(
    name="graphbuilder",
    version="2.0.0",
    description="Enterprise-grade knowledge graph builder",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="GraphBuilder Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "neo4j>=5.0.0",
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "beautifulsoup4>=4.11.0",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        "click>=8.0.0",
        "rich>=12.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "aiofiles>=23.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "graphbuilder=graphbuilder.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
