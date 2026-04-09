#!/usr/bin/env python3
"""
GraphBuilder Migration Script - Move legacy files to new enterprise structure.

This script intelligently migrates existing files to the new sophisticated
directory structure while preserving functionality and updating imports.
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class GraphBuilderMigration:
    """
    Sophisticated migration engine for GraphBuilder project restructuring.
    
    Handles intelligent file migration, import updates, and structure
    transformation while preserving all functionality.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_root = self.project_root / "src" / "graphbuilder"
        
        # Migration mappings: old_file -> (new_location, new_name)
        self.file_mappings = {
            # Core database and graph operations
            "dbAccess.py": ("infrastructure/database", "neo4j_client.py"),
            "graphTransformer.py": ("core/graph", "transformer.py"),
            
            # Processing and extraction
            "processing.py": ("core/processing", "processor.py"),
            "llm.py": ("infrastructure/services", "legacy_llm.py"),
            
            # Content retrieval
            "webpage_retriever.py": ("infrastructure/crawlers", "web_crawler.py"),
            "sync_urlRetriever.py": ("infrastructure/crawlers", "sync_crawler.py"),
            "json_retriever.py": ("infrastructure/crawlers", "json_crawler.py"),
            "local_file.py": ("infrastructure/crawlers", "file_crawler.py"),
            
            # Main applications
            "main_url.py": ("application/cli", "legacy_url_main.py"),
            "main_json.py": ("application/cli", "legacy_json_main.py"),
            "main_para.py": ("application/cli", "legacy_parallel_main.py"),
            "main_test.py": ("application/cli", "legacy_test_main.py"),
            "main_url_sync.py": ("application/cli", "legacy_url_sync_main.py"),
            
            # Samples and examples
            "sample_fromJson.py": ("examples", "json_sample.py"),
            
            # Shared utilities
            "shared/common_fn.py": ("core/utils", "common_functions.py"),
            "shared/constants.py": ("core/utils", "constants.py"),
            "shared/schema_extraction.py": ("core/schema", "extraction.py"),
            
            # Entities
            "entities/source_node.py": ("domain/entities", "source_node.py"),
            "entities/user_credential.py": ("domain/entities", "user_credential.py"),
            
            # Image processing
            "ImageEmbed/clip.py": ("infrastructure/image", "clip_processor.py"),
            "ImageEmbed/getPictures.py": ("infrastructure/image", "image_downloader.py"),
        }
        
        # Import transformation rules
        self.import_transformations = [
            # Update relative imports to absolute
            (r"from\s+\.(\w+)", r"from graphbuilder.\1"),
            (r"import\s+\.(\w+)", r"import graphbuilder.\1"),
            
            # Update specific module imports
            (r"from\s+dbAccess", r"from graphbuilder.infrastructure.database.neo4j_client"),
            (r"import\s+dbAccess", r"import graphbuilder.infrastructure.database.neo4j_client as dbAccess"),
            (r"from\s+processing", r"from graphbuilder.core.processing.processor"),
            (r"import\s+processing", r"import graphbuilder.core.processing.processor as processing"),
            (r"from\s+llm", r"from graphbuilder.infrastructure.services.legacy_llm"),
            (r"import\s+llm", r"import graphbuilder.infrastructure.services.legacy_llm as llm"),
            
            # Update shared imports
            (r"from\s+shared\.common_fn", r"from graphbuilder.core.utils.common_functions"),
            (r"from\s+shared\.constants", r"from graphbuilder.core.utils.constants"),
            (r"from\s+shared\.schema_extraction", r"from graphbuilder.core.schema.extraction"),
            
            # Update entity imports
            (r"from\s+entities\.source_node", r"from graphbuilder.domain.entities.source_node"),
            (r"from\s+entities\.user_credential", r"from graphbuilder.domain.entities.user_credential"),
        ]
    
    def run_migration(self) -> None:
        """Execute complete migration process."""
        
        print("🚀 Starting GraphBuilder Enterprise Migration")
        print("=" * 60)
        
        # Step 1: Backup original files
        print("\n📦 Creating backup of original files...")
        backup_dir = self.create_backup()
        print(f"   Backup created at: {backup_dir}")
        
        # Step 2: Migrate files
        print("\n📁 Migrating files to new structure...")
        migrated_files = self.migrate_files()
        print(f"   Successfully migrated {len(migrated_files)} files")
        
        # Step 3: Update imports
        print("\n🔗 Updating import statements...")
        updated_files = self.update_imports()
        print(f"   Updated imports in {len(updated_files)} files")
        
        # Step 4: Create additional files
        print("\n✨ Creating additional enterprise files...")
        self.create_additional_files()
        
        # Step 5: Update configuration files
        print("\n⚙️  Updating configuration files...")
        self.update_config_files()
        
        print("\n Migration completed successfully!")
        print("\n Migration Summary:")
        print(f"   • Migrated files: {len(migrated_files)}")
        print(f"   • Updated imports: {len(updated_files)}")
        print(f"   • Backup location: {backup_dir}")
        print(f"   • New structure: {self.src_root}")
        
        print("\n🎉 Your GraphBuilder project is now enterprise-ready!")
        print("   Run 'python -m graphbuilder.cli.main --help' to get started")
    
    def create_backup(self) -> Path:
        """Create backup of original files."""
        
        from datetime import datetime
        
        backup_dir = self.project_root / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(exist_ok=True)
        
        # Copy original files
        for old_file in self.file_mappings.keys():
            old_path = self.project_root / old_file
            if old_path.exists():
                if old_path.is_file():
                    backup_path = backup_dir / old_file
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old_path, backup_path)
                elif old_path.is_dir():
                    shutil.copytree(old_path, backup_dir / old_file)
        
        # Copy other important files
        for filename in ["README.md", "README_ZH.md", "environment.yml", "visited_links.txt"]:
            old_path = self.project_root / filename
            if old_path.exists():
                shutil.copy2(old_path, backup_dir / filename)
        
        return backup_dir
    
    def migrate_files(self) -> List[Tuple[str, str]]:
        """Migrate files to new locations."""
        
        migrated = []
        
        for old_file, (new_dir, new_name) in self.file_mappings.items():
            old_path = self.project_root / old_file
            new_path = self.src_root / new_dir / new_name
            
            if old_path.exists():
                # Create target directory
                new_path.parent.mkdir(parents=True, exist_ok=True)
                
                if old_path.is_file():
                    # Copy and update file
                    shutil.copy2(old_path, new_path)
                    self.add_file_header(new_path, old_file)
                    migrated.append((str(old_path), str(new_path)))
                    print(f"   📄 {old_file} -> {new_dir}/{new_name}")
                elif old_path.is_dir():
                    # Copy directory
                    if new_path.exists():
                        shutil.rmtree(new_path)
                    shutil.copytree(old_path, new_path)
                    migrated.append((str(old_path), str(new_path)))
                    print(f"   📁 {old_file}/ -> {new_dir}/{new_name}/")
        
        return migrated
    
    def add_file_header(self, file_path: Path, original_name: str) -> None:
        """Add enterprise header to migrated file."""
        
        header = f'''"""
{file_path.stem.replace('_', ' ').title()} - Migrated from {original_name}

This module has been migrated to the new GraphBuilder enterprise structure.
Original functionality is preserved with improved organization and standards.

Migration Date: {self.get_current_date()}
Original File: {original_name}
New Location: {file_path.relative_to(self.project_root)}
"""

'''
        
        # Read original content
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Skip if already has docstring
        if original_content.strip().startswith('"""') or original_content.strip().startswith("'''"):
            return
        
        # Add header
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(header + original_content)
    
    def update_imports(self) -> List[str]:
        """Update import statements in all Python files."""
        
        updated_files = []
        
        # Get all Python files in the new structure
        python_files = list(self.src_root.rglob("*.py"))
        
        for file_path in python_files:
            if self.update_file_imports(file_path):
                updated_files.append(str(file_path))
                print(f"   🔗 Updated imports in {file_path.relative_to(self.project_root)}")
        
        return updated_files
    
    def update_file_imports(self, file_path: Path) -> bool:
        """Update imports in a specific file."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply import transformations
            for pattern, replacement in self.import_transformations:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            # Write back if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
            return False
            
        except Exception as e:
            print(f"   ⚠️  Warning: Could not update imports in {file_path}: {e}")
            return False
    
    def create_additional_files(self) -> None:
        """Create additional enterprise files."""
        
        # Create __main__.py for module execution
        main_file = self.src_root / "__main__.py"
        main_content = '''"""
GraphBuilder - Main module entry point.

Allows running GraphBuilder as a module: python -m graphbuilder
"""

from .cli.main import cli

if __name__ == '__main__':
    cli()
'''
        
        with open(main_file, 'w') as f:
            f.write(main_content)
        
        print("   ✨ Created __main__.py for module execution")
        
        # Create setup.py for package installation
        setup_file = self.project_root / "setup.py"
        if not setup_file.exists():
            setup_content = '''"""
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
'''
            
            with open(setup_file, 'w') as f:
                f.write(setup_content)
            
            print("   ✨ Created setup.py for package installation")
        
        # Create pyproject.toml for modern Python packaging
        pyproject_file = self.project_root / "pyproject.toml"
        if not pyproject_file.exists():
            pyproject_content = '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "graphbuilder"
version = "2.0.0"
description = "Enterprise-grade knowledge graph builder"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "GraphBuilder Team"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
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
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
]

[project.scripts]
graphbuilder = "graphbuilder.cli.main:cli"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
src_paths = ["src"]

[tool.mypy]
python_version = "3.8"
strict = true
'''
            
            with open(pyproject_file, 'w') as f:
                f.write(pyproject_content)
            
            print("   ✨ Created pyproject.toml for modern packaging")
        
        # Create requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        requirements_content = '''# GraphBuilder Core Dependencies
neo4j>=5.0.0
langchain>=0.1.0
openai>=1.0.0
beautifulsoup4>=4.11.0
requests>=2.28.0
aiohttp>=3.8.0
click>=8.0.0
rich>=12.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
aiofiles>=23.0.0

# Optional dependencies for enhanced functionality
PyPDF2>=3.0.0  # PDF processing
python-docx>=0.8.11  # DOCX processing
pillow>=10.0.0  # Image processing
sentence-transformers>=2.2.0  # Text embeddings
'''
        
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        
        print("   ✨ Created requirements.txt")
    
    def update_config_files(self) -> None:
        """Update configuration files for enterprise setup."""
        
        # Create .env.example
        env_example = self.project_root / ".env.example"
        env_content = '''# GraphBuilder Configuration Example
# Copy this file to .env and update with your actual values

# Database Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
# AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# LLM Model Settings
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000

# Processing Configuration
CHUNK_SIZE=1000
OVERLAP_SIZE=100
MAX_CONCURRENT_TASKS=5

# Crawler Configuration
USER_AGENT="GraphBuilder/2.0.0 (+https://github.com/graphbuilder)"
CRAWLER_TIMEOUT=30
MAX_CONCURRENT_REQUESTS=10

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=detailed
ENABLE_FILE_LOGGING=true
LOG_ROTATION_SIZE=10MB
LOG_RETENTION_DAYS=30

# Security Configuration
ENABLE_API_RATE_LIMITING=true
API_RATE_LIMIT_REQUESTS=100
API_RATE_LIMIT_WINDOW=3600
'''
        
        with open(env_example, 'w') as f:
            f.write(env_content)
        
        print("   ⚙️  Created .env.example")
        
        # Update .gitignore
        gitignore_file = self.project_root / ".gitignore"
        gitignore_additions = '''
# GraphBuilder Enterprise
.env
config/local.yaml
logs/
*.log
backup_*/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation
docs/_build/
'''
        
        if gitignore_file.exists():
            with open(gitignore_file, 'a') as f:
                f.write(gitignore_additions)
        else:
            with open(gitignore_file, 'w') as f:
                f.write(gitignore_additions)
        
        print("   ⚙️  Updated .gitignore")
    
    def get_current_date(self) -> str:
        """Get current date string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")


def main():
    """Main migration function."""
    
    # Get project root (current directory)
    project_root = os.getcwd()
    
    print("GraphBuilder Enterprise Migration Tool")
    print("=====================================")
    print(f"Project Root: {project_root}")
    
    # Confirm migration
    response = input("\n🤔 Do you want to proceed with the migration? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Migration cancelled.")
        return
    
    # Run migration
    migration = GraphBuilderMigration(project_root)
    migration.run_migration()


if __name__ == "__main__":
    main()