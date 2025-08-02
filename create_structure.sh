#!/bin/bash

# Math AI Agent - Project Structure Setup Script
# This script creates the complete directory structure and initial files for the math-ai-agent project

echo "Creating math-ai-agent project structure..."

# Create all directories
echo "Creating directories..."
mkdir -p data/benchmarks
mkdir -p docs
mkdir -p examples
mkdir -p src/core
mkdir -p src/interface
mkdir -p src/mcps
mkdir -p src/solvers
mkdir -p src/utils
mkdir -p tests

# Create __init__.py files for Python packages
echo "Creating __init__.py files..."
touch src/core/__init__.py
touch src/interface/__init__.py
touch src/mcps/__init__.py
touch src/solvers/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# Create requirements.txt with core dependencies
echo "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
openai
gradio
sympy
numpy
scipy
pandas
pytest
python-dotenv
EOF

# Create .gitignore file
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
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
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# VS Code
.vscode/

# macOS
.DS_Store

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini
EOF

# Create README.md
echo "Creating README.md..."
cat > README.md << 'EOF'
# Math AI Agent

A specialized math AI agent that surpasses traditional LLMs in accuracy and problem-solving capabilities for AI-related mathematics.
EOF

echo "Project structure created successfully!"
echo ""
echo "Directory structure:"
echo "math-ai-agent/"
echo "├── data/"
echo "│   └── benchmarks/"
echo "├── docs/"
echo "├── examples/"
echo "├── src/"
echo "│   ├── core/"
echo "│   │   └── __init__.py"
echo "│   ├── interface/"
echo "│   │   └── __init__.py"
echo "│   ├── mcps/"
echo "│   │   └── __init__.py"
echo "│   ├── solvers/"
echo "│   │   └── __init__.py"
echo "│   └── utils/"
echo "│       └── __init__.py"
echo "├── tests/"
echo "│   └── __init__.py"
echo "├── .gitignore"
echo "├── README.md"
echo "└── requirements.txt"
echo ""
echo "Setup complete! You can now start developing the math AI agent."