# Math AI Agent Documentation

This directory contains the complete Sphinx documentation for the Math AI Agent project.

## Documentation Structure

### Technical Documentation (DOC-001)
- **`architecture.rst`** - System architecture overview with data flow diagrams
- **`developer_guide.rst`** - Complete development setup and contribution guide
- **`api_reference.rst`** - Auto-generated API documentation from docstrings
- **`adding_new_modules.rst`** - Tutorial for extending the system with new solvers

### User Documentation (DOC-002)
- **`quickstart.rst`** - Getting started guide for new users
- **`tutorials/`** - Domain-specific tutorials:
  - `linear_algebra.rst` - Matrix operations, eigenvalues, decompositions
  - `optimization.rst` - Gradient descent, critical points, constrained optimization
  - `statistics.rst` - Descriptive stats, hypothesis testing, distributions
- **`faq.rst`** - Frequently asked questions and troubleshooting

### Main Files
- **`index.rst`** - Main landing page with project overview
- **`conf.py`** - Sphinx configuration with autodoc, theming, and extensions
- **`Makefile`** - Build commands for generating HTML documentation

## Building the Documentation

### Prerequisites
```bash
pip install -r requirements-docs.txt
```

### Build Commands
```bash
# Generate HTML documentation
make html

# Clean and rebuild
make clean html

# For development with auto-reload
make livehtml
```

### View Documentation
After building, open `_build/html/index.html` in your browser.

## Documentation Features

### Sphinx Extensions
- **autodoc** - Automatic API documentation from docstrings
- **napoleon** - Google/NumPy style docstring support
- **viewcode** - Source code links in documentation
- **intersphinx** - Cross-references to external docs (NumPy, SciPy, etc.)
- **myst-parser** - Markdown support alongside reStructuredText
- **sphinxcontrib-mermaid** - Mermaid diagram support

### Theme and Styling
- **Read the Docs theme** with custom CSS
- **Responsive design** for mobile and desktop
- **Syntax highlighting** for code examples
- **Copy button** for code blocks
- **Custom styling** for mathematical content

### Interactive Elements
- **Mermaid diagrams** for architecture visualization
- **Graphviz** support for complex diagrams
- **LaTeX math** rendering with MathJax
- **Cross-references** between sections
- **Search functionality** built-in

## Content Guidelines

### Writing Style
- Clear, concise explanations
- Step-by-step tutorials with examples
- Code examples with expected outputs
- Real-world use cases and applications

### Technical Standards
- All public functions have comprehensive docstrings
- Google-style docstring format
- Type hints throughout the codebase
- Examples in docstrings where helpful

## Verification

Run the documentation verification script:
```bash
python3 docs/verify_docs.py
```

This checks that all required files exist and have proper structure.

## ReadTheDocs Integration

This documentation is designed to be hosted on ReadTheDocs with:
- Automatic builds from the main branch
- Version management
- PDF/EPUB export support
- Search index generation

## Contributing to Documentation

1. **Edit Content**: Modify `.rst` files in the appropriate directories
2. **Add Examples**: Include working code examples with expected outputs
3. **Update API Docs**: Ensure docstrings are comprehensive and up-to-date
4. **Test Building**: Run `make html` to verify no errors
5. **Check Links**: Verify all internal and external links work

## File Organization

```
docs/
├── _build/           # Generated documentation (git-ignored)
├── _static/          # Static assets (CSS, images)
├── _templates/       # Custom Sphinx templates  
├── tutorials/        # User tutorial files
├── *.rst            # Main documentation files
├── conf.py          # Sphinx configuration
├── Makefile         # Build commands
└── README.md        # This file
```

## Quality Standards

- All documentation builds without warnings
- Cross-references work correctly
- Code examples are tested and accurate
- Mobile-responsive design
- Professional appearance and navigation

This documentation serves both developers working on the Math AI Agent system and end-users learning to solve mathematical problems with the tool.