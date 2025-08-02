#!/usr/bin/env python3
"""
Simple documentation verification script.
Checks that all required files exist and have basic structure.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report result."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def check_file_content(filepath, required_content, description):
    """Check if a file contains required content."""
    if not os.path.exists(filepath):
        print(f"❌ {description}: {filepath} - FILE NOT FOUND")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if all(req in content for req in required_content):
            print(f"✅ {description}: {filepath} - Required content found")
            return True
        else:
            missing = [req for req in required_content if req not in content]
            print(f"⚠️  {description}: {filepath} - Missing content: {missing}")
            return False
    except Exception as e:
        print(f"❌ {description}: {filepath} - Error reading file: {e}")
        return False

def main():
    """Main verification function."""
    print("📚 Math AI Agent Documentation Verification")
    print("=" * 50)
    
    success_count = 0
    total_checks = 0
    
    # Core Sphinx files
    core_files = [
        ("docs/conf.py", "Sphinx configuration"),
        ("docs/index.rst", "Main index file"),
        ("docs/Makefile", "Build Makefile"),
        ("docs/_static/custom.css", "Custom CSS"),
        ("requirements-docs.txt", "Documentation requirements")
    ]
    
    for filepath, description in core_files:
        total_checks += 1
        if check_file_exists(filepath, description):
            success_count += 1
    
    # Technical documentation files
    tech_docs = [
        ("docs/architecture.rst", "Architecture documentation"),
        ("docs/developer_guide.rst", "Developer guide"),
        ("docs/api_reference.rst", "API reference"),
        ("docs/adding_new_modules.rst", "Module addition tutorial")
    ]
    
    for filepath, description in tech_docs:
        total_checks += 1
        if check_file_exists(filepath, description):
            success_count += 1
    
    # User documentation files
    user_docs = [
        ("docs/quickstart.rst", "Quick start guide"),
        ("docs/tutorials/linear_algebra.rst", "Linear algebra tutorial"),
        ("docs/tutorials/optimization.rst", "Optimization tutorial"),
        ("docs/tutorials/statistics.rst", "Statistics tutorial"),
        ("docs/faq.rst", "FAQ")
    ]
    
    for filepath, description in user_docs:
        total_checks += 1
        if check_file_exists(filepath, description):
            success_count += 1
    
    # Check directory structure
    required_dirs = [
        "docs/_static",
        "docs/_templates", 
        "docs/tutorials"
    ]
    
    print("\n📁 Directory Structure Check:")
    print("-" * 30)
    
    for dir_path in required_dirs:
        total_checks += 1
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"✅ Directory exists: {dir_path}")
            success_count += 1
        else:
            print(f"❌ Directory missing: {dir_path}")
    
    # Check key content in important files
    print("\n📝 Content Verification:")
    print("-" * 25)
    
    content_checks = [
        ("docs/conf.py", ["sphinx_rtd_theme", "autodoc", "napoleon"], "Sphinx config content"),
        ("docs/index.rst", ["Math AI Agent", "toctree", "quickstart"], "Index content"),
        ("requirements-docs.txt", ["sphinx", "sphinx-rtd-theme"], "Requirements content")
    ]
    
    for filepath, required_content, description in content_checks:
        total_checks += 1
        if check_file_content(filepath, required_content, description):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {success_count}/{total_checks} checks")
    print(f"❌ Failed: {total_checks - success_count}/{total_checks} checks")
    
    if success_count == total_checks:
        print("🎉 All documentation files are present and correctly structured!")
        print("\n📋 Next steps:")
        print("1. Install Sphinx: pip install -r requirements-docs.txt")
        print("2. Build docs: make html")
        print("3. View docs: open _build/html/index.html")
        return 0
    else:
        print(f"⚠️  Documentation setup incomplete. Please address the missing files/content.")
        return 1

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)  # Go to project root
    sys.exit(main())