#!/usr/bin/env python3
"""
Global Project Refactoring Script

This script performs comprehensive project-wide refactoring operations:
1. Rename "Math AI Agent" to "MathBoardAI Agent" (case-sensitive)
2. Remove all emoji characters from text files

The script processes all text-based files recursively while ignoring
binary files and specified directories.

Author: MathBoardAI Agent Team
Task: Global Project Refactoring - Rename and Clean
"""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Set
import argparse
from datetime import datetime

# Import emoji patterns from separate module
try:
    from emoji_regex import EMOJI_PATTERN, remove_emojis, has_emojis, count_emojis
except ImportError:
    print("Warning: emoji_regex.py not found. Using fallback emoji pattern.")
    
    # Fallback emoji pattern if emoji_regex.py is not available
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F018-\U0001F270"  # various symbols
        "\U00003030"             # wavy dash
        "\U0000303D"             # part alternation mark
        "\U00003297"             # circled ideograph congratulation
        "\U00003299"             # circled ideograph secret
        "\U0001F004"             # mahjong red dragon
        "\U0001F0CF"             # playing card black joker
        "\U0001F170-\U0001F251"  # enclosed alphanumeric supplement
        "]+",
        flags=re.UNICODE
    )
    
    def remove_emojis(text: str) -> str:
        """Fallback emoji removal function."""
        cleaned_text = EMOJI_PATTERN.sub('', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        lines = cleaned_text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        result = '\n'.join(cleaned_lines)
        return result.strip()
    
    def has_emojis(text: str) -> bool:
        """Fallback emoji detection function."""
        return bool(EMOJI_PATTERN.search(text))
    
    def count_emojis(text: str) -> int:
        """Fallback emoji counting function."""
        matches = EMOJI_PATTERN.findall(text)
        return len(matches)


class ProjectRefactorer:
    """
    Comprehensive project refactoring class that handles renaming and emoji removal.
    """
    
    # File extensions to process (text-based files only)
    TARGET_EXTENSIONS = {
        '.py', '.md', '.txt', '.rst', '.sh', '.yml', '.yaml', 
        '.json', '.cfg', '.ini', '.conf', '.dockerfile', 
        '.gitignore', '.env', '.example'
    }
    
    # Directories to ignore during processing
    IGNORED_DIRECTORIES = {
        '.git', '.idea', '__pycache__', 'venv', 'env', 'ENV',
        'node_modules', '.vscode', '_build', 'build', 'dist',
        '.pytest_cache', '.mypy_cache', '.coverage', 'htmlcov',
        'test_results', 'docker-data', '.DS_Store'
    }
    
    # Files to ignore (exact matches)
    IGNORED_FILES = {
        '.DS_Store', 'Thumbs.db', 'desktop.ini'
    }
    
    def __init__(self, root_path: str = '.', dry_run: bool = False, 
                 create_backups: bool = True, verbose: bool = True):
        """
        Initialize the project refactorer.
        
        Args:
            root_path (str): Root directory path to start refactoring
            dry_run (bool): If True, only simulate changes without modifying files
            create_backups (bool): If True, create .bak files before modification
            verbose (bool): If True, print detailed progress information
        """
        self.root_path = Path(root_path).resolve()
        self.dry_run = dry_run
        self.create_backups = create_backups
        self.verbose = verbose
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'files_with_name_changes': 0,
            'files_with_emoji_removals': 0,
            'total_name_replacements': 0,
            'total_emojis_removed': 0,
            'errors': 0,
            'skipped_files': 0
        }
        
        # Processing log
        self.log_entries = []
        
        print(f"Initializing ProjectRefactorer")
        print(f"Root path: {self.root_path}")
        print(f"Dry run: {self.dry_run}")
        print(f"Create backups: {self.create_backups}")
        print(f"Verbose: {self.verbose}")
    
    def is_text_file(self, file_path: Path) -> bool:
        """
        Determine if a file is a text file that should be processed.
        
        Args:
            file_path (Path): Path to the file to check
        
        Returns:
            bool: True if file should be processed, False otherwise
        """
        # Check if file has a target extension
        if file_path.suffix.lower() in self.TARGET_EXTENSIONS:
            return True
        
        # Check for files without extensions that might be text
        if not file_path.suffix:
            # Common text files without extensions
            text_files_no_ext = {
                'README', 'LICENSE', 'CHANGELOG', 'MANIFEST',
                'Dockerfile', 'Makefile', 'requirements'
            }
            if file_path.name in text_files_no_ext:
                return True
        
        # Try to detect text files by attempting to read them
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 1024 bytes to check if it's text
                sample = f.read(1024)
                # If we can read it and it contains printable characters, it's likely text
                if sample and all(ord(char) < 128 or char.isprintable() for char in sample[:100]):
                    return True
        except Exception:
            pass
        
        return False
    
    def should_skip_path(self, path: Path) -> bool:
        """
        Check if a path should be skipped during processing.
        
        Args:
            path (Path): Path to check
        
        Returns:
            bool: True if path should be skipped, False otherwise
        """
        # Skip ignored directories
        for part in path.parts:
            if part in self.IGNORED_DIRECTORIES:
                return True
        
        # Skip ignored files
        if path.name in self.IGNORED_FILES:
            return True
        
        return False
    
    def create_backup(self, file_path: Path) -> bool:
        """
        Create a backup of the file before modification.
        
        Args:
            file_path (Path): Path to the file to backup
        
        Returns:
            bool: True if backup was created successfully, False otherwise
        """
        if not self.create_backups:
            return True
        
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        
        try:
            shutil.copy2(file_path, backup_path)
            if self.verbose:
                self.log_entries.append(f"  Created backup: {backup_path}")
            return True
        except Exception as e:
            self.log_entries.append(f"  ERROR: Failed to create backup for {file_path}: {e}")
            return False
    
    def process_file_content(self, content: str) -> Tuple[str, int, int]:
        """
        Process file content by performing name replacement and emoji removal.
        
        Args:
            content (str): Original file content
        
        Returns:
            Tuple[str, int, int]: (processed_content, name_replacements, emojis_removed)
        """
        # Step 1: Replace "Math AI Agent" with "MathBoardAI Agent" (case-sensitive)
        original_content = content
        name_replacements = content.count("Math AI Agent")
        content = content.replace("Math AI Agent", "MathBoardAI Agent")
        
        # Step 2: Remove emojis
        emojis_before = count_emojis(content)
        if emojis_before > 0:
            content = remove_emojis(content)
            emojis_after = count_emojis(content)
            emojis_removed = emojis_before - emojis_after
        else:
            emojis_removed = 0
        
        return content, name_replacements, emojis_removed
    
    def process_file(self, file_path: Path) -> bool:
        """
        Process a single file for refactoring.
        
        Args:
            file_path (Path): Path to the file to process
        
        Returns:
            bool: True if file was processed successfully, False otherwise
        """
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
            
            # Process the content
            processed_content, name_replacements, emojis_removed = self.process_file_content(original_content)
            
            # Check if any changes were made
            if original_content == processed_content:
                if self.verbose:
                    self.log_entries.append(f"  No changes needed")
                return True
            
            # Update statistics
            self.stats['files_processed'] += 1
            if name_replacements > 0:
                self.stats['files_with_name_changes'] += 1
                self.stats['total_name_replacements'] += name_replacements
            if emojis_removed > 0:
                self.stats['files_with_emoji_removals'] += 1
                self.stats['total_emojis_removed'] += emojis_removed
            
            # Log changes
            changes = []
            if name_replacements > 0:
                changes.append(f"{name_replacements} name replacement(s)")
            if emojis_removed > 0:
                changes.append(f"{emojis_removed} emoji(s) removed")
            
            if self.verbose and changes:
                self.log_entries.append(f"  Changes: {', '.join(changes)}")
            
            # Write the processed content back to the file (if not dry run)
            if not self.dry_run:
                # Create backup first
                if not self.create_backup(file_path):
                    self.stats['errors'] += 1
                    return False
                
                # Write the modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
            
            return True
            
        except Exception as e:
            self.log_entries.append(f"  ERROR: Failed to process {file_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def scan_and_process(self) -> List[Path]:
        """
        Scan the project directory and process all eligible files.
        
        Returns:
            List[Path]: List of files that were processed
        """
        processed_files = []
        
        print(f"\nScanning directory: {self.root_path}")
        print("=" * 60)
        
        # Walk through all files and directories
        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self.should_skip_path(root_path / d)]
            
            for file_name in files:
                file_path = root_path / file_name
                
                # Skip if path should be ignored
                if self.should_skip_path(file_path):
                    self.stats['skipped_files'] += 1
                    continue
                
                # Skip if not a text file
                if not self.is_text_file(file_path):
                    self.stats['skipped_files'] += 1
                    continue
                
                # Process the file
                relative_path = file_path.relative_to(self.root_path)
                print(f"Processing: {relative_path}")
                
                if self.process_file(file_path):
                    processed_files.append(file_path)
                
                # Print log entries for this file
                if self.verbose and self.log_entries:
                    for entry in self.log_entries[-10:]:  # Show last few entries
                        if entry.startswith("  "):
                            print(entry)
                    self.log_entries = []  # Clear entries after printing
        
        return processed_files
    
    def run_refactoring(self) -> bool:
        """
        Execute the complete refactoring process.
        
        Returns:
            bool: True if refactoring completed successfully, False otherwise
        """
        print(f"Starting project refactoring at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target extensions: {sorted(self.TARGET_EXTENSIONS)}")
        print(f"Ignored directories: {sorted(self.IGNORED_DIRECTORIES)}")
        
        if self.dry_run:
            print("\n*** DRY RUN MODE - No files will be modified ***")
        
        try:
            # Process all files
            processed_files = self.scan_and_process()
            
            # Print summary
            self.print_summary()
            
            return self.stats['errors'] == 0
            
        except KeyboardInterrupt:
            print("\n\nRefactoring interrupted by user.")
            return False
        except Exception as e:
            print(f"\n\nFatal error during refactoring: {e}")
            return False
    
    def print_summary(self):
        """Print a comprehensive summary of the refactoring results."""
        print("\n" + "=" * 60)
        print("REFACTORING SUMMARY")
        print("=" * 60)
        
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files skipped: {self.stats['skipped_files']}")
        print(f"Errors encountered: {self.stats['errors']}")
        
        print(f"\nNAME CHANGES:")
        print(f"  Files with name changes: {self.stats['files_with_name_changes']}")
        print(f"  Total 'Math AI Agent' → 'MathBoardAI Agent': {self.stats['total_name_replacements']}")
        
        print(f"\nEMOJI REMOVAL:")
        print(f"  Files with emoji removals: {self.stats['files_with_emoji_removals']}")
        print(f"  Total emojis removed: {self.stats['total_emojis_removed']}")
        
        if self.dry_run:
            print(f"\n*** DRY RUN COMPLETE - No files were actually modified ***")
        else:
            print(f"\n*** REFACTORING COMPLETE ***")
        
        if self.stats['errors'] > 0:
            print(f"\n⚠️  Warning: {self.stats['errors']} errors occurred during processing")
        else:
            print(f"\n✓ All files processed successfully")
    
    def verify_changes(self) -> Tuple[bool, bool]:
        """
        Verify that the refactoring was successful by searching for remaining instances.
        
        Returns:
            Tuple[bool, bool]: (no_math_ai_agent_found, no_emojis_found)
        """
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        
        math_ai_instances = []
        emoji_instances = []
        
        # Scan for remaining instances
        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self.should_skip_path(root_path / d)]
            
            for file_name in files:
                file_path = root_path / file_name
                
                # Skip if path should be ignored
                if self.should_skip_path(file_path):
                    continue
                
                # Skip if not a text file
                if not self.is_text_file(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Check for "Math AI Agent"
                    if "Math AI Agent" in content:
                        math_ai_instances.append(file_path.relative_to(self.root_path))
                    
                    # Check for emojis
                    if has_emojis(content):
                        emoji_count = count_emojis(content)
                        emoji_instances.append((file_path.relative_to(self.root_path), emoji_count))
                
                except Exception as e:
                    print(f"Warning: Could not verify {file_path}: {e}")
        
        # Report results
        no_math_ai = len(math_ai_instances) == 0
        no_emojis = len(emoji_instances) == 0
        
        print(f"Checking for remaining 'Math AI Agent' instances:")
        if no_math_ai:
            print("  ✓ No instances found")
        else:
            print(f"  ✗ Found {len(math_ai_instances)} files with remaining instances:")
            for file_path in math_ai_instances:
                print(f"    - {file_path}")
        
        print(f"\nChecking for remaining emoji characters:")
        if no_emojis:
            print("  ✓ No emojis found")
        else:
            print(f"  ✗ Found emojis in {len(emoji_instances)} files:")
            for file_path, count in emoji_instances:
                print(f"    - {file_path} ({count} emojis)")
        
        return no_math_ai, no_emojis


def main():
    """Main function to run the refactoring script."""
    parser = argparse.ArgumentParser(
        description="Global Project Refactoring: Rename 'Math AI Agent' to 'MathBoardAI Agent' and remove emojis"
    )
    parser.add_argument(
        '--root', '-r', 
        default='.', 
        help="Root directory path (default: current directory)"
    )
    parser.add_argument(
        '--dry-run', '-d', 
        action='store_true', 
        help="Perform a dry run without modifying files"
    )
    parser.add_argument(
        '--no-backup', '-n', 
        action='store_true', 
        help="Don't create backup files"
    )
    parser.add_argument(
        '--quiet', '-q', 
        action='store_true', 
        help="Reduce output verbosity"
    )
    parser.add_argument(
        '--verify', '-v', 
        action='store_true', 
        help="Run verification after refactoring"
    )
    
    args = parser.parse_args()
    
    # Initialize refactorer
    refactorer = ProjectRefactorer(
        root_path=args.root,
        dry_run=args.dry_run,
        create_backups=not args.no_backup,
        verbose=not args.quiet
    )
    
    # Run refactoring
    success = refactorer.run_refactoring()
    
    # Run verification if requested
    if args.verify and not args.dry_run:
        no_math_ai, no_emojis = refactorer.verify_changes()
        
        if no_math_ai and no_emojis:
            print("\n🎉 Verification successful! All acceptance criteria met.")
        else:
            print("\n⚠️  Verification found remaining issues. Please review the output above.")
            success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()