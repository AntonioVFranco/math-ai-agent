"""
Emoji Regular Expression Patterns

This module contains comprehensive regex patterns for detecting and removing
Unicode emoji characters from text. The patterns cover the full range of
Unicode emoji blocks and sequences.

Author: MathBoardAI Agent Team
Task: Global Project Refactoring - Emoji Removal
"""

import re

# Comprehensive emoji regex pattern covering all Unicode emoji blocks
# Based on Unicode 15.0 emoji specification
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

# Alternative comprehensive pattern using Unicode categories
# This catches more edge cases and newer emoji
EMOJI_PATTERN_EXTENDED = re.compile(
    r'[\U00010000-\U0010ffff]|'  # All supplementary plane characters
    r'[\u2600-\u26FF]|'          # Miscellaneous Symbols
    r'[\u2700-\u27BF]|'          # Dingbats
    r'[\u3000-\u303F]|'          # CJK Symbols and Punctuation (partial)
    r'[\u3297-\u3299]|'          # Circled ideographs
    r'[\uFE00-\uFE0F]|'          # Variation Selectors
    r'[\U0001F000-\U0001F02F]|'  # Mahjong Tiles
    r'[\U0001F0A0-\U0001F0FF]|'  # Playing Cards
    r'[\U0001F100-\U0001F1FF]|'  # Enclosed Alphanumeric Supplement
    r'[\U0001F200-\U0001F2FF]|'  # Enclosed Ideographic Supplement
    r'[\U0001F300-\U0001F5FF]|'  # Miscellaneous Symbols and Pictographs
    r'[\U0001F600-\U0001F64F]|'  # Emoticons
    r'[\U0001F680-\U0001F6FF]|'  # Transport and Map Symbols
    r'[\U0001F700-\U0001F77F]|'  # Alchemical Symbols
    r'[\U0001F780-\U0001F7FF]|'  # Geometric Shapes Extended
    r'[\U0001F800-\U0001F8FF]|'  # Supplemental Arrows-C
    r'[\U0001F900-\U0001F9FF]|'  # Supplemental Symbols and Pictographs
    r'[\U0001FA00-\U0001FA6F]|'  # Chess Symbols
    r'[\U0001FA70-\U0001FAFF]',  # Symbols and Pictographs Extended-A
    flags=re.UNICODE
)

# Most commonly used emoji patterns for quick testing
COMMON_EMOJI_PATTERNS = [
    r'🎉',  # party popper
    r'✅',  # check mark
    r'🚀',  # rocket
    r'📊',  # bar chart
    r'🔧',  # wrench
    r'📚',  # books
    r'🧪',  # test tube
    r'🎯',  # direct hit
    r'🔒',  # lock
    r'🤝',  # handshake
    r'🏆',  # trophy
    r'💡',  # light bulb
    r'📈',  # chart increasing
    r'🔍',  # magnifying glass
    r'🎨',  # artist palette
    r'🔮',  # crystal ball
    r'📞',  # telephone receiver
    r'🧮',  # abacus
    r'🌟',  # glowing star
    r'🛠️',  # hammer and wrench
    r'🐳',  # whale
    r'🤖',  # robot
    r'❌',  # cross mark
    r'⚡',  # high voltage
    r'📦',  # package
    r'📄',  # page facing up
    r'💻',  # laptop
    r'🔗',  # link
    r'📲',  # mobile phone with arrow
]

def remove_emojis(text: str, pattern=None) -> str:
    """
    Remove emoji characters from text using the specified pattern.
    
    Args:
        text (str): The input text containing potential emoji characters
        pattern (re.Pattern, optional): Custom regex pattern. Defaults to EMOJI_PATTERN.
    
    Returns:
        str: Text with emoji characters removed
    """
    if pattern is None:
        pattern = EMOJI_PATTERN
    
    # Remove emojis and clean up extra whitespace
    cleaned_text = pattern.sub('', text)
    
    # Clean up multiple consecutive spaces that might result from emoji removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Clean up leading/trailing whitespace on each line
    lines = cleaned_text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    
    # Rejoin lines, preserving intentional line breaks
    result = '\n'.join(cleaned_lines)
    
    # Remove any trailing whitespace from the entire text
    return result.strip()

def count_emojis(text: str, pattern=None) -> int:
    """
    Count the number of emoji characters in text.
    
    Args:
        text (str): The input text to analyze
        pattern (re.Pattern, optional): Custom regex pattern. Defaults to EMOJI_PATTERN.
    
    Returns:
        int: Number of emoji characters found
    """
    if pattern is None:
        pattern = EMOJI_PATTERN
    
    matches = pattern.findall(text)
    return len(matches)

def has_emojis(text: str, pattern=None) -> bool:
    """
    Check if text contains any emoji characters.
    
    Args:
        text (str): The input text to check
        pattern (re.Pattern, optional): Custom regex pattern. Defaults to EMOJI_PATTERN.
    
    Returns:
        bool: True if emojis are found, False otherwise
    """
    if pattern is None:
        pattern = EMOJI_PATTERN
    
    return bool(pattern.search(text))

def test_emoji_patterns():
    """
    Test function to validate emoji detection patterns.
    """
    test_text = "Hello 🎉 World ✅ This is a test 🚀 with emojis 📊!"
    
    print("Testing emoji detection:")
    print(f"Original text: {test_text}")
    print(f"Has emojis: {has_emojis(test_text)}")
    print(f"Emoji count: {count_emojis(test_text)}")
    print(f"Cleaned text: {remove_emojis(test_text)}")
    
    # Test with extended pattern
    print(f"\nTesting with extended pattern:")
    print(f"Has emojis (extended): {has_emojis(test_text, EMOJI_PATTERN_EXTENDED)}")
    print(f"Emoji count (extended): {count_emojis(test_text, EMOJI_PATTERN_EXTENDED)}")
    print(f"Cleaned text (extended): {remove_emojis(test_text, EMOJI_PATTERN_EXTENDED)}")

if __name__ == "__main__":
    test_emoji_patterns()