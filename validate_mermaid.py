#!/usr/bin/env python3
"""
Standalone Mermaid diagram validator to identify syntax errors.
"""

import re
from pathlib import Path
import json

def extract_mermaid_diagrams(file_path):
    """Extract all Mermaid diagrams from a markdown file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to find mermaid code blocks
    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    diagrams = re.findall(mermaid_pattern, content, re.DOTALL)
    
    return diagrams

def validate_mermaid_syntax(diagram):
    """Basic validation of Mermaid diagram syntax."""
    errors = []
    lines = diagram.strip().split('\n')
    
    # Check for diagram type
    first_line = lines[0].strip() if lines else ""
    valid_types = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'erDiagram', 'gantt', 'pie', 'journey', 'gitGraph']
    
    diagram_type = None
    for vtype in valid_types:
        if first_line.startswith(vtype):
            diagram_type = vtype
            break
    
    if not diagram_type:
        errors.append(f"Invalid diagram type. First line: '{first_line}'")
    
    # Basic syntax checks
    bracket_count = 0
    quote_count = 0
    in_comment = False
    
    for i, line in enumerate(lines):
        # Skip comments
        if line.strip().startswith('%%'):
            continue
            
        # Count brackets
        bracket_count += line.count('[') - line.count(']')
        bracket_count += line.count('{') - line.count('}')
        bracket_count += line.count('(') - line.count(')')
        
        # Count quotes
        quote_count += line.count('"') + line.count("'")
        
        # Check for common issues
        if '-->' in line and '-->>' in line:
            errors.append(f"Line {i+1}: Mixed arrow types in same line")
        
        if line.strip().endswith(','):
            errors.append(f"Line {i+1}: Trailing comma")
    
    if bracket_count != 0:
        errors.append(f"Unbalanced brackets: {bracket_count}")
    
    if quote_count % 2 != 0:
        errors.append("Unbalanced quotes")
    
    return errors

def check_mermaid_specifics(diagram):
    """Check for Mermaid-specific issues that might cause rendering errors."""
    issues = []
    lines = diagram.strip().split('\n')
    
    # Check for problematic characters in labels
    for i, line in enumerate(lines):
        if '<br/>' in line and not ('"' in line or "'" in line):
            issues.append(f"Line {i+1}: <br/> should be inside quotes")
        
        # Check for emojis without quotes
        emoji_pattern = r'[\U0001F300-\U0001F9FF]'
        if re.search(emoji_pattern, line):
            # Check if emoji is in a quoted string
            quoted_strings = re.findall(r'"[^"]*"', line) + re.findall(r"'[^']*'", line)
            line_without_quotes = line
            for qs in quoted_strings:
                line_without_quotes = line_without_quotes.replace(qs, '')
            
            if re.search(emoji_pattern, line_without_quotes):
                issues.append(f"Line {i+1}: Emoji outside quotes might cause issues")
    
    # Check for special characters in node IDs
    node_id_pattern = r'^\s*(\w+)\s*\['
    for i, line in enumerate(lines):
        match = re.match(node_id_pattern, line)
        if match:
            node_id = match.group(1)
            if not node_id.replace('_', '').isalnum():
                issues.append(f"Line {i+1}: Node ID '{node_id}' contains special characters")
    
    return issues

def main():
    """Validate all Mermaid diagrams in the docs folder."""
    docs_path = Path("docs")
    
    # Files to check
    files = [
        "azure-architecture-diagram.md",
        "azure-architecture-simplified.md"
    ]
    
    print("🔍 Validating Mermaid Diagrams...\n")
    
    total_errors = 0
    
    for file_name in files:
        file_path = docs_path / file_name
        if not file_path.exists():
            print(f"❌ {file_name}: File not found")
            continue
        
        print(f"📄 Checking {file_name}:")
        diagrams = extract_mermaid_diagrams(file_path)
        
        if not diagrams:
            print("  ⚠️  No Mermaid diagrams found")
            continue
        
        for idx, diagram in enumerate(diagrams):
            print(f"\n  Diagram {idx + 1}:")
            
            # Get first line for identification
            first_line = diagram.strip().split('\n')[0] if diagram.strip() else "Empty"
            print(f"  Type: {first_line}")
            
            # Validate syntax
            errors = validate_mermaid_syntax(diagram)
            issues = check_mermaid_specifics(diagram)
            
            if errors or issues:
                total_errors += len(errors) + len(issues)
                
                if errors:
                    print("  ❌ Syntax Errors:")
                    for error in errors:
                        print(f"    - {error}")
                
                if issues:
                    print("  ⚠️  Potential Issues:")
                    for issue in issues:
                        print(f"    - {issue}")
            else:
                print("  ✅ No errors found")
    
    print(f"\n{'='*50}")
    print(f"Total issues found: {total_errors}")
    
    if total_errors == 0:
        print("✅ All Mermaid diagrams appear to be syntactically correct!")
        print("\nIf you're still experiencing rendering issues, they might be due to:")
        print("- Browser compatibility")
        print("- Mermaid version differences")
        print("- Rendering engine limitations")
        print("- Complex diagram structures")
        print("\nTry using the alternative viewer (view_architecture_alt.py) which uses external services.")
    else:
        print("\n⚠️  Fix the issues above to ensure proper rendering.")

if __name__ == "__main__":
    main()