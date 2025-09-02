#!/usr/bin/env python3
"""
Fix Mermaid diagram syntax issues by properly quoting node labels.
"""

import re
from pathlib import Path

def fix_mermaid_node_labels(content):
    """Fix node labels by properly quoting them."""
    # Pattern to match node definitions with labels containing emojis or <br/>
    # Matches patterns like: NodeID[Label with emoji 🤖 or <br/>]
    pattern = r'(\w+)\[((?:[^\[\]])+)\]'
    
    def fix_label(match):
        node_id = match.group(1)
        label = match.group(2)
        
        # Check if label needs quoting (contains emoji or <br/>)
        needs_quoting = False
        if '<br/>' in label or re.search(r'[\U0001F300-\U0001F9FF]', label):
            needs_quoting = True
        
        # If already quoted, return as is
        if label.startswith('"') and label.endswith('"'):
            return match.group(0)
        if label.startswith("'") and label.endswith("'"):
            return match.group(0)
        
        # Quote if needed
        if needs_quoting:
            # Escape any existing quotes in the label
            label = label.replace('"', '\\"')
            return f'{node_id}["{label}"]'
        
        return match.group(0)
    
    # Process each line
    lines = content.split('\n')
    fixed_lines = []
    in_mermaid = False
    
    for line in lines:
        if line.strip() == '```mermaid':
            in_mermaid = True
            fixed_lines.append(line)
        elif line.strip() == '```' and in_mermaid:
            in_mermaid = False
            fixed_lines.append(line)
        elif in_mermaid:
            # Skip comment lines
            if line.strip().startswith('%%'):
                fixed_lines.append(line)
            else:
                # Fix node labels
                fixed_line = re.sub(pattern, fix_label, line)
                fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def main():
    """Fix Mermaid diagrams in documentation files."""
    docs_path = Path("docs")
    
    files = [
        "azure-architecture-diagram.md",
        "azure-architecture-simplified.md"
    ]
    
    for file_name in files:
        file_path = docs_path / file_name
        if not file_path.exists():
            print(f"❌ {file_name}: File not found")
            continue
        
        print(f"📄 Processing {file_name}...")
        
        # Read content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix content
        fixed_content = fix_mermaid_node_labels(content)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✅ Fixed {file_name}")
    
    print("\n✨ All Mermaid diagrams have been updated!")

if __name__ == "__main__":
    main()