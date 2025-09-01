#!/usr/bin/env python3
"""
Knowledge Base Refinement Module

This module provides functionality to analyze and optimize knowledge graphs
stored in Neo4j using LLM-powered insights and graph analysis techniques.
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class ReportGenerator:
    """Generates formatted reports for knowledge graph analysis"""
    
    def __init__(self, base_filename: str = "graph_analysis_report"):
        self.base_filename = base_filename
        self.timestamp = datetime.now()
        self.report_data = {}
        
    def get_versioned_filename(self, extension: str) -> str:
        """Generate a versioned filename with timestamp"""
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{self.base_filename}_{timestamp_str}.{extension}"
    
    def format_table(self, headers: List[str], rows: List[List[str]], title: str = "") -> str:
        """Format data as a nice ASCII table"""
        if not rows:
            return f"\n{title}\nNo data available.\n"
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Create table
        lines = []
        if title:
            lines.append(f"\n{title}")
            lines.append("=" * len(title))
        
        # Header
        header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
        lines.append(header_line)
        lines.append("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
        
        # Rows
        for row in rows:
            row_line = "| " + " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)) + " |"
            lines.append(row_line)
        
        return "\n".join(lines) + "\n"
    
    def format_markdown_table(self, headers: List[str], rows: List[List[str]], title: str = "") -> str:
        """Format data as a markdown table"""
        if not rows:
            return f"\n## {title}\n\nNo data available.\n\n"
        
        lines = []
        if title:
            lines.append(f"\n## {title}\n")
        
        # Header
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        
        # Rows
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return "\n".join(lines) + "\n\n"
    
    def generate_markdown_report(self, refiner: 'KnowledgeGraphRefiner') -> str:
        """Generate a comprehensive markdown report"""
        lines = [
            f"# Knowledge Graph Analysis Report",
            f"",
            f"**Generated on:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Analysis Type:** Comprehensive Graph Structure Analysis",
            f"",
            f"---",
            f""
        ]
        
        # Graph Statistics
        stats = refiner.get_graph_statistics()
        lines.extend([
            f"## 📊 Graph Overview",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Nodes | {stats['nodes']:,} |",
            f"| Total Relationships | {stats['relationships']:,} |",
            f"",
            f"---",
            f""
        ])
        
        # Node Labels
        node_labels = refiner.analyze_node_labels(20)
        if node_labels:
            headers = ["Rank", "Node Label", "Count"]
            rows = [[str(i), result['label'], f"{result['count']:,}"] 
                   for i, result in enumerate(node_labels, 1)]
            lines.append(self.format_markdown_table(headers, rows, "🏷️ Top 20 Node Labels"))
        
        # Relationship Types
        rel_types = refiner.analyze_relationship_types(20)
        if rel_types:
            headers = ["Rank", "Relationship Type", "Count"]
            rows = [[str(i), result['relationship_type'], f"{result['count']:,}"] 
                   for i, result in enumerate(rel_types, 1)]
            lines.append(self.format_markdown_table(headers, rows, "🔗 Top 20 Relationship Types"))
        
        # Outbound Relationships
        outbound = refiner.analyze_outbound_relationships(10)
        if outbound:
            headers = ["Rank", "Entity Name", "Entity Type", "Outbound Count"]
            rows = [[str(i), result['entity_name'] or 'Unknown', result['entity_type'], f"{result['outbound_count']:,}"] 
                   for i, result in enumerate(outbound, 1)]
            lines.append(self.format_markdown_table(headers, rows, "📤 Top 10 Entities with Most Outbound Relationships"))
        
        # Inbound Relationships
        inbound = refiner.analyze_inbound_relationships(10)
        if inbound:
            headers = ["Rank", "Entity Name", "Entity Type", "Inbound Count"]
            rows = [[str(i), result['entity_name'] or 'Unknown', result['entity_type'], f"{result['inbound_count']:,}"] 
                   for i, result in enumerate(inbound, 1)]
            lines.append(self.format_markdown_table(headers, rows, "📥 Top 10 Entities with Most Inbound Relationships"))
        
        # Common Relationships
        common_rels = refiner.analyze_common_relationships(15)
        if common_rels:
            headers = ["Rank", "Source Type", "Relationship", "Target Type", "Count"]
            rows = [[str(i), result['source_type'], result['relationship_type'], result['target_type'], f"{result['count']:,}"] 
                   for i, result in enumerate(common_rels, 1)]
            lines.append(self.format_markdown_table(headers, rows, "🔄 Most Common Relationships Between Entity Types"))
        
        # Orphaned Nodes
        orphaned = refiner.analyze_orphaned_nodes()
        if orphaned:
            total_orphaned = sum(result['count'] for result in orphaned)
            lines.extend([
                f"## 👤 Orphaned Nodes Analysis",
                f"",
                f"**Total Orphaned Nodes:** {total_orphaned:,}",
                f""
            ])
            
            if orphaned:
                headers = ["Node Type", "Count"]
                rows = [[result['node_type'], f"{result['count']:,}"] for result in orphaned]
                lines.append(self.format_markdown_table(headers, rows, "Orphaned Nodes by Type"))
        else:
            lines.extend([
                f"## 👤 Orphaned Nodes Analysis",
                f"",
                f"✅ No orphaned nodes found - all nodes have relationships.",
                f""
            ])
        
        # Graph Metrics
        metrics = refiner.calculate_graph_metrics()
        lines.extend([
            f"## 🌐 Graph Connectivity Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Graph Density | {metrics['density_percentage']:.4f}% |",
            f"| Average Relationships per Node | {metrics['average_relationships_per_node']:.2f} |",
            f"",
            f"---",
            f""
        ])
        
        return "\n".join(lines)
    
    def generate_html_report(self, markdown_content: str) -> str:
        """Generate HTML report from markdown content"""
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
            border-left: 2px solid #3498db;
            padding-left: 10px;
            font-size: 1.2em;
        }}
        h4, h5, h6 {{
            color: #34495e;
            margin-top: 15px;
            border-left: 1px solid #3498db;
            padding-left: 8px;
            font-size: 1.1em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .metric-value {{
            font-weight: bold;
            color: #2980b9;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
        }}
        hr {{
            border: none;
            height: 2px;
            background: linear-gradient(to right, #3498db, #e74c3c);
            margin: 30px 0;
        }}
        .emoji {{
            font-size: 1.2em;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        {self._markdown_to_html(markdown_content)}
        <div class="footer">
            <p>Generated by Atlas Knowledge Graph Refiner on {self.timestamp.strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
        return html_template
    
    def _markdown_to_html(self, markdown: str) -> str:
        """Improved markdown to HTML converter"""
        import re
        
        lines = markdown.split('\n')
        html_lines = []
        in_table = False
        table_header_processed = False
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                    table_header_processed = False
                html_lines.append('')
                i += 1
                continue
            
            # Headers - support multiple levels (check longer prefixes first)
            if line.startswith('###### '):
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                    table_header_processed = False
                html_lines.append(f'<h6>{line[7:].strip()}</h6>')
                i += 1
                continue
            elif line.startswith('##### '):
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                    table_header_processed = False
                html_lines.append(f'<h5>{line[6:].strip()}</h5>')
                i += 1
                continue
            elif line.startswith('#### '):
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                    table_header_processed = False
                html_lines.append(f'<h4>{line[5:].strip()}</h4>')
                i += 1
                continue
            elif line.startswith('### '):
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                    table_header_processed = False
                html_lines.append(f'<h3>{line[4:].strip()}</h3>')
                i += 1
                continue
            elif line.startswith('## '):
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                    table_header_processed = False
                html_lines.append(f'<h2>{line[3:].strip()}</h2>')
                i += 1
                continue
            elif line.startswith('# '):
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                    table_header_processed = False
                html_lines.append(f'<h1>{line[2:].strip()}</h1>')
                i += 1
                continue
            
            # Horizontal rules
            elif line == '---':
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                    table_header_processed = False
                html_lines.append('<hr>')
                i += 1
                continue
            
            # Tables
            elif '|' in line and line.strip().startswith('|'):
                # Check if this is a table separator line
                if re.match(r'^\|[\s\-\|]+\|$', line):
                    i += 1
                    continue
                
                if not in_table:
                    html_lines.append('<table>')
                    in_table = True
                    table_header_processed = False
                
                # Extract cells
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                
                # Determine if this is a header row (first row of table)
                tag = 'th' if not table_header_processed else 'td'
                if not table_header_processed:
                    table_header_processed = True
                
                # Process each cell for bold text
                processed_cells = []
                for cell in cells:
                    # Handle bold text
                    cell = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell)
                    processed_cells.append(cell)
                
                row_html = '<tr>' + ''.join(f'<{tag}>{cell}</{tag}>' for cell in processed_cells) + '</tr>'
                html_lines.append(row_html)
                i += 1
                continue
            
            # Regular text (with bold formatting)
            else:
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                    table_header_processed = False
                
                # Process bold text
                processed_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
                
                # If it's not empty after processing, wrap in paragraph
                if processed_line.strip():
                    html_lines.append(f'<p>{processed_line}</p>')
                else:
                    html_lines.append('')
                
                i += 1
                continue
        
        # Close any remaining table
        if in_table:
            html_lines.append('</table>')
        
        # Join and clean up
        html = '\n'.join(html_lines)
        
        # Clean up extra empty paragraphs and spacing
        html = re.sub(r'<p>\s*</p>', '', html)
        html = re.sub(r'\n\s*\n\s*\n+', '\n\n', html)
        
        return html
    
    def save_reports(self, refiner: 'KnowledgeGraphRefiner', llm_recommendations: Optional[str] = None):
        """Save both markdown and HTML reports"""
        # Generate markdown content
        markdown_content = self.generate_markdown_report(refiner)
        
        # Add LLM recommendations if available
        if llm_recommendations:
            markdown_content += f"\n## 🤖 LLM Optimization Recommendations\n\n{llm_recommendations}\n"
        
        # Save markdown report
        md_filename = self.get_versioned_filename("md")
        md_path = Path(md_filename)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Generate and save HTML report
        html_content = self.generate_html_report(markdown_content)
        html_filename = self.get_versioned_filename("html")
        html_path = Path(html_filename)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return md_path, html_path


class KnowledgeGraphRefiner:
    """
    A class to analyze and refine knowledge graphs with comprehensive
    graph structure analysis and LLM-powered optimization recommendations.
    """
    
    def __init__(self, graph_store, llm_model=None):
        """
        Initialize the Knowledge Graph Refiner
        
        Args:
            graph_store: Neo4j graph store instance
            llm_model: Optional LLM model for generating recommendations
        """
        self.graph_store = graph_store
        self.llm_model = llm_model
    
    def get_graph_statistics(self) -> Dict[str, int]:
        """Get basic graph statistics"""
        return self.graph_store.get_stats()
    
    def analyze_node_labels(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Analyze node labels in the graph
        
        Args:
            limit: Maximum number of labels to return
            
        Returns:
            List of dictionaries with label and count information
        """
        query = """
        MATCH (n)
        RETURN labels(n)[0] as label, count(*) as count
        ORDER BY count DESC
        LIMIT $limit
        """
        
        return self.graph_store.graph.query(query, {"limit": limit})
    
    def analyze_relationship_types(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Analyze relationship types in the graph
        
        Args:
            limit: Maximum number of relationship types to return
            
        Returns:
            List of dictionaries with relationship type and count information
        """
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(*) as count
        ORDER BY count DESC
        LIMIT $limit
        """
        
        return self.graph_store.graph.query(query, {"limit": limit})
    
    def analyze_outbound_relationships(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find entities with the most outbound relationships
        
        Args:
            limit: Maximum number of entities to return
            
        Returns:
            List of dictionaries with entity information and outbound counts
        """
        query = """
        MATCH (n)-[r]->()
        RETURN n.name as entity_name, labels(n)[0] as entity_type, count(r) as outbound_count
        ORDER BY outbound_count DESC
        LIMIT $limit
        """
        
        return self.graph_store.graph.query(query, {"limit": limit})
    
    def analyze_inbound_relationships(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find entities with the most inbound relationships
        
        Args:
            limit: Maximum number of entities to return
            
        Returns:
            List of dictionaries with entity information and inbound counts
        """
        query = """
        MATCH ()-[r]->(n)
        RETURN n.name as entity_name, labels(n)[0] as entity_type, count(r) as inbound_count
        ORDER BY inbound_count DESC
        LIMIT $limit
        """
        
        return self.graph_store.graph.query(query, {"limit": limit})
    
    def analyze_common_relationships(self, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Find the most common relationships between entity types
        
        Args:
            limit: Maximum number of relationship patterns to return
            
        Returns:
            List of dictionaries with relationship pattern information
        """
        query = """
        MATCH (a)-[r]->(b)
        RETURN labels(a)[0] as source_type, type(r) as relationship_type, 
               labels(b)[0] as target_type, count(*) as count
        ORDER BY count DESC
        LIMIT $limit
        """
        
        return self.graph_store.graph.query(query, {"limit": limit})
    
    def analyze_orphaned_nodes(self) -> List[Dict[str, Any]]:
        """
        Find orphaned nodes (nodes with no relationships)
        
        Returns:
            List of dictionaries with orphaned node type and count information
        """
        query = """
        MATCH (n)
        WHERE NOT (n)-[]-()
        RETURN labels(n)[0] as node_type, count(*) as count
        ORDER BY count DESC
        """
        
        return self.graph_store.graph.query(query)
    
    def calculate_graph_metrics(self) -> Dict[str, float]:
        """
        Calculate various graph connectivity metrics
        
        Returns:
            Dictionary with graph metrics
        """
        # Get basic counts
        total_nodes_query = "MATCH (n) RETURN count(n) as total_nodes"
        total_rels_query = "MATCH ()-[r]->() RETURN count(r) as total_relationships"
        
        total_nodes = self.graph_store.graph.query(total_nodes_query)[0]['total_nodes']
        total_rels = self.graph_store.graph.query(total_rels_query)[0]['total_relationships']
        
        # Calculate metrics
        max_possible_rels = total_nodes * (total_nodes - 1) if total_nodes > 1 else 0
        density = (total_rels / max_possible_rels) * 100 if max_possible_rels > 0 else 0
        avg_rels_per_node = (total_rels * 2) / total_nodes if total_nodes > 0 else 0
        
        return {
            'total_nodes': total_nodes,
            'total_relationships': total_rels,
            'density_percentage': density,
            'average_relationships_per_node': avg_rels_per_node
        }
    
    def analyze_connected_components(self) -> Optional[List[Dict[str, Any]]]:
        """
        Analyze connected components in the graph (requires Neo4j GDS)
        
        Returns:
            List of component information or None if GDS not available
        """
        query = """
        CALL gds.wcc.stream('myGraph', {})
        YIELD nodeId, componentId
        RETURN componentId, count(*) as component_size
        ORDER BY component_size DESC
        LIMIT 10
        """
        
        try:
            return self.graph_store.graph.query(query)
        except Exception:
            return None
    
    def get_sample_relationships(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get a sample of relationships for LLM analysis
        
        Args:
            limit: Maximum number of relationships to sample
            
        Returns:
            List of sample relationship information
        """
        query = """
        MATCH (n)-[r]->(m)
        RETURN labels(n)[0] as source_type, type(r) as rel_type, labels(m)[0] as target_type, 
               n.name as source_name, m.name as target_name
        LIMIT $limit
        """
        
        return self.graph_store.graph.query(query, {"limit": limit})
    
    async def generate_llm_recommendations(self) -> Optional[str]:
        """
        Generate optimization recommendations using LLM
        
        Returns:
            LLM-generated recommendations or None if LLM not available
        """
        if not self.llm_model:
            return None
        
        # Gather data for LLM analysis
        stats = self.get_graph_statistics()
        sample_data = self.get_sample_relationships(50)
        
        # Create prompt for LLM
        prompt = f"""
        You are a knowledge graph optimization expert. Analyze the following graph structure and provide actionable recommendations.

        Graph Statistics:
        - Total Nodes: {stats['nodes']:,}
        - Total Relationships: {stats['relationships']:,}
        
        Sample Relationships:
        {chr(10).join([f"({r['source_type']}:{r['source_name']}) -[{r['rel_type']}]-> ({r['target_type']}:{r['target_name']})" for r in sample_data[:10]])}
        
        Please provide:
        1. Data Quality Issues: Identify potential duplicates, inconsistencies, or missing relationships
        2. Schema Optimization: Suggest improvements to node labels and relationship types
        3. Performance Optimization: Recommend indexing strategies and query optimizations
        4. Graph Structure Improvements: Suggest ways to reduce orphaned nodes and improve connectivity
        5. Semantic Enhancements: Recommend additional relationships or properties to add
        
        Format your response with clear headings and actionable bullet points.
        """
        
        try:
            from langchain_core.messages import HumanMessage
            response = await self.llm_model.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"❌ Error generating LLM recommendations: {str(e)}")
            return None
    
    def display_node_labels(self, limit: int = 20, report_gen: Optional[ReportGenerator] = None):
        """Display top node labels with improved formatting"""
        top_labels = self.analyze_node_labels(limit)
        
        if report_gen:
            headers = ["Rank", "Node Label", "Count"]
            rows = [[str(i), result['label'], f"{result['count']:,}"] 
                   for i, result in enumerate(top_labels, 1)]
            print(report_gen.format_table(headers, rows, f"🏷️ Top {limit} Node Labels"))
        else:
            print(f"\n🏷️  Top {limit} Node Labels:")
            for i, result in enumerate(top_labels, 1):
                print(f"   {i:2d}. {result['label']}: {result['count']:,} nodes")
    
    def display_relationship_types(self, limit: int = 20, report_gen: Optional[ReportGenerator] = None):
        """Display top relationship types with improved formatting"""
        top_rels = self.analyze_relationship_types(limit)
        
        if report_gen:
            headers = ["Rank", "Relationship Type", "Count"]
            rows = [[str(i), result['relationship_type'], f"{result['count']:,}"] 
                   for i, result in enumerate(top_rels, 1)]
            print(report_gen.format_table(headers, rows, f"🔗 Top {limit} Relationship Types"))
        else:
            print(f"\n🔗 Top {limit} Relationship Types:")
            for i, result in enumerate(top_rels, 1):
                print(f"   {i:2d}. {result['relationship_type']}: {result['count']:,} relationships")
    
    def display_outbound_relationships(self, limit: int = 10, report_gen: Optional[ReportGenerator] = None):
        """Display entities with most outbound relationships with improved formatting"""
        outbound_results = self.analyze_outbound_relationships(limit)
        
        if report_gen:
            headers = ["Rank", "Entity Name", "Entity Type", "Outbound Count"]
            rows = [[str(i), result['entity_name'] or 'Unknown', result['entity_type'], f"{result['outbound_count']:,}"] 
                   for i, result in enumerate(outbound_results, 1)]
            print(report_gen.format_table(headers, rows, f"📤 Top {limit} Entities with Most Outbound Relationships"))
        else:
            print(f"\n📤 Top {limit} Entities with Most Outbound Relationships:")
            for i, result in enumerate(outbound_results, 1):
                entity_name = result['entity_name'] or 'Unknown'
                print(f"   {i:2d}. {entity_name} ({result['entity_type']}): {result['outbound_count']:,} outbound")
    
    def display_inbound_relationships(self, limit: int = 10, report_gen: Optional[ReportGenerator] = None):
        """Display entities with most inbound relationships with improved formatting"""
        inbound_results = self.analyze_inbound_relationships(limit)
        
        if report_gen:
            headers = ["Rank", "Entity Name", "Entity Type", "Inbound Count"]
            rows = [[str(i), result['entity_name'] or 'Unknown', result['entity_type'], f"{result['inbound_count']:,}"] 
                   for i, result in enumerate(inbound_results, 1)]
            print(report_gen.format_table(headers, rows, f"📥 Top {limit} Entities with Most Inbound Relationships"))
        else:
            print(f"\n📥 Top {limit} Entities with Most Inbound Relationships:")
            for i, result in enumerate(inbound_results, 1):
                entity_name = result['entity_name'] or 'Unknown'
                print(f"   {i:2d}. {entity_name} ({result['entity_type']}): {result['inbound_count']:,} inbound")
    
    def display_common_relationships(self, limit: int = 15, report_gen: Optional[ReportGenerator] = None):
        """Display most common relationships between entity types with improved formatting"""
        common_rels = self.analyze_common_relationships(limit)
        
        if report_gen:
            headers = ["Rank", "Source Type", "Relationship", "Target Type", "Count"]
            rows = [[str(i), result['source_type'], result['relationship_type'], result['target_type'], f"{result['count']:,}"] 
                   for i, result in enumerate(common_rels, 1)]
            print(report_gen.format_table(headers, rows, f"🔄 Most Common Relationships Between Entity Types"))
        else:
            print(f"\n🔄 Most Common Relationships Between Entity Types:")
            for i, result in enumerate(common_rels, 1):
                print(f"   {i:2d}. {result['source_type']} -[{result['relationship_type']}]-> {result['target_type']}: {result['count']:,}")
    
    def display_orphaned_nodes(self, report_gen: Optional[ReportGenerator] = None):
        """Display orphaned nodes analysis with improved formatting"""
        orphaned_results = self.analyze_orphaned_nodes()
        total_orphaned = sum(result['count'] for result in orphaned_results)
        
        if report_gen:
            if total_orphaned > 0:
                headers = ["Node Type", "Count"]
                rows = [[result['node_type'], f"{result['count']:,}"] for result in orphaned_results]
                print(report_gen.format_table(headers, rows, f"👤 Orphaned Nodes Analysis (Total: {total_orphaned:,})"))
            else:
                print(f"\n👤 Orphaned Nodes Analysis")
                print("=" * len("👤 Orphaned Nodes Analysis"))
                print("✅ No orphaned nodes found - all nodes have relationships\n")
        else:
            print(f"\n👤 Orphaned Nodes Analysis:")
            if total_orphaned > 0:
                print(f"   Total Orphaned Nodes: {total_orphaned:,}")
                print(f"   Orphaned Nodes by Type:")
                for result in orphaned_results:
                    print(f"     • {result['node_type']}: {result['count']:,} nodes")
            else:
                print(f"   ✅ No orphaned nodes found - all nodes have relationships")
    
    def display_graph_metrics(self, report_gen: Optional[ReportGenerator] = None):
        """Display graph connectivity metrics with improved formatting"""
        metrics = self.calculate_graph_metrics()
        components = self.analyze_connected_components()
        
        if report_gen:
            headers = ["Metric", "Value"]
            rows = [
                ["Graph Density", f"{metrics['density_percentage']:.4f}%"],
                ["Average Relationships per Node", f"{metrics['average_relationships_per_node']:.2f}"]
            ]
            if components:
                rows.extend([
                    ["Connected Components", str(len(components))],
                    ["Largest Component Size", f"{components[0]['component_size']:,} nodes"]
                ])
            else:
                rows.append(["Connected Components", "Analysis requires Neo4j GDS library"])
            
            print(report_gen.format_table(headers, rows, "🌐 Graph Connectivity Metrics"))
        else:
            print(f"\n🌐 Graph Connectivity Metrics:")
            print(f"   Graph Density: {metrics['density_percentage']:.4f}%")
            print(f"   Average Relationships per Node: {metrics['average_relationships_per_node']:.2f}")
            
            if components:
                print(f"   Connected Components: {len(components)}")
                print(f"   Largest Component Size: {components[0]['component_size']:,} nodes")
            else:
                print(f"   Connected Components: Analysis requires Neo4j GDS library")
    
    async def display_llm_recommendations(self):
        """Display LLM-generated optimization recommendations"""
        print(f"\n🤖 Generating Optimization Recommendations...")
        
        recommendations = await self.generate_llm_recommendations()
        
        if recommendations:
            print(f"\n📋 LLM Optimization Recommendations:")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(recommendations)
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            print(f"💡 LLM recommendations not available - continuing with basic analysis...")
    
    async def run_comprehensive_analysis(self, save_reports: bool = True):
        """Run comprehensive graph analysis with optional report generation"""
        print(f"\n📈 Analyzing Graph Structure...")
        
        # Create report generator for improved formatting
        report_gen = ReportGenerator()
        
        # Display all analysis sections with improved formatting
        self.display_node_labels(20, report_gen)
        self.display_relationship_types(20, report_gen)
        self.display_outbound_relationships(10, report_gen)
        self.display_inbound_relationships(10, report_gen)
        self.display_common_relationships(15, report_gen)
        self.display_orphaned_nodes(report_gen)
        self.display_graph_metrics(report_gen)
        
        # Generate LLM recommendations if available
        llm_recommendations = None
        if self.llm_model:
            print(f"\n🤖 Generating Optimization Recommendations...")
            llm_recommendations = await self.generate_llm_recommendations()
            
            if llm_recommendations:
                print(f"\n📋 LLM Optimization Recommendations:")
                print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(llm_recommendations)
                print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            else:
                print(f"💡 LLM recommendations not available - continuing with basic analysis...")
        
        # Save reports if requested
        if save_reports:
            print(f"\n📄 Generating Reports...")
            try:
                md_path, html_path = report_gen.save_reports(self, llm_recommendations)
                print(f"✅ Reports saved successfully:")
                print(f"   📝 Markdown: {md_path}")
                print(f"   🌐 HTML: {html_path}")
            except Exception as e:
                print(f"⚠️ Error saving reports: {str(e)}")
        
        return report_gen


async def refine_command(args):
    """
    Main entry point for the refine command
    
    Args:
        args: Command line arguments
    """
    # Check if required arguments are set
    if not args.llm_provider:
        print("❌ Error: No LLM provider specified. Use --llm-provider or set DEFAULT_LLM_PROVIDER in environment")
        sys.exit(1)
    if not args.model:
        print("❌ Error: No model specified. Use --model or set DEFAULT_LLM_MODEL in environment")
        sys.exit(1)
    
    print(f"🔧 Knowledge Graph Refinement")
    print(f"  LLM Provider: {args.llm_provider}")
    print(f"  Model: {args.model}")
    
    # Import validation function from atlas
    from atlas import validate_provider_config, initialize_llm
    validate_provider_config(args.llm_provider)
    
    # Initialize Neo4j graph store
    try:
        from atlas import Neo4jGraphStore
        graph_store = Neo4jGraphStore()
        print(f"✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("💡 Please ensure Neo4j is running and connection details are correct")
        sys.exit(1)
    
    # Get initial graph statistics
    initial_stats = graph_store.get_stats()
    print(f"\n📊 Current Graph Statistics:")
    print(f"   Total Nodes: {initial_stats['nodes']:,}")
    print(f"   Total Relationships: {initial_stats['relationships']:,}")
    
    if initial_stats['nodes'] == 0:
        print("⚠️ No nodes found in the graph. Please run 'analyze' command first to generate knowledge graph.")
        sys.exit(1)
    
    print(f"\n🔍 Starting Knowledge Graph Analysis...")
    
    # Initialize LLM for optimization suggestions
    llm_model = initialize_llm(args.llm_provider, args.model)
    
    # Create refiner instance and run analysis
    refiner = KnowledgeGraphRefiner(graph_store, llm_model)
    await refiner.run_comprehensive_analysis()
    
    print(f"\n✅ Knowledge Graph Analysis Complete!")