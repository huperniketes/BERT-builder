#!/usr/bin/env python3
"""
Comprehensive Data Quality Analysis for C Repositories

Provides detailed metrics for assessing the quality of C source code repositories
for research purposes, following the standards outlined in GEMINI.md.
"""

import argparse
import os
import re
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict

class FileMetrics:
    """Metrics for individual files."""
    def __init__(self, path, size_bytes, lines_total, lines_code, lines_comments, 
                 lines_blank, functions_count, non_ascii_chars, max_line_length, 
                 has_includes, complexity_indicators):
        self.path = path
        self.size_bytes = size_bytes
        self.lines_total = lines_total
        self.lines_code = lines_code
        self.lines_comments = lines_comments
        self.lines_blank = lines_blank
        self.functions_count = functions_count
        self.non_ascii_chars = non_ascii_chars
        self.max_line_length = max_line_length
        self.has_includes = has_includes
        self.complexity_indicators = complexity_indicators
    
    def to_dict(self):
        return self.__dict__

class QualityMetrics:
    """Overall repository quality metrics."""
    def __init__(self, total_files, total_size_bytes, total_lines, code_lines,
                 comment_ratio, avg_file_size, file_size_distribution, 
                 language_distribution, non_ascii_ratio, duplicate_content_ratio,
                 function_density, complexity_score, issues, warnings):
        self.total_files = total_files
        self.total_size_bytes = total_size_bytes
        self.total_lines = total_lines
        self.code_lines = code_lines
        self.comment_ratio = comment_ratio
        self.avg_file_size = avg_file_size
        self.file_size_distribution = file_size_distribution
        self.language_distribution = language_distribution
        self.non_ascii_ratio = non_ascii_ratio
        self.duplicate_content_ratio = duplicate_content_ratio
        self.function_density = function_density
        self.complexity_score = complexity_score
        self.issues = issues
        self.warnings = warnings
    
    def to_dict(self):
        return self.__dict__

class DataQualityAnalyzer:
    """Analyzes data quality metrics for C repositories."""
    
    C_EXTENSIONS = {'.c', '.h', '.cpp', '.cc', '.cxx', '.hpp', '.hxx'}
    COMMENT_PATTERNS = [
        re.compile(r'//.*$', re.MULTILINE),
        re.compile(r'/\*.*?\*/', re.DOTALL)
    ]
    FUNCTION_PATTERN = re.compile(r'^\s*(?:static\s+)?(?:inline\s+)?(?:\w+\s+)*\w+\s*\*?\s*(\w+)\s*\([^)]*\)\s*{', re.MULTILINE)
    COMPLEXITY_PATTERNS = [
        re.compile(r'\b(if|while|for|switch|case)\b'),
        re.compile(r'\b(goto|break|continue|return)\b'),
        re.compile(r'[?:]'),  # Ternary operators
    ]
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.file_metrics: List[FileMetrics] = []
        self.content_hashes: Dict[str, List[str]] = defaultdict(list)
        
    def analyze(self) -> QualityMetrics:
        """Perform comprehensive quality analysis."""
        print(f"Analyzing repository: {self.repo_path}")
        
        # Collect file metrics
        for file_path in self._find_c_files():
            try:
                metrics = self._analyze_file(file_path)
                self.file_metrics.append(metrics)
                self._track_content_hash(file_path, metrics)
            except Exception as e:
                print(f"Warning: Failed to analyze {file_path}: {e}")
        
        return self._compute_quality_metrics()
    
    def _find_c_files(self) -> List[Path]:
        """Find all C/C++ source files in the repository."""
        files = []
        for root, _, filenames in os.walk(self.repo_path):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in self.C_EXTENSIONS):
                    files.append(Path(root) / filename)
        return files
    
    def _analyze_file(self, file_path: Path) -> FileMetrics:
        """Analyze individual file metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Cannot read file: {e}")
        
        # Basic metrics
        size_bytes = len(content.encode('utf-8'))
        lines = content.split('\n')
        lines_total = len(lines)
        
        # Remove comments for code analysis
        code_content = content
        for pattern in self.COMMENT_PATTERNS:
            code_content = pattern.sub('', code_content)
        
        # Line classification
        lines_code = 0
        lines_comments = 0
        lines_blank = 0
        max_line_length = 0
        
        for line in lines:
            stripped = line.strip()
            max_line_length = max(max_line_length, len(line))
            
            if not stripped:
                lines_blank += 1
            elif stripped.startswith('//') or stripped.startswith('/*') or stripped.endswith('*/'):
                lines_comments += 1
            else:
                lines_code += 1
        
        # Function count
        functions = self.FUNCTION_PATTERN.findall(code_content)
        functions_count = len(functions)
        
        # Non-ASCII character count
        non_ascii_chars = sum(1 for char in content if ord(char) > 127)
        
        # Include detection
        has_includes = '#include' in content
        
        # Complexity indicators
        complexity_indicators = 0
        for pattern in self.COMPLEXITY_PATTERNS:
            complexity_indicators += len(pattern.findall(code_content))
        
        return FileMetrics(
            str(file_path.relative_to(self.repo_path)), size_bytes, lines_total,
            lines_code, lines_comments, lines_blank, functions_count,
            non_ascii_chars, max_line_length, has_includes, complexity_indicators
        )
    
    def _track_content_hash(self, file_path: Path, metrics: FileMetrics):
        """Track content for duplicate detection."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            # Simple hash based on normalized content
            normalized = re.sub(r'\s+', ' ', content).strip()
            content_hash = str(hash(normalized))
            self.content_hashes[content_hash].append(str(file_path))
        except Exception:
            pass  # Skip if cannot read
    
    def _compute_quality_metrics(self) -> QualityMetrics:
        """Compute overall quality metrics."""
        if not self.file_metrics:
            return QualityMetrics(
                0, 0, 0, 0, 0.0, 0.0, {}, {}, 0.0, 0.0, 0.0, 0.0, [], []
            )
        
        # Basic aggregations
        total_files = len(self.file_metrics)
        total_size_bytes = sum(m.size_bytes for m in self.file_metrics)
        total_lines = sum(m.lines_total for m in self.file_metrics)
        code_lines = sum(m.lines_code for m in self.file_metrics)
        comment_lines = sum(m.lines_comments for m in self.file_metrics)
        total_functions = sum(m.functions_count for m in self.file_metrics)
        total_non_ascii = sum(m.non_ascii_chars for m in self.file_metrics)
        total_complexity = sum(m.complexity_indicators for m in self.file_metrics)
        
        # Derived metrics
        comment_ratio = comment_lines / max(total_lines, 1)
        avg_file_size = total_size_bytes / total_files
        non_ascii_ratio = total_non_ascii / max(total_size_bytes, 1)
        function_density = total_functions / max(code_lines, 1) * 1000  # Functions per 1K lines
        complexity_score = total_complexity / max(code_lines, 1) * 100  # Complexity per 100 lines
        
        # File size distribution
        size_ranges = {'<1KB': 0, '1-10KB': 0, '10-100KB': 0, '>100KB': 0}
        for m in self.file_metrics:
            if m.size_bytes < 1024:
                size_ranges['<1KB'] += 1
            elif m.size_bytes < 10240:
                size_ranges['1-10KB'] += 1
            elif m.size_bytes < 102400:
                size_ranges['10-100KB'] += 1
            else:
                size_ranges['>100KB'] += 1
        
        # Language distribution
        lang_dist = Counter()
        for m in self.file_metrics:
            ext = Path(m.path).suffix.lower()
            lang_dist[ext] += 1
        
        # Duplicate detection
        duplicates = sum(1 for files in self.content_hashes.values() if len(files) > 1)
        duplicate_ratio = duplicates / max(total_files, 1)
        
        # Quality issues and warnings
        issues = []
        warnings = []
        
        # Check for quality issues
        if comment_ratio < 0.1:
            issues.append(f"Low comment ratio: {comment_ratio:.2%}")
        if non_ascii_ratio > 0.01:
            warnings.append(f"High non-ASCII character ratio: {non_ascii_ratio:.2%}")
        if duplicate_ratio > 0.05:
            warnings.append(f"Potential duplicate content: {duplicate_ratio:.2%}")
        if avg_file_size > 50000:
            warnings.append(f"Large average file size: {avg_file_size:.0f} bytes")
        
        # Check for files without includes
        files_without_includes = sum(1 for m in self.file_metrics if not m.has_includes and m.path.endswith('.c'))
        if files_without_includes > 0:
            warnings.append(f"{files_without_includes} C files without includes")
        
        # Check for very long lines
        long_lines = sum(1 for m in self.file_metrics if m.max_line_length > 120)
        if long_lines > total_files * 0.1:
            warnings.append(f"{long_lines} files with lines >120 characters")
        
        return QualityMetrics(
            total_files, total_size_bytes, total_lines, code_lines,
            comment_ratio, avg_file_size, size_ranges, dict(lang_dist),
            non_ascii_ratio, duplicate_ratio, function_density, complexity_score,
            issues, warnings
        )

def analyze_file_quality(file_path: str) -> Dict:
    """Analyze quality metrics for a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        return {
            'file_size': 0,
            'comment_ratio': 0,
            'avg_line_length': 0,
            'encoding': 'unknown',
            'error': str(e)
        }
    
    # Basic metrics
    file_size = len(content.encode('utf-8'))
    lines = content.split('\n')
    total_lines = len(lines)
    
    # Comment detection
    comment_lines = 0
    total_line_length = 0
    
    for line in lines:
        total_line_length += len(line)
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.endswith('*/'):
            comment_lines += 1
    
    comment_ratio = comment_lines / max(total_lines, 1)
    avg_line_length = total_line_length / max(total_lines, 1)
    
    return {
        'file_size': file_size,
        'comment_ratio': comment_ratio,
        'avg_line_length': avg_line_length,
        'encoding': 'utf-8'
    }

def format_bytes(bytes_count: int) -> str:
    """Format byte count in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f}TB"

def print_report(metrics: QualityMetrics, detailed: bool = False):
    """Print formatted quality report."""
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📊 OVERVIEW")
    print(f"  Total Files: {metrics.total_files:,}")
    print(f"  Total Size: {format_bytes(metrics.total_size_bytes)}")
    print(f"  Total Lines: {metrics.total_lines:,}")
    print(f"  Code Lines: {metrics.code_lines:,}")
    print(f"  Average File Size: {format_bytes(int(metrics.avg_file_size))}")
    
    print(f"\n📈 QUALITY METRICS")
    print(f"  Comment Ratio: {metrics.comment_ratio:.2%}")
    print(f"  Non-ASCII Ratio: {metrics.non_ascii_ratio:.2%}")
    print(f"  Duplicate Content: {metrics.duplicate_content_ratio:.2%}")
    print(f"  Function Density: {metrics.function_density:.1f}/1K lines")
    print(f"  Complexity Score: {metrics.complexity_score:.1f}/100 lines")
    
    print(f"\n📁 FILE DISTRIBUTION")
    for size_range, count in metrics.file_size_distribution.items():
        percentage = count / metrics.total_files * 100
        print(f"  {size_range}: {count:,} files ({percentage:.1f}%)")
    
    print(f"\n🔤 LANGUAGE DISTRIBUTION")
    for ext, count in sorted(metrics.language_distribution.items()):
        percentage = count / metrics.total_files * 100
        print(f"  {ext}: {count:,} files ({percentage:.1f}%)")
    
    if metrics.issues:
        print(f"\n❌ ISSUES")
        for issue in metrics.issues:
            print(f"  • {issue}")
    
    if metrics.warnings:
        print(f"\n⚠️  WARNINGS")
        for warning in metrics.warnings:
            print(f"  • {warning}")
    
    if not metrics.issues and not metrics.warnings:
        print(f"\n✅ No quality issues detected")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze data quality metrics for C repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_data_quality.py                    # Analyze current directory
  python analyze_data_quality.py /path/to/repo     # Analyze specific repository
  python analyze_data_quality.py --json output.json # Save results as JSON
        """
    )
    parser.add_argument(
        "repo_path", 
        nargs='?', 
        default=os.getcwd(),
        help="Path to repository (default: current directory)"
    )
    parser.add_argument(
        "--json", 
        type=str,
        help="Save detailed results to JSON file"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-file metrics"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.repo_path):
        print(f"Error: Repository path '{args.repo_path}' does not exist")
        sys.exit(1)
    
    analyzer = DataQualityAnalyzer(args.repo_path)
    metrics = analyzer.analyze()
    
    print_report(metrics, args.detailed)
    
    if args.json:
        output_data = {
            'metrics': metrics.to_dict(),
            'file_details': [fm.to_dict() for fm in analyzer.file_metrics] if args.detailed else []
        }
        
        os.makedirs(os.path.dirname(args.json) if os.path.dirname(args.json) else '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Detailed results saved to: {args.json}")

if __name__ == "__main__":
    main()