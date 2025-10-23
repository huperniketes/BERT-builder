"""
Semantic preservation validator for C-BERT preprocessing pipeline.
Ensures code transformations maintain semantic integrity.
"""

import re
import ast
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SemanticIssue:
    """Represents a semantic preservation issue."""
    severity: str  # 'critical', 'warning', 'info'
    message: str
    location: Optional[str] = None
    original_snippet: Optional[str] = None
    processed_snippet: Optional[str] = None

class SemanticPreservationValidator:
    """Validates semantic preservation during code preprocessing."""
    
    def __init__(self):
        self.c_keywords = {
            'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
            'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
            'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof',
            'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void',
            'volatile', 'while'
        }
        
        self.operators = {
            '++', '--', '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', 
            '<=', '>=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', 
            '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='
        }
    
    def validate_preprocessing(self, original: str, processed: str) -> List[SemanticIssue]:
        """Validate that preprocessing preserves semantic meaning."""
        issues = []
        
        # Check function signatures
        issues.extend(self._check_function_signatures(original, processed))
        
        # Check variable declarations
        issues.extend(self._check_variable_declarations(original, processed))
        
        # Check control flow structures
        issues.extend(self._check_control_flow(original, processed))
        
        # Check string literals
        issues.extend(self._check_string_literals(original, processed))
        
        # Check macro definitions
        issues.extend(self._check_macros(original, processed))
        
        # Check operator preservation
        issues.extend(self._check_operators(original, processed))
        
        return issues
    
    def _check_function_signatures(self, original: str, processed: str) -> List[SemanticIssue]:
        """Check that function signatures are preserved."""
        issues = []
        
        # Extract function signatures
        func_pattern = r'(\w+\s+)+(\w+)\s*\([^)]*\)\s*{'
        
        orig_funcs = set(re.findall(func_pattern, original))
        proc_funcs = set(re.findall(func_pattern, processed))
        
        missing_funcs = orig_funcs - proc_funcs
        for func in missing_funcs:
            issues.append(SemanticIssue(
                severity='critical',
                message=f"Function signature lost during preprocessing: {func}",
                original_snippet=str(func)
            ))
        
        return issues
    
    def _check_variable_declarations(self, original: str, processed: str) -> List[SemanticIssue]:
        """Check that variable declarations are preserved."""
        issues = []
        
        # Pattern for variable declarations
        var_pattern = r'((?:const\s+)?(?:static\s+)?(?:extern\s+)?(?:int|char|float|double|void|long|short|unsigned|signed)\s*\*?\s+\w+(?:\s*=\s*[^;]+)?;)'
        
        orig_vars = set(re.findall(var_pattern, original))
        proc_vars = set(re.findall(var_pattern, processed))
        
        missing_vars = orig_vars - proc_vars
        for var in missing_vars:
            issues.append(SemanticIssue(
                severity='warning',
                message=f"Variable declaration potentially modified: {var}",
                original_snippet=var
            ))
        
        return issues
    
    def _check_control_flow(self, original: str, processed: str) -> List[SemanticIssue]:
        """Check that control flow structures are preserved."""
        issues = []
        
        control_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'break', 'continue', 'return']
        
        for keyword in control_keywords:
            orig_count = len(re.findall(rf'\b{keyword}\b', original))
            proc_count = len(re.findall(rf'\b{keyword}\b', processed))
            
            if orig_count != proc_count:
                issues.append(SemanticIssue(
                    severity='critical',
                    message=f"Control flow keyword '{keyword}' count mismatch: {orig_count} -> {proc_count}",
                ))
        
        return issues
    
    def _check_string_literals(self, original: str, processed: str) -> List[SemanticIssue]:
        """Check that string literals are preserved."""
        issues = []
        
        # Extract string literals
        string_pattern = r'"(?:[^"\\]|\\.)*"'
        
        orig_strings = re.findall(string_pattern, original)
        proc_strings = re.findall(string_pattern, processed)
        
        if len(orig_strings) != len(proc_strings):
            issues.append(SemanticIssue(
                severity='warning',
                message=f"String literal count mismatch: {len(orig_strings)} -> {len(proc_strings)}"
            ))
        
        # Check for modified strings
        orig_set = set(orig_strings)
        proc_set = set(proc_strings)
        
        missing_strings = orig_set - proc_set
        for string in missing_strings:
            issues.append(SemanticIssue(
                severity='warning',
                message=f"String literal potentially modified: {string}",
                original_snippet=string
            ))
        
        return issues
    
    def _check_macros(self, original: str, processed: str) -> List[SemanticIssue]:
        """Check that macro definitions are preserved."""
        issues = []
        
        # Extract #define macros
        macro_pattern = r'#define\s+\w+(?:\([^)]*\))?\s+.*'
        
        orig_macros = set(re.findall(macro_pattern, original))
        proc_macros = set(re.findall(macro_pattern, processed))
        
        missing_macros = orig_macros - proc_macros
        for macro in missing_macros:
            issues.append(SemanticIssue(
                severity='critical',
                message=f"Macro definition lost: {macro}",
                original_snippet=macro
            ))
        
        return issues
    
    def _check_operators(self, original: str, processed: str) -> List[SemanticIssue]:
        """Check that operators are preserved."""
        issues = []
        
        for op in self.operators:
            # Escape special regex characters
            escaped_op = re.escape(op)
            orig_count = len(re.findall(escaped_op, original))
            proc_count = len(re.findall(escaped_op, processed))
            
            if orig_count != proc_count and abs(orig_count - proc_count) > 1:  # Allow small variations
                issues.append(SemanticIssue(
                    severity='warning',
                    message=f"Operator '{op}' count significantly changed: {orig_count} -> {proc_count}"
                ))
        
        return issues
    
    def _extract_identifiers(self, code: str) -> Set[str]:
        """Extract all identifiers from C code."""
        # Pattern for C identifiers
        identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers = set(re.findall(identifier_pattern, code))
        
        # Remove C keywords
        identifiers -= self.c_keywords
        
        return identifiers
    
    def check_identifier_preservation(self, original: str, processed: str) -> List[SemanticIssue]:
        """Check that important identifiers are preserved."""
        issues = []
        
        orig_identifiers = self._extract_identifiers(original)
        proc_identifiers = self._extract_identifiers(processed)
        
        missing_identifiers = orig_identifiers - proc_identifiers
        
        # Filter out likely unimportant identifiers (single letters, common names)
        important_missing = {
            id for id in missing_identifiers 
            if len(id) > 2 and not id.startswith('_')
        }
        
        for identifier in important_missing:
            issues.append(SemanticIssue(
                severity='warning',
                message=f"Important identifier potentially lost: {identifier}"
            ))
        
        return issues