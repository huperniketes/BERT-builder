"""
C Language AST Evaluator using ANTLR
Provides token-level AST node labeling for C code
"""

import sys
import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# Add generated parsers to path
script_dir = Path(__file__).parent.parent / "grammars"
generated_dir = script_dir / "generated"
if generated_dir.exists():
    sys.path.insert(0, str(generated_dir))

try:
    from CLexer import CLexer
    from CParser import CParser
    from CVisitor import CVisitor
    from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker, Vocabulary
except ImportError as e:
    logging.warning(f"ANTLR C parsers not available: {e}")
    CLexer = CParser = CVisitor = None

logger = logging.getLogger(__name__)

class CASTNodeMapper(CVisitor):
    """Maps C AST nodes to token labels"""
    
    def __init__(self):
        super().__init__()
        self.token_labels = []
        self.current_tokens = []
        
    def get_c_label_map(self) -> Dict[str, int]:
        """Get C-specific AST node label mapping"""
        return {
            # Function-related
            'FunctionDefinition': 0,
            'FunctionDeclaration': 1,
            'Parameter': 2,
            'FunctionCall': 3,
            
            # Variable-related
            'VariableDeclaration': 4,
            'VariableAssignment': 5,
            'VariableReference': 6,
            
            # Type-related
            'TypeSpecifier': 7,
            'Typedef': 8,
            'StructDefinition': 9,
            'UnionDefinition': 10,
            'EnumDefinition': 11,
            
            # Control flow
            'IfStatement': 12,
            'ElseStatement': 13,
            'SwitchStatement': 14,
            'CaseStatement': 15,
            
            # Loops
            'ForLoop': 16,
            'WhileLoop': 17,
            'DoWhileLoop': 18,
            
            # Jump statements
            'ReturnStatement': 19,
            'BreakStatement': 20,
            'ContinueStatement': 21,
            'GotoStatement': 22,
            
            # Expressions
            'BinaryExpression': 23,
            'UnaryExpression': 24,
            'TernaryExpression': 25,
            'AssignmentExpression': 26,
            
            # Literals
            'IntegerLiteral': 27,
            'FloatLiteral': 28,
            'CharacterLiteral': 29,
            'StringLiteral': 30,
            
            # Preprocessor
            'IncludeDirective': 31,
            'DefineDirective': 32,
            'ConditionalDirective': 33,
            
            # Other
            'Comment': 34,
            'Other': 35
        }
    
    def visitTerminal(self, node):
        """Visit terminal nodes and assign labels"""
        token = node.symbol
        if token:
            token_text = token.text
            token_type = token.type
            
            # Map token to AST node type
            label = self._map_token_to_label(token_text, token_type)
            self.token_labels.append(label)
    
    def _map_token_to_label(self, token_text: str, token_type: int) -> int:
        """Map individual token to AST label"""
        label_map = self.get_c_label_map()
        
        # Handle keywords
        if token_text in ['int', 'char', 'float', 'double', 'void', 'long', 'short', 'unsigned', 'signed']:
            return label_map['TypeSpecifier']
        elif token_text == 'typedef':
            return label_map['Typedef']
        elif token_text == 'struct':
            return label_map['StructDefinition']
        elif token_text == 'union':
            return label_map['UnionDefinition']
        elif token_text == 'enum':
            return label_map['EnumDefinition']
        elif token_text == 'if':
            return label_map['IfStatement']
        elif token_text == 'else':
            return label_map['ElseStatement']
        elif token_text == 'switch':
            return label_map['SwitchStatement']
        elif token_text == 'case':
            return label_map['CaseStatement']
        elif token_text == 'for':
            return label_map['ForLoop']
        elif token_text == 'while':
            return label_map['WhileLoop']
        elif token_text == 'do':
            return label_map['DoWhileLoop']
        elif token_text == 'return':
            return label_map['ReturnStatement']
        elif token_text == 'break':
            return label_map['BreakStatement']
        elif token_text == 'continue':
            return label_map['ContinueStatement']
        elif token_text == 'goto':
            return label_map['GotoStatement']
        elif token_text == '#include':
            return label_map['IncludeDirective']
        elif token_text == '#define':
            return label_map['DefineDirective']
        elif token_text in ['#if', '#ifdef', '#ifndef', '#else', '#endif']:
            return label_map['ConditionalDirective']
        
        # Handle literals
        elif token_text.startswith('"') and token_text.endswith('"'):
            return label_map['StringLiteral']
        elif token_text.startswith("'") and token_text.endswith("'"):
            return label_map['CharacterLiteral']
        elif token_text.replace('_', '').replace('.', '').isdigit():
            return label_map['IntegerLiteral']
        elif token_text.replace('_', '').replace('.', '').replace('f', '').replace('F', '').replace('l', '').replace('L', '').isdigit():
            return label_map['FloatLiteral']
        
        # Handle operators
        elif token_text in ['+', '-', '*', '/', '%', '<<', '>>', '&', '|', '^']:
            return label_map['BinaryExpression']
        elif token_text in ['++', '--', '!', '~', '&', '*', '+', '-']:
            return label_map['UnaryExpression']
        elif token_text in ['=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=']:
            return label_map['AssignmentExpression']
        elif token_text == '?':
            return label_map['TernaryExpression']
        
        # Handle identifiers (function calls, variable references)
        elif token_text.replace('_', '').isalnum():
            # This is a simplified heuristic - in practice, you'd need more context
            return label_map['VariableReference']
        
        # Handle comments
        elif token_text.startswith('//') or token_text.startswith('/*'):
            return label_map['Comment']
        
        # Default
        else:
            return label_map['Other']

class CASTEvaluator:
    """C Language AST Evaluator"""
    
    def __init__(self):
        if CLexer is None:
            raise ImportError("ANTLR C parsers not available. Run build_parsers.py first.")
        
        self.label_map = CASTNodeMapper().get_c_label_map()
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def parse_c_code(self, code: str) -> Optional[CParser.CompilationUnitContext]:
        """Parse C code using ANTLR"""
        try:
            input_stream = InputStream(code)
            lexer = CLexer(input_stream)
            token_stream = CommonTokenStream(lexer)
            parser = CParser(token_stream)
            
            # Parse compilation unit
            tree = parser.compilationUnit()
            
            return tree
        except Exception as e:
            logger.error(f"Failed to parse C code: {e}")
            return None
    
    def get_ast_labels(self, code: str) -> List[int]:
        """Get token-level AST labels for C code"""
        try:
            tree = self.parse_c_code(code)
            if tree is None:
                return [self.label_map['Other']] * len(code.split())
            
            # Create visitor and walk the tree
            mapper = CASTNodeMapper()
            walker = ParseTreeWalker()
            walker.walk(mapper, tree)
            
            return mapper.token_labels if mapper.token_labels else [self.label_map['Other']]
            
        except Exception as e:
            logger.error(f"Failed to get AST labels: {e}")
            return [self.label_map['Other']] * len(code.split())
    
    def get_label_map(self) -> Dict[str, int]:
        """Get C AST label mapping"""
        return self.label_map.copy()
    
    def get_num_classes(self) -> int:
        """Get number of AST label classes"""
        return len(self.label_map)

def is_c_available() -> bool:
    """Check if C AST evaluation is available"""
    try:
        from CLexer import CLexer
        from CParser import CParser
        return True
    except ImportError:
        return False