"""
Enhanced semantic preservation tests for C-BERT preprocessing.
"""

import unittest
from src.cbert.data import preprocess_code
from src.cbert.semantic_validator import SemanticPreservationValidator

class TestSemanticPreservation(unittest.TestCase):
    """Test semantic preservation during preprocessing."""
    
    def setUp(self):
        self.validator = SemanticPreservationValidator()
    
    def test_function_signature_preservation(self):
        """Test that function signatures are preserved."""
        code = """
        int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
        """
        
        processed = preprocess_code(code)
        issues = self.validator.validate_preprocessing(code, processed)
        
        # Should preserve function name and structure
        self.assertIn('factorial', processed)
        self.assertIn('int n', processed)
        
        # No critical semantic issues
        critical_issues = [i for i in issues if i.severity == 'critical']
        self.assertEqual(len(critical_issues), 0, f"Critical issues found: {critical_issues}")
    
    def test_string_literal_preservation(self):
        """Test that string literals are preserved exactly."""
        code = '''
        #include <stdio.h>
        int main() {
            printf("Hello, World!\\n");
            char *msg = "Test string with \\"quotes\\"";
            return 0;
        }
        '''
        
        processed = preprocess_code(code)
        issues = self.validator.validate_preprocessing(code, processed)
        
        # String literals should be preserved
        self.assertIn('"Hello, World!\\n"', processed)
        self.assertIn('"Test string with \\"quotes\\""', processed)
        
        # Check for string preservation issues
        string_issues = [i for i in issues if 'string' in i.message.lower()]
        self.assertEqual(len(string_issues), 0, f"String preservation issues: {string_issues}")
    
    def test_macro_preservation(self):
        """Test that macro definitions are preserved."""
        code = """
        #define MAX_SIZE 1024
        #define MIN(a, b) ((a) < (b) ? (a) : (b))
        
        int buffer[MAX_SIZE];
        """
        
        processed = preprocess_code(code)
        issues = self.validator.validate_preprocessing(code, processed)
        
        # Macros should be preserved
        self.assertIn('#define MAX_SIZE', processed)
        self.assertIn('#define MIN(a, b)', processed)
        
        # Check for macro issues
        macro_issues = [i for i in issues if 'macro' in i.message.lower()]
        self.assertEqual(len(macro_issues), 0, f"Macro preservation issues: {macro_issues}")
    
    def test_control_flow_preservation(self):
        """Test that control flow structures are preserved."""
        code = """
        int process_array(int *arr, int size) {
            for (int i = 0; i < size; i++) {
                if (arr[i] < 0) {
                    continue;
                }
                if (arr[i] > 100) {
                    break;
                }
                switch (arr[i] % 3) {
                    case 0:
                        return 1;
                    case 1:
                        break;
                    default:
                        continue;
                }
            }
            return 0;
        }
        """
        
        processed = preprocess_code(code)
        issues = self.validator.validate_preprocessing(code, processed)
        
        # Control flow keywords should be preserved
        control_keywords = ['for', 'if', 'continue', 'break', 'switch', 'case', 'default', 'return']
        for keyword in control_keywords:
            self.assertIn(keyword, processed, f"Control flow keyword '{keyword}' missing")
        
        # Check for control flow issues
        control_issues = [i for i in issues if 'control flow' in i.message.lower()]
        self.assertEqual(len(control_issues), 0, f"Control flow issues: {control_issues}")
    
    def test_comment_removal_safety(self):
        """Test that comment removal doesn't break code structure."""
        code = """
        int main() {
            /* This is a block comment */
            int x = 42; // This is a line comment
            char *str = "This /* is not */ a comment";
            // Another comment
            return x;
        }
        """
        
        processed = preprocess_code(code)
        
        # Code structure should be preserved
        self.assertIn('int main()', processed)
        self.assertIn('int x = 42;', processed)
        self.assertIn('return x;', processed)
        
        # String with comment-like content should be preserved
        self.assertIn('"This /* is not */ a comment"', processed)
        
        # Comments should be removed
        self.assertNotIn('This is a block comment', processed)
        self.assertNotIn('This is a line comment', processed)
        self.assertNotIn('Another comment', processed)
    
    def test_operator_preservation(self):
        """Test that operators are preserved correctly."""
        code = """
        int calculate(int a, int b) {
            int result = a + b * 2;
            result += 10;
            result -= a % b;
            return (result > 0) ? result : -result;
        }
        """
        
        processed = preprocess_code(code)
        issues = self.validator.validate_preprocessing(code, processed)
        
        # Operators should be preserved
        operators = ['+', '*', '+=', '-=', '%', '>', '?', ':']
        for op in operators:
            self.assertIn(op, processed, f"Operator '{op}' missing")
        
        # Check for operator issues
        operator_issues = [i for i in issues if 'operator' in i.message.lower()]
        # Allow minor operator count variations
        critical_op_issues = [i for i in operator_issues if 'significantly' in i.message]
        self.assertEqual(len(critical_op_issues), 0, f"Critical operator issues: {critical_op_issues}")
    
    def test_complex_code_preservation(self):
        """Test semantic preservation on complex code."""
        code = """
        #include <stdlib.h>
        #define BUFFER_SIZE 256
        
        typedef struct {
            int id;
            char name[BUFFER_SIZE];
        } Person;
        
        Person* create_person(int id, const char* name) {
            /* Allocate memory for person */
            Person* p = malloc(sizeof(Person));
            if (!p) return NULL; // Memory allocation failed
            
            p->id = id;
            strncpy(p->name, name, BUFFER_SIZE - 1);
            p->name[BUFFER_SIZE - 1] = '\\0'; // Ensure null termination
            
            return p;
        }
        """
        
        processed = preprocess_code(code)
        issues = self.validator.validate_preprocessing(code, processed)
        
        # Key elements should be preserved
        self.assertIn('typedef struct', processed)
        self.assertIn('Person* create_person', processed)
        self.assertIn('malloc(sizeof(Person))', processed)
        self.assertIn('strncpy', processed)
        
        # Critical semantic elements preserved
        critical_issues = [i for i in issues if i.severity == 'critical']
        self.assertLessEqual(len(critical_issues), 1, f"Too many critical issues: {critical_issues}")

if __name__ == '__main__':
    unittest.main()