
import unittest
from cbert.data import remove_comments

class TestDataProcessing(unittest.TestCase):

    def test_remove_comments(self):
        c_code_with_comments = """
        /* This is a block comment. */
        int main() { // This is a line comment
            return 0;
        }
        """
        expected_code = """

        int main() { 
            return 0;
        }
        """
        self.assertEqual(remove_comments(c_code_with_comments).strip(), expected_code.strip())

if __name__ == '__main__':
    unittest.main()
