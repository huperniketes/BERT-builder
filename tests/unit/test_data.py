
import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from cbert.data import remove_comments, process_file, main

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_test_file(self, filename, content, encoding='utf-8'):
        file_path = os.path.join(self.test_dir, filename)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return file_path

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

        # Test with no comments
        self.assertEqual(remove_comments("int x = 1;"), "int x = 1;")
        # Test with only block comment
        self.assertEqual(remove_comments("/* only comment */").strip(), "")
        # Test with only line comment
        self.assertEqual(remove_comments("// only comment").strip(), "")

    def test_process_file_valid_c_code(self):
        file_path = self._create_test_file("test.c", "int main() { /* comment */ return 0; }")
        cleaned_text = process_file(file_path)
        self.assertEqual(cleaned_text.strip(), "int main() {  return 0; }")

    def test_process_file_invalid_utf8(self):
        # Create a file with invalid UTF-8 sequence (e.g., a lone continuation byte)
        file_path = os.path.join(self.test_dir, "invalid.c")
        with open(file_path, 'wb') as f:
            f.write(b"int main() { \x80 return 0; }") # \x80 is an invalid start byte in UTF-8
        
        # process_file should replace invalid chars, not raise error
        cleaned_text = process_file(file_path)
        self.assertIn('\ufffd', cleaned_text) # Check for replacement character
        self.assertIn("int main() { \ufffd return 0; }", cleaned_text)

    def test_process_file_with_null_bytes(self):
        file_path = self._create_test_file("null.c", "int main() { \x00 return 0; }")
        cleaned_text = process_file(file_path)
        self.assertNotIn('\x00', cleaned_text) # Null bytes should be removed
        self.assertEqual(cleaned_text.strip(), "int main() {  return 0; }")

    @patch('builtins.print') # Mock print to capture warnings
    def test_process_file_non_ascii_warning(self, mock_print):
        file_path = self._create_test_file("non_ascii.c", "int main() { char *s = \"résumé\"; return 0; }")
        cleaned_text = process_file(file_path)
        self.assertIn('\ufffd', cleaned_text) # Invalid UTF-8 will be replaced
        mock_print.assert_called_with(self.assertRegex(mock_print.call_args[0][0], r"Warning: Non-ASCII characters found in.*"))

    def test_process_file_only_comments(self):
        file_path = self._create_test_file("comments_only.c", "/* comment */ // another")
        cleaned_text = process_file(file_path)
        self.assertEqual(cleaned_text.strip(), "")

    @patch('cbert.data.process_file')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('argparse.ArgumentParser')
    def test_main_filters_empty_lines(self, mock_argparse, mock_open, mock_makedirs, mock_process_file):
        # Mock argparse to control input_dir and output_file
        mock_args = MagicMock()
        mock_args.input_dir = os.path.join(self.test_dir, "input")
        mock_args.output_file = os.path.join(self.test_dir, "output.txt")
        mock_argparse.return_value.parse_args.return_value = mock_args

        os.makedirs(mock_args.input_dir, exist_ok=True)
        self._create_test_file("file1.c", "int x = 1;")
        self._create_test_file("file2.c", "/* only comments */") # This will result in empty cleaned_text
        self._create_test_file("file3.h", "#define FOO")

        # Mock os.walk to return our test files
        with patch('os.walk', return_value=[(mock_args.input_dir, [], ["file1.c", "file2.c", "file3.h"])]):
            # Mock process_file to return specific cleaned content
            mock_process_file.side_effect = [
                "int x = 1;", # from file1.c
                "",           # from file2.c (empty after comments removed)
                "#define FOO" # from file3.h
            ]
            
            # Mock the actual file writing
            mock_outfile = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_outfile

            main()

            # Assert that outfile.write was called only for non-empty content
            calls = mock_outfile.write.call_args_list
            self.assertEqual(len(calls), 4) # 2 content lines + 2 newlines
            self.assertEqual(calls[0].args[0], "int x = 1;")
            self.assertEqual(calls[1].args[0], "\n")
            self.assertEqual(calls[2].args[0], "#define FOO")
            self.assertEqual(calls[3].args[0], "\n")
            self.assertNotIn("", [call.args[0] for call in calls]) # Ensure no empty strings were written

    @patch('cbert.data.process_file')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('argparse.ArgumentParser')
    def test_main_writes_valid_content(self, mock_argparse, mock_open, mock_makedirs, mock_process_file):
        mock_args = MagicMock()
        mock_args.input_dir = os.path.join(self.test_dir, "input")
        mock_args.output_file = os.path.join(self.test_dir, "output.txt")
        mock_argparse.return_value.parse_args.return_value = mock_args

        os.makedirs(mock_args.input_dir, exist_ok=True)
        self._create_test_file("file1.c", "int a = 1;")
        self._create_test_file("file2.h", "#include <stdio.h>")

        with patch('os.walk', return_value=[(mock_args.input_dir, [], ["file1.c", "file2.h"])]):
            mock_process_file.side_effect = [
                "int a = 1;",
                "#include <stdio.h>"
            ]
            mock_outfile = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_outfile

            main()

            calls = mock_outfile.write.call_args_list
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0].args[0], "int a = 1;")
            self.assertEqual(calls[2].args[0], "#include <stdio.h>")

if __name__ == '__main__':
    unittest.main()
