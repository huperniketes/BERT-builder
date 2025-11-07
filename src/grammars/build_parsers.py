#!/usr/bin/env python3
"""
ANTLR Parser Build Script for C-BERT Project
Generates Python parsers from ANTLR grammars
"""

import os
import sys
import subprocess
import urllib.request
import hashlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ANTLR configuration
ANTLR_VERSION = "4.13.2"
ANTLR_JAR_URL = f"https://www.antlr.org/download/antlr-{ANTLR_VERSION}-complete.jar"
ANTLR_JAR_NAME = f"antlr-{ANTLR_VERSION}-complete.jar"

def get_antlr_jar_path():
    """Get the path to the ANTLR jar file"""
    script_dir = Path(__file__).parent
    jar_path = script_dir / "build_tools" / "antlr" / ANTLR_JAR_NAME
    return jar_path

def download_antlr_jar():
    """Download ANTLR jar if not present"""
    jar_path = get_antlr_jar_path()
    jar_path.parent.mkdir(parents=True, exist_ok=True)
    
    if jar_path.exists():
        logger.info(f"ANTLR jar already exists at {jar_path}")
        return jar_path
    
    logger.info(f"Downloading ANTLR {ANTLR_VERSION} from {ANTLR_JAR_URL}")
    try:
        urllib.request.urlretrieve(ANTLR_JAR_URL, jar_path)
        logger.info(f"Successfully downloaded ANTLR jar to {jar_path}")
        return jar_path
except Exception as e:
        logger.error(f"Failed to download ANTLR jar: {e}")
        logger.error("Please download ANTLR manually from https://www.antlr.org/download/")
        logger.error(f"Expected location: {jar_path}")
        raise

def generate_python_parser(grammar_file, output_dir):
    """Generate Python parser from ANTLR grammar"""
    jar_path = download_antlr_jar()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    grammar_path = Path(grammar_file)
    if not grammar_path.exists():
        raise FileNotFoundError(f"Grammar file not found: {grammar_path}")
    
    # Check if Java is available
    try:
        subprocess.run(["java", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Java is not installed or not in PATH. Please install Java to use ANTLR.")
        return False
    
    cmd = [
        "java", "-jar", str(jar_path),
        "-Dlanguage=Python3",
        "-visitor", "-no-listener",
        "-o", str(output_dir),
        str(grammar_path)
    ]
    
    logger.info(f"Generating Python parser from {grammar_path}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully generated parser in {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate parser: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Java not found. Please install Java and ensure it's in your PATH.")
        return False

def validate_generated_files(output_dir):
    """Validate that required parser files were generated"""
    output_dir = Path(output_dir)
    required_files = [
        "CLexer.py",
        "CParser.py", 
        "CVisitor.py",
        "CBaseVisitor.py"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = output_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.error(f"Missing generated files: {missing_files}")
        return False
    
    logger.info("All required parser files generated successfully")
    return True

def create_init_file(output_dir):
    """Create __init__.py file for the generated parsers package"""
    output_dir = Path(output_dir)
    init_file = output_dir / "__init__.py"
    
    if not init_file.exists():
        with open(init_file, 'w') as f:
            f.write('"""Generated ANTLR parsers for C language"""\n')
        logger.info(f"Created __init__.py in {output_dir}")

def main():
    """Main build function"""
    script_dir = Path(__file__).parent
    grammar_file = script_dir / "C.g4"
    output_dir = script_dir / "generated"
    
    logger.info("Starting ANTLR parser generation")
    
    try:
        # Check prerequisites
        if not grammar_file.exists():
            logger.error(f"Grammar file not found: {grammar_file}")
            sys.exit(1)
        
        # Generate Python parser
        if not generate_python_parser(grammar_file, output_dir):
            logger.error("Parser generation failed. Please check the error messages above.")
            sys.exit(1)
        
        # Validate generated files
        if not validate_generated_files(output_dir):
            logger.error("Generated files validation failed.")
            sys.exit(1)
        
        # Create __init__.py
        create_init_file(output_dir)
        
        logger.info("Parser generation completed successfully")
        print(f"Generated parsers are available in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Parser generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
