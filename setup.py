from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import subprocess
import sys
import os
from pathlib import Path

class CustomBuildPy(build_py):
    """Custom build command that generates ANTLR parsers before building"""
    
    def run(self):
        # Generate parsers before building
        script_dir = Path(__file__).parent / "src" / "grammars"
        build_script = script_dir / "build_parsers.py"
        
        if build_script.exists():
            print("Generating ANTLR parsers...")
            try:
                result = subprocess.run([sys.executable, str(build_script)], 
                                     check=True, capture_output=True, text=True)
                print("ANTLR parsers generated successfully")
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate ANTLR parsers: {e}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                sys.exit(1)
        else:
            print("Warning: build_parsers.py not found, skipping parser generation")
        
        super().run()

setup(
    name="cbert",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmdclass={"build_py": CustomBuildPy},
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers", 
        "sentencepiece",
        "clang",
        "requests",
        "numpy<2.0",
        "antlr4-python3-runtime==4.13.2"
    ],
    extras_require={
        "dev": [
            "pytest"
        ]
    }
)
