from pydriller import Repository
import os
import csv
import ast
import pandas as pd
from typing import List, Tuple
import re

def extract_methods_from_python(code: str) -> List[Tuple[str, str]]:
    """Extract methods from Python source code using ast parser."""
    methods = []
    try:
        # Try parsing with Python 3 syntax first
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # If Python 3 parsing fails, try to fix common Python 2 issues
            # Replace print statements with print function calls
            code = re.sub(r'print ([^(].*$)', r'print(\1)', code, flags=re.MULTILINE)
            tree = ast.parse(code)
            
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    method_lines = code.splitlines()[node.lineno-1:node.end_lineno]
                    method_code = '\n'.join(method_lines)
                    methods.append((node.name, method_code))
                except (AttributeError, TypeError):
                    # For older Python versions that don't have end_lineno
                    # Try to extract the function another way
                    method_code = get_function_code(code, node)
                    if method_code:
                        methods.append((node.name, method_code))
    except Exception as e:
        print(f"Error parsing Python code: {e}")
    return methods

def get_function_code(code: str, node: ast.FunctionDef) -> str:
    """Extract function code when end_lineno is not available."""
    try:
        lines = code.splitlines()
        # Get the function definition line
        start_line = node.lineno - 1
        
        # Find the end of the function by indentation
        base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        current_line = start_line + 1
        
        while (current_line < len(lines) and 
               (not lines[current_line].strip() or  # Empty lines
                len(lines[current_line]) - len(lines[current_line].lstrip()) > base_indent)):  # Indented lines
            current_line += 1
            
        return '\n'.join(lines[start_line:current_line])
    except Exception:
        return None

def extract_methods_to_csv(repo_url: str, output_csv: str):
    """Extract methods from a Python repository to CSV."""
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Branch Name", "Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])

        try:
            branch_name = "main"  # Try main first
            repo = Repository(repo_url, only_in_branch=branch_name)
            # Test if branch exists
            next(repo.traverse_commits(), None)
        except Exception:
            branch_name = "master"  # Fallback to master
            repo = Repository(repo_url, only_in_branch=branch_name)

        for commit in repo.traverse_commits():
            print(f"Processing commit: {commit.hash}")

            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".py") and modified_file.source_code:
                    methods = extract_methods_from_python(modified_file.source_code)

                    for method_name, method_code in methods:
                        commit_link = f"{repo_url}/commit/{commit.hash}"
                        csv_writer.writerow([
                            branch_name,
                            commit.hash,
                            modified_file.filename,
                            method_name,
                            method_code,
                            commit_link
                        ])

if __name__ == "__main__":
    df_res = pd.read_csv('.\data\data.csv')
    
    repoList = []
    for idx,row in df_res.iterrows():
        repoList.append("https://www.github.com/{}".format(row['name']))
        repoList[0:5]
    
    for repo in repoList[0:1]:
        # Output to the data folder
        output_csv_file = "./data/extracted_methods.csv"
        # Run the extraction
        extract_methods_to_csv(repo, output_csv_file)

    print(repo)
    