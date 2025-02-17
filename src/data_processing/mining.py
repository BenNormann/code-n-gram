from pydriller import Repository
import os
import csv
import ast
from typing import List, Tuple
import pandas as pd

def extract_methods_from_python(code: str) -> List[Tuple[str, str]]:
    """
    Extract methods from Python source code using ast parser.

    Args:
        code (str): The Python source code.

    Returns:
        list: A list of tuples containing method names and their full source code.
    """
    methods = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    # Get the source code lines for this function
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
    """
    Extract methods from Python files in a repository and save them in a CSV file.

    Args:
        repo_url (str): URL or path to the Git repository.
        output_csv (str): Path to the output CSV file.
    """
    print(f"Mining repository: {repo_url}")
    print(f"Output will be saved to: {output_csv}")
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "Repository",
            "Branch Name", 
            "Commit Hash", 
            "File Name", 
            "Method Name", 
            "Method Code", 
            "Commit Link"
        ])

        try:
            # Try main branch first
            branch_name = "main"
            repo = Repository(repo_url, only_in_branch=branch_name)
            # Test if branch exists
            next(repo.traverse_commits(), None)
        except Exception:
            # Fallback to master branch
            branch_name = "master"
            repo = Repository(repo_url, only_in_branch=branch_name)

        methods_found = 0
        for commit in repo.traverse_commits():
            print(f"Processing commit: {commit.hash}")

            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".py") and modified_file.source_code:
                    try:
                        methods = extract_methods_from_python(modified_file.source_code)

                        for method_name, method_code in methods:
                            commit_link = f"{repo_url}/commit/{commit.hash}"
                            csv_writer.writerow([
                                repo_url,
                                branch_name, 
                                commit.hash, 
                                modified_file.filename, 
                                method_name, 
                                method_code, 
                                commit_link
                            ])
                            methods_found += 1

                        if methods:
                            print(f"Extracted {len(methods)} methods from {modified_file.filename}")

                    except Exception as e:
                        print(f"Error processing file {modified_file.filename}: {e}")
                        continue

        print(f"\nMining completed:")
        print(f"Total methods extracted: {methods_found}")
        print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Set up the output file path
    output_csv = os.path.join(data_dir, "extracted_methods.csv")
    
    # Use the current project directory as the repository to analyze
    repo_path = project_root
    
    # Extract methods
    extract_methods_to_csv(repo_path, output_csv) 