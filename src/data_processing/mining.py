from pydriller import Repository
import os
import csv
import javalang
from typing import List, Tuple
import pandas as pd

def extract_methods_from_java(code: str) -> List[Tuple[str, str]]:
    """
    Extract methods from Java source code using javalang parser.

    Args:
        code (str): The Java source code.

    Returns:
        list: A list of tuples containing method names and their full source code.
    """
    methods = []
    try:
        # Parse the code into an Abstract Syntax Tree (AST)
        tree = javalang.parse.parse(code)
        lines = code.splitlines()

        # Traverse the tree to find method declarations
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            try:
                method_name = node.name
                start_line = node.position.line - 1  # Convert to 0-based indexing
                
                # Find the end of the method by tracking braces
                brace_count = 0
                end_line = start_line
                
                for i, line in enumerate(lines[start_line:], start=start_line):
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0 and '{' in line:
                        continue
                    if brace_count == 0:
                        end_line = i + 1  # Include the closing brace line
                        break
                
                method_code = '\n'.join(lines[start_line:end_line])
                methods.append((method_name, method_code))
                
            except Exception as e:
                print(f"Error extracting method {node.name}: {str(e)}")
                continue

    except Exception as e:
        print(f"Error parsing Java code: {str(e)}")
        
    return methods

def extract_methods_to_csv_from_master(repo_path: str, output_csv: str):
    """
    Extract methods from Java files in the master branch and save them in a CSV file.

    Args:
        repo_path (str): Path to the Git repository.
        output_csv (str): Path to the output CSV file.
    """
    total_methods = 0  # Add counter for total methods
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])

        for commit in Repository(repo_path, only_in_branch="master").traverse_commits():
            print(f"Processing commit: {commit.hash}")

            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".java") and modified_file.source_code:
                    methods = extract_methods_from_java(modified_file.source_code)

                    for method_name, method_code in methods:
                        commit_link = f"{repo_path}/commit/{commit.hash}"
                        csv_writer.writerow([
                            commit.hash,
                            modified_file.filename,
                            method_name,
                            method_code,
                            commit_link
                        ])
                        total_methods += 1  # Increment counter for each method

                    if methods:
                        print(f"Extracted methods from {modified_file.filename} in commit {commit.hash}")

    print(f"\nMining completed:")
    print(f"Total methods extracted: {total_methods}")  # Print final count

if __name__ == "__main__":
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Input/output files
    data_csv = os.path.join(data_dir, "data.csv")
    
    # Read repository list
    df_res = pd.read_csv(data_csv)
    repo_list = [f"https://www.github.com/{name}" for name in df_res['name']]
    
    # Process first repository only (matching notebook)
    if repo_list:
        repo_url = repo_list[0]
        output_csv = os.path.join(data_dir, f"extracted_methods.csv")
        
        print(f"Processing repository: {repo_url}")
        print(f"Results will be saved to: {output_csv}")
        
        # Extract methods from repository
        extract_methods_to_csv_from_master(repo_url, output_csv)
    else:
        print("No repositories found in input file") 