import pandas as pd
import os
import csv
import javalang
from typing import List, Tuple
from datetime import datetime
from pydriller import Repository
import sys

MAX_METHODS = 25000  # Global parameter for maximum methods to extract

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
            method_name = node.name

            # Determine the start and end lines of the method
            start_line = node.position.line - 1
            end_line = None

            # Use the body of the method to determine its end position
            if node.body:
                last_statement = node.body[-1]
                if hasattr(last_statement, 'position') and last_statement.position:
                    end_line = last_statement.position.line

            # Extract method code
            if end_line:
                method_code = "\n".join(lines[start_line:end_line+1])
            else:
                # If end_line couldn't be determined, extract up to the end of the file
                method_code = "\n".join(lines[start_line:])

            methods.append((method_name, method_code))

    except Exception as e:
        print(f"Error parsing Java code: {e}")
    return methods

def extract_methods_to_csv(repo_data: pd.Series, output_csv: str, method_count: int = 0) -> int:
    """
    Extract methods from Java files in a repository and save them to CSV.
    Returns the total number of methods extracted.
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Only write headers if file is empty
        if os.path.getsize(output_csv) == 0:
            csv_writer.writerow([
                "Branch Name",
                "Commit Hash",
                "File Name",
                "Method Name",
                "Method Code",
                "Commit Link"
            ])

        repo_name = repo_data['name']
        default_branch = repo_data['defaultBranch']
        repo_url = f"https://github.com/{repo_name}"
        
        try:
            print(f"Processing repository: {repo_name}")
            
            for commit in Repository(repo_url, only_in_branch=default_branch).traverse_commits():
                print(f"Processing commit: {commit.hash}")
                
                for modified_file in commit.modified_files:
                    if modified_file.filename.endswith(".java") and modified_file.source_code:
                        methods = extract_methods_from_java(modified_file.source_code)
                        
                        for method_name, method_code in methods:
                            commit_link = f"{repo_url}/commit/{commit.hash}"
                            csv_writer.writerow([
                                default_branch,
                                commit.hash,
                                modified_file.filename,
                                method_name,
                                method_code,
                                commit_link
                            ])
                            method_count += 1
                            print(f"Methods extracted: {method_count}")
                            
                            if method_count >= MAX_METHODS:
                                print(f"\nReached limit of {MAX_METHODS} methods")
                                return method_count
                        
                        print(f"Extracted methods from {modified_file.filename} in commit {commit.hash}")

        except Exception as e:
            print(f"Error processing repository {repo_name}: {e}")

        return method_count

def main():
    # Load the repository data
    df = pd.read_csv('./data/initialdata.csv')
    
    # Create output file in the data directory
    output_csv = "./data/extracted_methods.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        pass  # Create/clear the file
    
    # Filter for Java repositories
    java_repos = df[df['mainLanguage'] == 'Java']
    
    # Process repositories until we hit the method limit
    total_methods = 0
    for _, repo in java_repos.iterrows():
        total_methods = extract_methods_to_csv(repo, output_csv, total_methods)
        if total_methods >= MAX_METHODS:
            print(f"Finished: Extracted {total_methods} methods")
            break
    
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
