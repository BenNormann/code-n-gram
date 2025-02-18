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
        # Parse Java code
        tree = javalang.parse.parse(code)
        lines = code.splitlines()
        
        # Helper function to extract method text from line numbers
        def get_method_text(start_pos, end_pos):
            # javalang positions are 1-based
            return '\n'.join(lines[start_pos-1:end_pos])

        # Walk through all class declarations
        for _, class_decl in tree.filter(javalang.tree.ClassDeclaration):
            # Extract all method declarations in the class
            for method in class_decl.methods:
                try:
                    method_name = method.name
                    # Get method position
                    start_line = method.position[0] if method.position else None
                    # Find the end of the method by looking for the closing brace
                    if start_line:
                        # Find method body
                        method_text = get_method_text(start_line, start_line + 20)  # Get enough lines to find method end
                        brace_count = 0
                        end_line = start_line
                        
                        for i, line in enumerate(lines[start_line-1:], start=start_line):
                            brace_count += line.count('{') - line.count('}')
                            if brace_count == 0 and '{' in line:
                                continue
                            if brace_count == 0:
                                end_line = i
                                break
                        
                        method_code = get_method_text(start_line, end_line)
                        methods.append((method_name, method_code))
                except Exception as e:
                    print(f"Error extracting method {method.name}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error parsing Java code: {e}")
    return methods

def extract_methods_to_csv(repo_url: str, output_csv: str, single_repo: bool = True):
    """
    Extract methods from Java files in a repository and save them in a CSV file.

    Args:
        repo_url (str): URL or path to the Git repository.
        output_csv (str): Path to the output CSV file.
        single_repo (bool): If True, process single repo. If False, treat repo_url as CSV containing repos.
    """
    if not single_repo:
        # Read repository list from CSV
        df_repos = pd.read_csv(repo_url)
        repo_list = [f"https://www.github.com/{name}" for name in df_repos['name']]
        repo_list = repo_list  # Process all repositories
    else:
        repo_list = [repo_url]

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

        for repo_url in repo_list:
            print(f"\nProcessing repository: {repo_url}")
            
            try:
                # Try main branch first
                branch_name = "main"
                repo = Repository(repo_url, only_in_branch=branch_name)
                next(repo.traverse_commits(), None)
            except Exception:
                # Fallback to master branch
                branch_name = "master"
                repo = Repository(repo_url, only_in_branch=branch_name)

            methods_found = 0
            for commit in repo.traverse_commits():
                print(f"Processing commit: {commit.hash}")

                for modified_file in commit.modified_files:
                    if modified_file.filename.endswith(".java") and modified_file.source_code:
                        try:
                            methods = extract_methods_from_java(modified_file.source_code)

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
    
    # Input and output file paths
    data_csv = os.path.join(data_dir, "data.csv")
    output_csv = os.path.join(data_dir, "extracted_methods.csv")
    
    if not os.path.exists(data_csv):
        print(f"Error: Input file not found: {data_csv}")
        exit(1)
        
    print(f"Reading repositories from: {data_csv}")
    print(f"Results will be saved to: {output_csv}")
    
    # Extract methods from repositories listed in data.csv
    extract_methods_to_csv(data_csv, output_csv, single_repo=False) 