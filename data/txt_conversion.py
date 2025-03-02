import csv
from pathlib import Path
import os
import argparse

def text_to_csv(input_txt: str, output_csv: str):
    """
    Convert a text file of Java methods to a CSV with column 'Method Code No Comments'.
    Assumes one method per line; adjust splitting logic if needed.
    """
    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_txt, 'r', encoding='utf-8') as txt_file, \
         open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(["Method Code No Comments"])
        
        # Read text file and write each line as a row
        for line in txt_file:
            line = line.strip()
            if line:  # Skip empty lines
                writer.writerow([line])

    print(f"Converted {input_txt} to {output_csv}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Convert a text file to CSV format for the n-gram model')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('--output', '-o', default='data/processed_methods2.csv', 
                        help='Path to the output CSV file (default: data/processed_methods2.csv)')
    
    args = parser.parse_args()
    
    # Convert the file
    print(f"Converting {args.input_file} to {args.output}")
    text_to_csv(args.input_file, args.output)