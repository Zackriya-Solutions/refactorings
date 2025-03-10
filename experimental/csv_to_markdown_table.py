import csv

def csv_to_markdown_table(input_file, output_file):
    # List to store table rows
    table_lines = []
    
    # Open and read the CSV file with UTF-8 encoding
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # Read the header row
        header = next(reader)
        # Add the header row to the table
        table_lines.append('| ' + ' | '.join(header) + ' |')
        # Add the separator row
        table_lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
        # Add each data row
        for row in reader:
            table_lines.append('| ' + ' | '.join(row) + ' |')
    
    # Write the table to the output file
    with open(output_file, 'w', encoding='utf-8') as mdfile:
        mdfile.write('\n'.join(table_lines))

# Usage: Convert 'data.csv' to 'results.md'
csv_to_markdown_table('data.csv', 'results.md')