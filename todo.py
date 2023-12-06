#!/usr/bin/env python3
import os

def find_todo_comments(directory, output_file, context_lines=2):
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    with open(output_file, 'w') as outfile:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and file != "todo.py":
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as infile:
                        lines = infile.readlines()

                    for line_number, line in enumerate(lines, start=1):
                        if '# TODO:' in line:
                            start = max(line_number - context_lines - 1, 0)
                            end = min(line_number + context_lines, len(lines))
                            context = ''.join(lines[start:end])

                            outfile.write(f"{20*'==='}\nFile: {file}, Line: {line_number}\n")
                            outfile.write(context)
                            outfile.write(f"{20*'==='}\n\n")

def main():
    project_directory = os.getcwd()
    output_file = os.path.join(os.getcwd(), "todo.txt")
    if os.path.exists(output_file):
        os.remove(output_file)
    find_todo_comments(project_directory, output_file)
    print(f"To-do list compiled in {output_file}")

if __name__ == "__main__":
    main()

