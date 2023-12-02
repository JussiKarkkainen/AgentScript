import os

def find_todo_comments(directory, output_file):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Open the output file
    with open(output_file, 'w') as outfile:
        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if the file is a Python file
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    # Open and read the Python file
                    with open(file_path, 'r') as infile:
                        for line in infile:
                            # Check for TODO comments
                            if line.strip().startswith('# TODO:'):
                                # Write the TODO comment to the output file
                                outfile.write(f"{file_path}: {line}")

def main():
    project_directory = os.getcwd()
    output_file = os.path.join(os.getcwd(), "todo.txt")
    if os.path.exists(output_file):
        os.remove(output_file)
    find_todo_comments(project_directory, output_file)
    print(f"To-do list compiled in {output_file}")

if __name__ == "__main__":
    main()

