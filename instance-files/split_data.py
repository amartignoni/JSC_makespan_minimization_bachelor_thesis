import os

def split_file(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    section_count = 1
    i = 0
    while i < len(lines):
        # Detect start of a new section based on the header line
        if lines[i].startswith("Nb of jobs"):
            # Create a new folder for this section
            folder_name = f"Tai{section_count:02}"
            os.makedirs(folder_name, exist_ok=True)
            
            # Extract metadata (first two lines of the section)
            metadata_filename = os.path.join(folder_name, f"{folder_name}_metadata.txt")
            with open(metadata_filename, 'w') as metadata_file:
                metadata_file.writelines(lines[i:i+2])
            
            # Extract problem data (remaining lines of the section)
            problem_filename = os.path.join(folder_name, f"{folder_name}_problem.txt")
            with open(problem_filename, 'w') as problem_file:
                i += 2  # Move past metadata lines
                while i < len(lines) and not lines[i].startswith("Nb of jobs"):
                    problem_file.write(lines[i])
                    i += 1
            
            print(f"Processed section {section_count} into {folder_name}/")
            section_count += 1
        else:
            i += 1

# Run the script on the uploaded file
split_file('all.txt')
