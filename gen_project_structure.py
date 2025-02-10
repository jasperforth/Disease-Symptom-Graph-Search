from pathlib import Path

# Define exclusion rules
EXCLUDED_DIRECTORIES = {
    'venv', 'node_modules', 'dist', 'database_lock', 'store_lock', 'dbms', 
    'transactions', 'system', 'playground'
}
EXCLUDED_PATTERNS = [
    'neostore', 'index-', 'checkpoint', '.archive.dump'
]

# Define specific directory paths to exclude
EXCLUDED_PATHS = [
    'neo4j_data/databases/neo4j'
]

def should_exclude(path: Path) -> bool:
    """
    Determines whether a file or directory should be excluded based on name, patterns, and specific paths.
    
    Args:
        path (Path): The file or directory path.
    
    Returns:
        bool: True if the path should be excluded, False otherwise.
    """
    # Exclude hidden files and directories, directories in EXCLUDED_DIRECTORIES, and files matching patterns
    if path.name.startswith('.') or path.name.startswith('_'):
        return True
    if path.is_dir() and path.name in EXCLUDED_DIRECTORIES:
        return True
    if any(str(path).endswith(excluded_path) for excluded_path in EXCLUDED_PATHS):
        return True
    for pattern in EXCLUDED_PATTERNS:
        if pattern in path.name:
            return True
    return False

def generate_project_structure(root_dir: Path, output_file: Path):
    """
    Generates a tree-structured text file containing the project structure,
    excluding hidden files, directories, and certain patterns.
    
    Args:
        root_dir (Path): The root directory of the project.
        output_file (Path): The output text file path.
    """
    def write_structure(current_path: Path, indent: str, file, is_last: bool):
        if should_exclude(current_path):
            return

        # Determine branch symbol
        branch = '└── ' if is_last else '├── '

        # Write directory or file name
        if current_path.is_dir():
            file.write(f"{indent}{branch}{current_path.name}/\n")
            # Recursively write contents of the directory
            children = sorted(current_path.iterdir())
            for i, child in enumerate(children):
                new_indent = indent + ("    " if is_last else "│   ")
                write_structure(child, new_indent, file, i == len(children) - 1)
        else:
            file.write(f"{indent}{branch}{current_path.name}\n")

    with output_file.open('w') as file:
        file.write(f"{root_dir.name}/\n")  # Write the project root
        children = sorted(root_dir.iterdir())
        for i, child in enumerate(children):
            write_structure(child, "", file, i == len(children) - 1)

if __name__ == "__main__":
    # Define the root directory and output file path
    project_root = Path(__file__).parent
    output_file = project_root / "project_structure.txt"

    # Generate the project structure file
    generate_project_structure(project_root, output_file)

    print(f"Project structure has been saved to {output_file}")
