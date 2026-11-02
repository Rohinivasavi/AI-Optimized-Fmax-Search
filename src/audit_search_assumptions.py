
import ast
from pathlib import Path

FILES = [
    "src/binary_search.py",
    "src/ml_windowed_search.py",
    "src/reference_document_search.py",
    "src/compare_grid_alignment_fixed.py"
]


def main():

    print("=" * 70)
    print("SEARCH ALGORITHM ASSUMPTION AUDIT")
    print("=" * 70)

    for file_path in FILES:

        path = Path(file_path)

        print(f"\nFILE: {file_path}")
        print("-" * 50)

        if not path.exists():
            print("File not found. Skipping.")
            continue

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        true_fmax_usage = []
        pass_fail_functions = []

        for node in ast.walk(tree):

            if isinstance(node, ast.Name):
                if node.id == "true_fmax":
                    true_fmax_usage.append(node.lineno)

            if isinstance(node, ast.FunctionDef):
                if any(word in node.name.lower()
                       for word in ["pass", "search", "binary"]):
                    pass_fail_functions.append(node.name)

        print("Functions related to search:",
              pass_fail_functions)

        if true_fmax_usage:
            print("Uses true_fmax at lines:",
                  sorted(set(true_fmax_usage)))
            print("Ground-truth dependency: YES")
        else:
            print("Ground-truth dependency: NOT DETECTED")

    print("\nAudit completed.")


if __name__ == "__main__":
    main()
