import sys
import json
import ast
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def validate_notebook(ipynb_path: str):
    print(f"Validating notebook: {ipynb_path}")
    path = Path(ipynb_path)
    assert path.exists(), f"Notebook file {ipynb_path} does not exist!"

    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb, "Notebook missing 'cells' attribute!"
    cells = nb["cells"]
    print(f"Total cells found: {len(cells)}")

    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    print(f"Code cells: {len(code_cells)} | Markdown cells: {len(md_cells)}")

    assert len(code_cells) >= 12, f"Expected at least 12 code cells, found {len(code_cells)}"

    for i, cell in enumerate(code_cells, start=1):
        code_text = "".join(cell["source"])
        # Check Python syntax (excluding bash commands with '!' or '%')
        python_lines = []
        for line in code_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("%"):
                continue
            python_lines.append(line)
        cleaned_code = "\n".join(python_lines)

        try:
            ast.parse(cleaned_code)
            print(f" [✓] Code Cell {i:02d}: AST Syntax Valid ({len(python_lines)} lines)")
        except SyntaxError as e:
            print(f" [✗] Code Cell {i:02d}: Syntax Error! Line {e.lineno}: {e.msg}")
            raise

    print("\nALL NOTEBOOK CELLS PASSED AST SYNTAX & STRUCTURAL VALIDATION!")

if __name__ == "__main__":
    validate_notebook(r"C:\Users\prabh\.gemini\antigravity\scratch\video_factory\local_video_factory_colab.ipynb")
