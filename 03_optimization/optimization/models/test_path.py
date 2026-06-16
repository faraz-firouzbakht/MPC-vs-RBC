from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
print(f"My current file is: {Path(__file__).resolve()}")
print(f"And 3 parents up is: {project_root}")