import json

with open("Module3_Pretrained_Models.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

with open("Module3_Pretrained_Models.py", "w", encoding="utf-8") as f:
    for cell in notebook.get("cells", []):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            f.write(source + "\n\n")
