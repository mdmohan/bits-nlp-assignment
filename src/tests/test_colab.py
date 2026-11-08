import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import yaml

from lm.colab import create_notebook
from lm.preparation import file_sha256


class ColabNotebookTests(unittest.TestCase):
    def test_generated_notebook_matches_bundle_and_compiles(self):
        root = Path(__file__).resolve().parents[2]
        config = {
            "name": "inaugural_convergence_v1",
            "corpora": {"inaugural": {}},
            "budgets_percent": [10, 50, 100],
            "training": {"epochs": 20, "patience": 3},
            "models": [
                {"family": "neural", "final_seeds": [42, 43, 44]},
                {"family": "neural", "final_seeds": [42, 43, 44]},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("input/suite.yaml", yaml.safe_dump(config))
            output = Path(temporary) / "followup.ipynb"
            create_notebook(SimpleNamespace(root=root), bundle, output)
            notebook = json.loads(output.read_text())
            settings = {}
            exec("".join(notebook["cells"][1]["source"]), settings)
            self.assertEqual(settings["EXPECTED_BUNDLE_SHA256"], file_sha256(bundle))
            self.assertEqual(settings["EPOCHS"], 20)
            self.assertEqual(settings["PATIENCE"], 3)
            self.assertEqual(settings["BUDGETS"], [10, 50, 100])
            self.assertEqual(settings["SEEDS"], [42, 43, 44])
            self.assertEqual(settings["CORPORA"], ["inaugural"])
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    compile("".join(cell["source"]), "<test-colab>", "exec")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                create_notebook(SimpleNamespace(root=root), bundle, root / "colab/nltk_data_efficiency.ipynb")


if __name__ == "__main__":
    unittest.main()
