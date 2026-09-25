import unittest
import pathlib
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class TestHybridRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the notebook once and keep the executed notebook object."""
        nb_path = pathlib.Path("RAG_pipeline.ipynb")  
        cls.nb = nbformat.read(str(nb_path), as_version=4)
        
        # Create a dictionary to hold the notebook's variables
        # We use a separate dict instead of globals() to keep the test environment clean
        cls.notebook_globals = {}

        for cell in cls.nb.cells:
            if cell.cell_type == "code":
                source = cell.source
                filtered_source = "\n".join(
                    [line for line in source.splitlines() if not line.strip().startswith('%')]
                )
                
                try:
                    exec(filtered_source, cls.notebook_globals)
                except Exception as e:
                    print(f"Error executing cell: {e}")
                    raise e

        # Verify that final_answer was actually created in the notebook
        if "final_answer" not in cls.notebook_globals:
            raise RuntimeError("Notebook executed successfully, but `final_answer` variable was not found.")

    def test_core_thesis_concept(self):
        """The generated answer should convey the core thesis about digital literacy."""
        answer = self.notebook_globals.get("final_answer", "").lower()
    
        # Check for thesis-conveying concepts instead of exact phrase
        thesis_keywords = ["digital", "internet", "leader", "technology"]
        keywords_found = sum(1 for kw in thesis_keywords if kw in answer)
        
        self.assertGreaterEqual(
            keywords_found, 
            3,
            msg=f"Answer should contain core thesis concepts. Found: {keywords_found}/4"
        )      
    
if __name__ == "__main__":
    unittest.main()
    