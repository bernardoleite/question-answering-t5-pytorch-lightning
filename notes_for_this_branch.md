This branch was created for further experiments in information retrieval. Not ready for usage.
Developer Notes (so as not to forget):
- This branch uses PyTerrier, which must be used with python version 3.8.x
- Run (in order):
  -  src/data/filter_squad_en_original_dev.py for filtering examples
  -  src/data/terrier_example.py (see path_index)
  -  eval_qa_squad1.1py with correct filtered files