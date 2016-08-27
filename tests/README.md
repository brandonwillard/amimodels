
# Usage

To run a single test from within an interactive Python session 
([ref](http://doc.pytest.org/en/latest/usage.html#calling-pytest-from-python-code))
and be able to drop into `ipdb` ([ref](http://blog.pytest.org/)):

```
import pytest
pytest.main(["-x", "-s", "--pdbcls=IPython.core.debugger:Pdb", 
             "tests/test_normal_hmm.py::test_prediction"]) 
```
