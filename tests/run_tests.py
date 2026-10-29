"""
Tiny test runner so you do not need pytest:   python tests/run_tests.py
(With pytest installed, `pytest` works too: every function named test_* is a normal pytest test.)
Run one file:   python tests/run_tests.py test_engines
"""
import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

wanted = sys.argv[1:]
modules = [f[:-3] for f in sorted(os.listdir(HERE)) if f.startswith("test_") and f.endswith(".py")]
if wanted:
    modules = [m for m in modules if m in wanted]
passed = failed = 0
for m in modules:
    mod = importlib.import_module(m)
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            try:
                getattr(mod, name)()
                print(f"  ok    {m}.{name}"); passed += 1
            except Exception:
                print(f"  FAIL  {m}.{name}"); traceback.print_exc(); failed += 1
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
