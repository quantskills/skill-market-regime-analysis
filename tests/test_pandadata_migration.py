import ast
import datetime
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def functions(relative, names, **context):
    source=ast.parse((ROOT/relative).read_text(encoding="utf-8"))
    nodes=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)]
    nodes.extend(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names)
    namespace={"pd":pd,"datetime":datetime,"sys":sys,**context}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),relative,"exec"),namespace)
    return namespace

SOURCE='examples/regime_switching_strategy.py'
class MigrationTests(unittest.TestCase):
    def test_multiplier_uses_exported_contract_metadata(self):
        from types import SimpleNamespace
        sdk=Mock()
        sdk.get_future_dominant.return_value=pd.DataFrame([{"symbol":"IF2601.CFE","nature_date":"20260102"}])
        sdk.get_future_detail.return_value=pd.DataFrame([{"symbol":"IF2601.CFE","contract_multiplier":300}])
        sdk.get_future_list.side_effect=AttributeError("removed")
        ns=functions(SOURCE,{"_preload_all_data"},panda_data=sdk)
        context=SimpleNamespace(products=["IF"])
        ns["_preload_all_data"](context)
        self.assertEqual(context._mul_map["IF2601.CFE"],300)
        sdk.get_future_detail.assert_called_once()

if __name__=="__main__":unittest.main()
