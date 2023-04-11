import shutil

import ggsolver.ioutils as io
import unittest
import pathlib


class TestJSON(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create folder if it does not exist.
        if not pathlib.Path("out/").exists():
            pathlib.Path("out/").mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("out/")

    def setUp(self):
        self.obj_dict = {"list": [1, 2, 3], "tuple": (4, 5, 6), "dict": {"t": "v"}}
        self.fpath = pathlib.Path("out/test.json").absolute().name

    def test_jsonify(self):
        io.to_json(self.fpath, self.obj_dict)
        loaded_dict = io.from_json(self.fpath)
        self.assertEqual(self.obj_dict, loaded_dict)
