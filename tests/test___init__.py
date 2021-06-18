import unittest
import os
import subprocess

from tests import TEST_DATA_DIR

test_dir = os.path.join(TEST_DATA_DIR, 'snr', 'GE')
# just need something to check CLI works

class Test_init(unittest.TestCase):
    def test_CLI_success(self):
        process = subprocess.run(['hazen', 'snr', test_dir],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True)
        assert process.returncode == 0  # check CLI returned OK

    def test_CLI_invalid_task(self):
        process = subprocess.run(['hazen', 'invalid_test_name_3413', test_dir],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True)
        assert process.returncode != 0  # check CLI failed


if __name__ == '__main__':
    unittest.main()
