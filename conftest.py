"""pytest fixtures for simplified testing."""
from __future__ import absolute_import
import pytest
pytest_plugins = ['aiida.manage.tests.pytest_fixtures']


@pytest.fixture(scope='function', autouse=True)
def clear_database_auto(clear_database):  # pylint: disable=unused-argument
    """Automatically clear database in between tests."""
@pytest.fixture(scope='function')
def deepmd_code(aiida_local_code_factory):
    """Get a deepmd code mock.
    """
    import os

    path = os.getcwd()
    code_path = os.path.join(path, 'mock/dp')
    code = aiida_local_code_factory(executable=code_path, entry_point='dptrain')
    return code
