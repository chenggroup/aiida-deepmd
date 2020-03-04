""" Tests for calculations

"""
import os
from aiida_deepmd import tests

from aiida import orm

def test_process(deepmd_code):
    """Test running a calculation
    note this does not test that the expected outputs are created of output parsing"""
    from aiida.plugins import DataFactory, CalculationFactory
    from aiida.engine import run

    # Prepare input parameters
    DiffParameters = DataFactory('deepmd')
    parameters = DiffParameters({'ignore-case': True})

    from aiida.orm import SinglefileData
    file1 = SinglefileData(
        file=os.path.join(tests.TEST_DIR, "input_files", 'file1.txt'))
    file2 = SinglefileData(
        file=os.path.join(tests.TEST_DIR, "input_files", 'file2.txt'))

    # data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_folder')
    data_folder = orm.FolderData(
        tree=os.path.join(tests.TEST_DIR, "input_files"))
    # data_folder = './input_folder'
    # set up calculation
    inputs = {
        'code': deepmd_code,
        'model': orm.Dict(dict={}),
        'learning_rate': orm.Dict(dict={}),
        'loss': orm.Dict(dict={}),
        'training': orm.Dict(dict={}),
        'file': {
            'box_raw': SinglefileData(
                file=os.path.join(tests.TEST_DIR, "input_files", 'file1.txt')),
            'coord_raw': SinglefileData(
                file=os.path.join(tests.TEST_DIR, "input_files", 'file2.txt'))
        },
        'metadata': {
            'dry_run': True,
            'options': {
                'max_wallclock_seconds': 30,
                'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1}
            },
        },
    }

    result = run(CalculationFactory('dptrain'), **inputs)
    computed_diff = result['deepmd'].get_content()

    assert 'content1' in computed_diff
    assert 'content2' in computed_diff
