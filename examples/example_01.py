#!/usr/bin/env python
"""Run a test calculation on localhost.

Usage: ./example_01.py
"""
import os
from aiida_deepmd import tests, helpers
from aiida import cmdline, engine
from aiida.plugins import DataFactory, CalculationFactory
from aiida.orm import Dict, SinglefileData
import click
import json

def test_run(deepmd_code):
    """Run a calculation on the localhost computer.

    Uses test helpers to create AiiDA Code on the fly.
    """
    if not deepmd_code:
        # get code
        computer = helpers.get_computer()
        deepmd_code = helpers.get_code(entry_point='deepmd', computer=computer)

    # Prepare input parameters
#    DiffParameters = DataFactory('deepmd')
#    parameters = DiffParameters({'ignore-case': True})
    with open(os.path.join(tests.TEST_DIR, "input_files", "water_se_a.json")) as f:
        train_param = json.load(f)
    box = SinglefileData(
        file=os.path.join(tests.TEST_DIR, "input_files", 'box.npy'))
    coord = SinglefileData(
        file=os.path.join(tests.TEST_DIR, "input_files", 'coord.npy'))
    energy = SinglefileData(
        file=os.path.join(tests.TEST_DIR, "input_files", 'energy.npy'))
    force = SinglefileData(
        file=os.path.join(tests.TEST_DIR, "input_files", 'force.npy'))
    typeraw = SinglefileData(
        file=os.path.join(tests.TEST_DIR, "input_files", 'type.raw'))

    # set up calculation
    inputs = {
        'code': deepmd_code,
#        'parameters': parameters,
        'model': Dict(dict=train_param["model"]),
        'learning_rate': Dict(dict=train_param["learning_rate"]),
        'loss': Dict(dict=train_param["loss"]),
        'training': Dict(dict=train_param["training"]),
        'file': {
            'box': box,
            'coord': coord,
            'energy': energy,
            'force': force,
            'type_raw': typeraw
        },
        'metadata': {
            'description': "Test job submission with the aiida_deepmd plugin",
            'dry_run': True,
            'options':{
                'resources': {
                    'tot_num_mpiprocs': 4
                },
                'queue_name': 'large'
            }
        },
    }

    # Note: in order to submit your calculation to the aiida daemon, do:
    # from aiida.engine import submit
    # future = submit(CalculationFactory('deepmd'), **inputs)
    result = engine.run(CalculationFactory('deepmd'), **inputs)
    print(result)
#    computed_diff = result['deepmd'].get_content()
#    print("Computed diff between files: \n{}".format(computed_diff))


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODE()
def cli(code):
    """Run example.

    Example usage: $ ./example_01.py --code diff@localhost

    Alternative (creates diff@localhost-test code): $ ./example_01.py

    Help: $ ./example_01.py --help
    """
    test_run(code)


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
