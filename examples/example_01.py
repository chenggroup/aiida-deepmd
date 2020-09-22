#!/usr/bin/env python
"""Run a test calculation on localhost.

Usage: ./example_01.py
"""
import os
from aiida_deepmd import helpers
from aiida import cmdline, engine
from aiida.plugins import DataFactory, CalculationFactory
from aiida.orm import Dict
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
    with open(os.path.join("../aiida_deepmd/tests/input_files/", "water_se_a.json")) as f:
        train_param = json.load(f)


    # set up calculation
    inputs = {
        'code': deepmd_code,
        'model': Dict(dict=train_param["model"]),
        'learning_rate': Dict(dict=train_param["learning_rate"]),
        'loss': Dict(dict=train_param["loss"]),
        'training': Dict(dict=train_param["training"]),
        'datadirs':["../aiida_deepmd/tests/input_files/train_data/",
                    "../aiida_deepmd/tests/input_files/train_data2/"],
        'metadata': {
            'description': "Test job submission with the aiida_deepmd plugin",
            'dry_run': False,
            'options':{
                'resources': {
                    'num_machines': 1,
                    'tot_num_mpiprocs': 2
                },
                'queue_name': 'gpu',
                'custom_scheduler_commands': '#SBATCH --gres=gpu:1'
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
