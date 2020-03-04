[![Build Status](https://github.com/unkcpz/aiida-deepmd/workflows/ci/badge.svg?branch=master)](https://travis-ci.org/unkcpz/aiida-deepmd/actions)
[![Coverage Status](https://coveralls.io/repos/github/unkcpz/aiida-deepmd/badge.svg?branch=master)](https://coveralls.io/github/unkcpz/aiida-deepmd?branch=master)
[![Docs status](https://readthedocs.org/projects/aiida-deepmd/badge)](http://aiida-deepmd.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/aiida-deepmd.svg)](https://badge.fury.io/py/aiida-deepmd)

# aiida-deepmd

AiiDA plugin for DeepMD a machine learning potential function

This plugin is the default output of the
[AiiDA plugin cutter](https://github.com/aiidateam/aiida-plugin-cutter),
intended to help developers get started with their AiiDA plugins.

Plugins templated using the plugin cutter

* include a calculation, parser and data type as well as an example of
  how to submit a calculation
* include basic regression tests using the [pytest](https://docs.pytest.org/en/latest/) framework ( (submitting a calculation, ...)
* can be directly pip-installed (and are prepared for submisson to [PyPI](https://pypi.org/)
* include a documentation template ready for [Read the Docs](http://aiida-diff.readthedocs.io/en/latest/)
* come with [Github Actions](https://github.com/features/actions) configuration - enable it to run tests and check test coverage at every commit
* come with pre-commit hooks that sanitize coding style and check for syntax errors - enable via `pre-commit install`

For more information on how to take advantage of these features,
see the [developer guide](https://aiida-diff.readthedocs.io/en/latest/developer_guide) of your plugin.


## Features

 * Add input files using `SinglefileData`:
   ```python
   SinglefileData = DataFactory('singlefile')
   inputs['file1'] = SinglefileData(file='/path/to/file1')
   inputs['file2'] = SinglefileData(file='/path/to/file2')
   ```

 * Specify command line options via a python dictionary and `DiffParameters`:
   ```python
   d = { 'ignore-case': True }
   DiffParameters = DataFactory('deepmd')
   inputs['parameters'] = DiffParameters(dict=d)
   ```

 * `DiffParameters` dictionaries are validated using [voluptuous](https://github.com/alecthomas/voluptuous).
   Find out about supported options:
   ```python
   DiffParameters = DataFactory('deepmd')
   print(DiffParameters.schema.schema)
   ```

## Installation

```shell
pip install aiida-deepmd
verdi quicksetup  # better to set up a new profile
verdi plugin list aiida.calculations  # should now show your calclulation plugins
```


## Usage

Here goes a complete example of how to submit a test calculation using this plugin.

A quick demo of how to submit a calculation:
```shell
verdi daemon start     # make sure the daemon is running
cd examples
./example_01.py        # run test calculation
verdi process list -a  # check record of calculation
```

The plugin also includes verdi commands to inspect its data types:
```shell
verdi data deepmd list
verdi data deepmd export <PK>
```

## Development

```shell
git clone https://github.com/unkcpz/aiida-deepmd .
cd aiida-deepmd
pip install -e .[pre-commit,testing]  # install extra dependencies
pre-commit install  # install pre-commit hooks
pytest -v  # discover and run all tests
```

See the [developer guide](http://aiida-deepmd.readthedocs.io/en/latest/developer_guide/index.html) for more information.

## License

MIT


## Contact

morty.yu@yahoo.com

