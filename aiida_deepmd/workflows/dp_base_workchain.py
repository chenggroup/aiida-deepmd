# -*- coding: utf-8 -*-

from aiida import orm
from aiida.engine import BaseRestartWorkChain

class DpBaseWorkChain(WorkChain):
    """The work chain combine train and freeze stage"""
    @classmethod
    def define(cls, spec):
        super(DpBaseWorkChain, cls).define(spec)
        spec.inputs('structure_set', valid_type=StrctureSet,
            help='datatype store property and structure infos of structures for training')
        spec.inputs('parameters', valid_type=orm.Dict,
            help='all input parameters as a whole, be checked in this WorkChain')
        spec.input('clean_workdir', valid_type=orm.Bool, default=orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        # spec.expose_inputs(DpTrainBaseWorkChain)
        # add other specific feature you want
        spec.expose_outputs(DpFreezeBaseWorkChain)
        spec.outline(
            cls.setup_parameters,
            cls.training,
            cls.freeze_model,
            cls.results,
        )
        spec.exit_code(201, 'ERROR_INVALID_INPUT_UNRECOGNIZED_KIND',
            message='Input `parameters` contains an unsupported kind.')
        spec.output('running_info', valid_type=orm.Dict)

    def setup_parameters(self):
        """validte the parameters"""
        pass

    def training(self):
        """start training"""
        pass
        inputs = AttributeDict({
            'model': {
                'fiiting_net': {
                    'resnet_dt': true,
                },
            },
            'loss': {
                'start_pref_e': 0.02,
            },
        })
        running = self.submit(DpTrainBaseWorkChain, **inputs)

        self.report('launching DpTrainBaseWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_training=running)

    def freeze_model(self):
        """freeze model after training is done"""
        running = self.submit(DpFreezeBaseWorkChain, **inputs)

        self.report('launching DpFreezeBaseWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_freeze=running)

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain succesfully completed')
        self.out('freeze_model', self.ctx.workchain_freeze.outputs.freeze_model)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super(DpBaseWorkChain, self).on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report('cleaned remote folders of calculations: {}'.format(' '.join(map(str, cleaned_calcs))))


class DpTrainBaseWorkChain(BaseRestartWorkChain):
    """dp base workchain combine both train stage"""

    _process_class = DpTrainCalculation

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super(DpTrainBaseWorkChain, cls).define(spec)
        spec.expose_inputs(DpTrainCalculation, namespace='train')

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results
        )
        spec.exit_code(400, 'DUMMY_ERROR', message='isomp')

        spec.expose_outputs(DpTrainCalculation)

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit the calculations in the
        internal loop.
        """
        super(DpTrainBaseWorkChain, self).setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(DpTrainCalculation, 'train'))

class DpFreezeBaseWorkChain(BaseRestartWorkChain):
    """dp base workchain combine both train stage"""

    _process_class = DpFreezeCalculation

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super(DpFreezeCalculationBaseWorkChain, cls).define(spec)
        spec.expose_inputs(DpFreezeCalculation, namespace='freeze')

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results
        )
        spec.exit_code(400, 'DUMMY_ERROR', message='isomp')

        spec.expose_outputs(DpFreezeCalculation)

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit the calculations in the
        internal loop.
        """
        super(DpFreezeBaseWorkChain, self).setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(DpFreezeCalculation, 'freeze'))
