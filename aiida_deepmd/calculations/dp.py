# -*- coding: utf-8 -*-
"""AiiDA-deepmd `dp train` command for train input plugin"""

from __future__ import absolute_import

import io
import six
from six.moves import map
import json
import os
import numpy as np

from aiida.engine import CalcJob
from aiida import orm
from aiida.common import CalcInfo, CodeInfo, InputValidationError


class DpCalculation(CalcJob):
    """
    This is a DpTrainCalculation, used to prepare input for
    deepmd dp training.
    It is a combine of train and freeze stage of dp command
    Users are supposed to get a frozen_model file which can be used for seq calculation
    For information on deepmd, refer to: https://github.com/deepmodeling/deepmd-kit
    """

    # Defaults for training
    _DEFAULT_INPUT_FILE = 'aiida.json'
    _DEFAULT_TRAIN_OUTPUT_FILE = 'aiida.out'
    _DEFAULT_PROJECT_NAME = 'aiida'
    _TRAIN_DATA_SUBFOLDER = './data/'
    _TRAIN_SET_SUBFOLDER = './data/set.000'
    _DEFAULT_OUTPUT_INFO_FILE = 'lcurve.out'
    _DEFAULT_CHECK_META_FILE = 'model.ckpt.meta'
    _DEFAULT_CHECK_INDEX_FILE = 'model.ckpt.index'
    _DEFAULT_CHECK_META_PREFIX = 'model.ckpt.data'

    # Defaults for freeze
    _DEFAULT_FREEZE_OUTPUT_FILE = 'model.pb'
    _DEFAULT_PARENT_CALC_FLDR_NAME = './'

    @classmethod
    def define(cls, spec):
        super(DpCalculation, cls).define(spec)

        # Input parameters
        spec.input('model', valid_type=orm.Dict, help='parameters of model')
        spec.input('learning_rate', valid_type=orm.Dict, help='parameters control learning behaviour')
        spec.input('loss', valid_type=orm.Dict, help='parameters of loss function')
        spec.input('training', valid_type=orm.Dict, help='parameters of training')

        # TODO: use folder here not store the data which confilct the provenance
        # a special datatype is need to write the files for training and then uploaded
        # spec.input_namespace('file')
        spec.input('file.type_raw', valid_type=orm.SinglefileData, required=True, help='raw or npy data')
        spec.input('file.box', valid_type=orm.SinglefileData, required=True, help='raw or npy data')
        spec.input('file.coord', valid_type=orm.SinglefileData, required=True, help='raw or npy data')
        spec.input('file.energy', valid_type=orm.SinglefileData, required=True, help='raw or npy data')
        spec.input('file.force', valid_type=orm.SinglefileData, required=True, help='raw or npy data')
        spec.input('file.virial', valid_type=orm.SinglefileData, required=False, help='raw or npy data')
        #
        # spec.input_namespace('add_file.type_map_raw', valid_type=orm.SinglefileData, required=False, help='raw data')
        # spec.input_namespace('add_file.type_raw', valid_type=orm.SinglefileData, required=False, help='raw data')

        # inputs.metadata.options.resources = {}
        spec.input('metadata.options.withmpi', valid_type=bool, default=False)

        # Exit codes
        spec.exit_code(100,
                       'ERROR_NO_RETRIEVED_FOLDER',
                       message='The retrieved folder data node could not be accessed.')

        # Output parameters
        # parser will parse the retrieved file into froze_model
        spec.output('folder', valid_type=orm.FolderData, required=True, help='the folder contain the meta files')
        # spec.default_output_node = 'output_parameters'

    def prepare_for_submission(self, folder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.

        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        # from aiida_deepmd.utils import DpInput

        # create json input file
        # input = DpInput(self.inputs.model.get_dict(), self.inputs.learning_rate.get_dict(), self.inputs.loss.get_dict(), self.inputs.training.get_dict())
        input = dict()
        input['model'] = self.inputs.model.get_dict()
        input['learning_rate'] = self.inputs.learning_rate.get_dict()
        input['loss'] = self.inputs.loss.get_dict()
        input['training'] = self.inputs.training.get_dict()
        json_str = json.dumps(input, indent=4, sort_keys=False)

        with io.open(folder.get_abs_path(self._DEFAULT_INPUT_FILE), mode="w", encoding="utf-8") as fobj:
            try:
                fobj.write(json_str)
            except ValueError as exc:
                raise InputValidationError("invalid keys or values in input parameters found")

        # Create the subfolder that will contain the train data

        folder.get_subfolder(self._TRAIN_DATA_SUBFOLDER, create=True)
        folder.get_subfolder(self._TRAIN_SET_SUBFOLDER, create=True)

        # remember to copy type map
        local_copy_list = []
        for name, obj in self.inputs.file.items():
            # if type.map, copy to the ./data
            if name == 'type_raw':
                dst_path = os.path.join(self._TRAIN_DATA_SUBFOLDER, obj.filename)
                local_copy_list.append((obj.uuid, obj.filename, dst_path))
            # copy other files to the ./data/set.000
            else:
                dst_path = os.path.join(self._TRAIN_SET_SUBFOLDER, obj.filename)
                local_copy_list.append((obj.uuid, obj.filename, dst_path))
#refactor
#        def create_array_from_files(files):
#            for f in files:
#                data_array = np.loadtxt(f)
#                # function from aiida_ce
#                pass
#            return data_array

        # create train set and store the data in
#        box_data = np.loadtxt(self.inputs.file.box_raw)
#        coord_data = np.loadtxt(self.inputs.file.coord_raw)


        # for simplicity do not split the folder
#        set_folder = folder.get_subfolder(os.path.join(self._TRAIN_DATA_SUBFOLDER, self._TRAIN_SET_PRFIX + str(n)), create=True)
#        try:
#            coord_data.dump("coord.npy")
#        except ValueError as exc:
 #           raise InputValidationError("invalid keys or values in input parameters found")

        # settings = self.inputs.settings.get_dict() if 'settings' in self.inputs else {}

        # set two code info here, once the training finished, the model will freeze then.
        # create code info for training
        codeinfotrain = CodeInfo()
        codeinfotrain.cmdline_params = ["train", self._DEFAULT_INPUT_FILE]
        #codeinfotrain.stdin_name = self._DEFAULT_INPUT_FILE
        codeinfotrain.stdout_name = self._DEFAULT_TRAIN_OUTPUT_FILE
        codeinfotrain.join_files = True
        codeinfotrain.code_uuid = self.inputs.code.uuid
        codeinfotrain.withmpi = self.inputs.metadata.options.withmpi

        # create code info for freeze
        codeinfofreeze = CodeInfo()
        codeinfofreeze.cmdline_params = ["freeze", '-o', self._DEFAULT_FREEZE_OUTPUT_FILE]
        #codeinfofreeze.stdout_name = self._DEFAULT_FREEZE_OUTPUT_FILE
        codeinfofreeze.code_uuid = self.inputs.code.uuid
        codeinfofreeze.withmpi = self.inputs.metadata.options.withmpi


        # create calc info
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.local_copy_list = local_copy_list
        calcinfo.codes_info = [codeinfotrain, codeinfofreeze]

        calcinfo.retrieve_list = [
            self._DEFAULT_TRAIN_OUTPUT_FILE,
            self._DEFAULT_FREEZE_OUTPUT_FILE,
            self._DEFAULT_OUTPUT_INFO_FILE,
            self._DEFAULT_CHECK_META_FILE,
            self._DEFAULT_CHECK_INDEX_FILE
        ]

        return calcinfo


