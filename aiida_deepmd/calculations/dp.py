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


class DpTrainCalculation(CalcJob):
    """
    This is a DpTrainCalculation, used to prepare input for
    deepmd dp training.
    It is a combine of train and freeze stage of dp command
    Users are supposed to get a frozen_model file which can be used for seq calculation
    For information on deepmd, refer to: https://github.com/deepmodeling/deepmd-kit
    """

    # Defaults
    _DEFAULT_INPUT_FILE = 'aiida.json'
    _DEFAULT_OUTPUT_FILE = 'aiida.out'
    _DEFAULT_PROJECT_NAME = 'aiida'
    _TRAIN_DATA_SUBFOLDER = './data/'
    _TRAIN_SET_PRFIX = 'set.'
    _DEFAULT_OUTPUT_INFO_FILE = 'lcurve.out'
    _DEFAULT_CHECK_META_FILE = 'model.ckpt.meta'
    _DEFAULT_CHECK_INDEX_FILE = 'model.ckpt.index'
    _DEFAULT_CHECK_META_PREFIX = 'model.ckpt.data'

    @classmethod
    def define(cls, spec):
        super(DpTrainCalculation, cls).define(spec)

        # Input parameters
        spec.input('model', valid_type=orm.Dict, help='parameters of model')
        spec.input('learning_rate', valid_type=orm.Dict, help='parameters control learning behaviour')
        spec.input('loss', valid_type=orm.Dict, help='parameters of loss function')
        spec.input('training', valid_type=orm.Dict, help='parameters of training')

        # TODO: use folder here not store the data which confilct the provenance
        # a special datatype is need to write the files for training and then uploaded
        # spec.input_namespace('file')
        spec.input('file.box_raw', valid_type=orm.SinglefileData, required=True, help='raw data')
        spec.input('file.coord_raw', valid_type=orm.SinglefileData, required=True, help='raw data')
        # spec.input('file.energy_raw', valid_type=orm.SinglefileData, required=True, help='raw data')
        # spec.input('file.force_raw', valid_type=orm.SinglefileData, required=True, help='raw data')
        #
        # spec.input_namespace('add_file.type_map_raw', valid_type=orm.SinglefileData, required=False, help='raw data')
        # spec.input_namespace('add_file.type_raw', valid_type=orm.SinglefileData, required=False, help='raw data')

        # inputs.metadata.options.resources = {}
        spec.input('metadata.options.withmpi', valid_type=bool, default=True)

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

        local_copy_list = []
        # Create the subfolder that will contain the train data
        folder.get_subfolder(self._TRAIN_DATA_SUBFOLDER, create=True)
        for fn in self.inputs.file:
            fobj = self.inputs.file[fn]
            src_path = fobj.filename
            dst_path = os.path.join(self._TRAIN_DATA_SUBFOLDER, fobj.filename)
            local_copy_list.append((fobj.uuid, src_path, dst_path))

        def create_array_from_files(files):
            for f in files:
                content = f.get_content()
                # function from aiida_ce
                pass
            return data_array

        # create train set and store the data in
        data_array = create_array_from_files(self.inputs.file)

        # for simplicity do not split the folder
        n = 0
        set_folder = folder.get_subfolder(os.path.join(self._TRAIN_DATA_SUBFOLDER, self._TRAIN_SET_PRFIX + str(n)), create=True)
        with io.open(set_folder.get_abs_path(coord.npy), mode="w") as fobj:
            try:
                np.dump(fobj, array=data_array)
            except ValueError as exc:
                raise InputValidationError("invalid keys or values in input parameters found")

        # settings = self.inputs.settings.get_dict() if 'settings' in self.inputs else {}

        # create code info
        codeinfo = CodeInfo()
        codeinfo.cmdline_params = ["train", self._DEFAULT_INPUT_FILE]
        codeinfo.stdout_name = self._DEFAULT_OUTPUT_FILE
        codeinfo.join_files = True
        codeinfo.code_uuid = self.inputs.code.uuid

        # create calc info
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.stdin_name = self._DEFAULT_INPUT_FILE
        calcinfo.stdout_name = self._DEFAULT_OUTPUT_FILE
        calcinfo.local_copy_list = local_copy_list
        calcinfo.codes_info = [codeinfo]

        calcinfo.retrieve_list = [
            # self._DEFAULT_OUTPUT_FILE,
            # self._DEFAULT_OUTPUT_INFO_FILE,
            # self._DEFAULT_CHECK_META_FILE,
            # self._DEFAULT_CHECK_INDEX_FILE,
        ]

        return calcinfo

class DpFreezeCalculation(CalcJob):
    """
    This is a DpFreezeCalculation, used to freeze the model.
    It is used in the BaseRestartWorkChain with train
    Users are supposed to get a frozen_model file which can be used for seq calculation
    For information on deepmd, refer to: https://github.com/deepmodeling/deepmd-kit
    """

    # Defaults
    _DEFAULT_OUTPUT_FILE = 'model.pb'
    _DEFAULT_PARENT_CALC_FLDR_NAME = './'

    @classmethod
    def define(cls, spec):
        super(DpFreezeCalculation, cls).define(spec)

        # Input parameters
        spec.input('parent_calc_folder', valid_type=orm.RemoteData, required=True, help='remote folder used for processing')

        # Exit codes
        spec.exit_code(100,
                       'ERROR_NO_RETRIEVED_FOLDER',
                       message='The retrieved folder data node could not be accessed.')

        # Output parameters
        # parser will parse the retrieved file into froze_model
        spec.output('model', valid_type=orm.SinglefileData, required=True, help='freeze model')

    def prepare_for_submission(self, folder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.

        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        # symlinks
        calcinfo.remote_symlink_list = []
        calcinfo.remote_copy_list = []

        comp_uuid = self.inputs.parent_calc_folder.computer.uuid
        remote_path = self.inputs.parent_calc_folder.get_remote_path()
        copy_info = (comp_uuid, remote_path, self._DEFAULT_PARENT_CALC_FLDR_NAME)
        if self.inputs.code.computer.uuid == comp_uuid:  # if running on the same computer - make a symlink
            # if not - copy the folder
            calcinfo.remote_symlink_list.append(copy_info)
        else:
            calcinfo.remote_copy_list.append(copy_info)


        # create code info
        codeinfo = CodeInfo()
        codeinfo.cmdline_params = ["freeze", '-o', self._DEFAULT_OUTPUT_FILE]
        codeinfo.stdout_name = self._DEFAULT_OUTPUT_FILE
        codeinfo.code_uuid = self.inputs.code.uuid

        calcinfo.uuid = self.uuid
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.codes_info = [codeinfo]
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list

        calcinfo.retrieve_list = [
            self._DEFAULT_OUTPUT_FILE,
        ]

        # check for left over settings
        if settings:
            raise InputValidationError("The following keys have been found " +
                                       "in the settings input node {}, ".format(self.pk) + "but were not understood: " +
                                       ",".join(settings.keys()))

        return calcinfo
