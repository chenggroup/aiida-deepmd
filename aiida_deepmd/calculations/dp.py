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
from aiida.orm.nodes.data.singlefile import SinglefileData
from aiida.common import CalcInfo, CodeInfo, InputValidationError


class DpCalculation(CalcJob):
    """
    This is a DpCalculation, used to prepare input for
    deepmd dp training and freeze model.
    It is a combine of train and freeze stage of dp command
    User will obtain a freezed model and a list of output
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

        spec.input('datadirs', valid_type=list, help='parameters of datadirs', non_db=True)

        # a special datatype is need to write the files for training and then uploaded

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

        # create json input file
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
        local_copy_list = []
        for datadir in self.inputs.datadirs:
            # change to absolute path
            if not os.path.exists(datadir):
                raise FileExistsError("This datadir dose not exist")
            absdatadir = os.path.abspath(datadir)
            # create subfolder
            datadir_basename = os.path.basename(absdatadir)
            datadir_in_workdir = os.path.join("./", datadir_basename)
            folder.get_subfolder(datadir_in_workdir, create=True)
            # this loop use to copy the training data under the datadir
            for root, directories, files in os.walk(top=absdatadir, topdown=True):
                relroot = os.path.relpath(root, absdatadir)
                # create subtree folders
                for name in directories:
                    folder.get_subfolder(
                        os.path.join(
                            datadir_basename,
                            relroot,
                            name),
                        create=True)

                # give the singlefiledata to file
                for name in files:
                    fobj = SinglefileData(
                        file=os.path.join(root, name)
                    )
                    # must save fobj otherwise the node is empty and can't be copied
                    fobj.store()
                    dst_path = os.path.join(
                        datadir_basename,
                        relroot,
                        name)
                    local_copy_list.append((fobj.uuid, fobj.filename, dst_path))



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


