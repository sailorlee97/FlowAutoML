import os
import shutil

from embedia.layers.model import Model as EmbediaModel
from embedia.model_generator.project_options import (
        ModelDataType,
        ProjectType,
        ProjectFiles,
        DebugMode
    )

from embedia.model_generator.generate_files import (
        generate_embedia_library,
        generate_embedia_model,
        generate_embedia_main,
        generate_codeblock_project
    )

from embedia.utils import file_management

from prettytable import PrettyTable

def format_model_name(model):
    model_name = model.name.lower()
    if not model_name.endswith('model'):
        model_name += '_model'

    return model_name


class ProjectGenerator:

    def __init__(self, options):
        self._dst_folder = None
        self._options = options

        if options.embedia_folder is None or options.embedia_folder == '':
            self.set_embedia_folder('embedia/')
        else:
            self.set_embedia_folder(options.embedia_folder)

    def set_embedia_folder(self, folder):
        if folder[-1] != '/':
            folder += '/'
        self._root_folder = os.path.abspath(folder) + '/'
        self._lib_folder = self._root_folder + 'libraries/'
        self._datatype_folder = self._lib_folder + self._datatype_subfolder(self._options.data_type)
        self._src_dbg_folder = self._lib_folder + 'debug/'

    def create_project(self, output_folder, project_name, model, options):

        embedia_model = EmbediaModel(options)
        layers_embedia = embedia_model.set_layers(model.layers)

        # prepare folders and extension of files to copy/create
        self._prepare_folders(output_folder, project_name, options)
        if options.project_type in [ProjectType.C, ProjectType.CODEBLOCK]:
            (h_ext, c_ext) = ('.h', '.c')
        else:
            (h_ext, c_ext) = ('.h', '.cpp')

        # copy library files
        if ProjectFiles.LIBRARY in options.files:

            embedia_files = generate_embedia_library(layers_embedia, self._datatype_folder, options)

            for filename in embedia_files:
                content = embedia_files[filename]

                if filename.endswith('.c'):
                    filename = filename.replace('.c', c_ext)
                elif filename.endswith('.h'):
                    filename = filename.replace('.h', h_ext)

                file_management.save_to_file(os.path.join(self._dst_folder, filename), content)

            # print layers memory size
            self.print_layers_info(embedia_model, embedia_files['embedia.h'])

        # create model files
        if ProjectFiles.MODEL in options.files:
            (text_model_h, text_model_c, filename) = generate_embedia_model(layers_embedia, self._lib_folder, model.name, options)
            file_management.save_to_file(os.path.join(self._dst_folder, filename + h_ext), text_model_h)
            file_management.save_to_file(os.path.join(self._dst_folder, filename + c_ext), text_model_c)

        # copy debug file
        if options.debug_mode != DebugMode.DISCARD:

            # add debug mode macro to header file
            content = file_management.read_from_file(os.path.join(self._src_dbg_folder,'embedia_debug.h'))
            # add include
            content = content.format(EMBEDIA_DEBUG='#define EMBEDIA_DEBUG %d\n' % options.debug_mode)

            file_management.save_to_file(os.path.join(self._dst_folder, 'embedia_debug.h'),''.join(content))

            # copy aditional debug file
            if options.project_type == ProjectType.ARDUINO:
                shutil.copy(os.path.join(self._src_dbg_folder, 'embedia_debug_def_arduino.h'), os.path.join(self._dst_folder, 'embedia_debug_def.h'))
                # copy implementation file
                shutil.copy(os.path.join(self._src_dbg_folder, 'embedia_debug.c'), os.path.join(self._dst_folder, 'embedia_debug.cpp'))
            else:
                shutil.copy(os.path.join(self._src_dbg_folder, 'embedia_debug_def_c.h'), os.path.join(self._dst_folder, 'embedia_debug_def.h'))
                # copy implementation file
                shutil.copy(os.path.join(self._src_dbg_folder, 'embedia_debug.c'), os.path.join(self._dst_folder, 'embedia_debug.c'))

        # create main file with an example
        if ProjectFiles.MAIN in options.files:
            (text_example_h, text_main_c) = generate_embedia_main(layers_embedia, self._lib_folder, filename, options)
            if options.project_type == ProjectType.ARDUINO:
                filename = project_name
                c_ext = '.ino'
            else:
                filename = "main"
                if options.project_type == ProjectType.CODEBLOCK:
                    model_name = format_model_name(model)
                    files = self._get_project_files(model_name, options)
                    project = generate_codeblock_project(project_name, files, self._lib_folder)
                    file_management.save_to_file(os.path.join(self._dst_folder, project_name+'.cbp'), project)

            file_management.save_to_file(os.path.join(self._dst_folder, filename + c_ext), text_main_c)
            if text_example_h is not None:
                file_management.save_to_file(os.path.join(self._dst_folder, 'example_file' + h_ext), text_example_h)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
    def _datatype_subfolder(self, data_type):
        # absolute path for copying files
        if data_type == ModelDataType.FIXED8:
            return 'fixed8/'
        elif data_type == ModelDataType.FIXED16:
            return 'fixed16/'
        elif data_type == ModelDataType.FIXED32:
            return 'fixed32/'
        elif data_type == ModelDataType.BINARY:
            return 'binary/'
        return 'float/'

    def _prepare_folders(self, output_folder, project_name, options):
        # create output folder if doesnt exists
        if output_folder[-1] != '/':
            output_folder += '/'

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output_folder += project_name+'/'

        if options.clean_output:
            shutil.rmtree(path=output_folder, ignore_errors=True)

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        self._dst_folder = os.path.abspath(output_folder)+'/'

    def _get_project_files(self, model_filename, options):

        project_files = list()

        # main file and files extensions
        if options.project_type in [ProjectType.C, ProjectType.CODEBLOCK]:
            (h_ext, c_ext) = ('.h', '.c')
            project_files.append('main.c')
        else:
            (h_ext, c_ext) = ('.h', '.cpp')
            if options.project_type != ProjectType.ARDUINO:
                project_files.append('main.cpp')

        # embedia files
        project_files.append('embedia'+c_ext)
        project_files.append('embedia'+h_ext)

        # model files
        project_files.append(model_filename+c_ext)
        project_files.append(model_filename+h_ext)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
        # fixed point files
        if options.data_type != ModelDataType.FLOAT and options.data_type != ModelDataType.BINARY:
            project_files.append('fixed'+c_ext)
            project_files.append('fixed'+h_ext)

        # debug file
        if options.debug_mode != DebugMode.DISCARD:
            project_files.append('embedia_debug'+c_ext)
            project_files.append('embedia_debug'+h_ext)
            project_files.append('embedia_debug_def'+h_ext)

        # test examples
        if options.example_data is not None:
            project_files.append('example_file'+h_ext)

        return project_files

    def print_layers_info(self, embedia_model, embedia_decl):
        layers_info = embedia_model.get_layers_info(embedia_decl)
        total_params = (0, 0)
        total_size = 0
        total_MACs = 0

        for i, (l_name, l_type, l_act, params, shape, MACs, size) in enumerate(layers_info):
            total_size += size
            total_MACs += MACs
            total_params = (total_params[0] + params[0], total_params[1] + params[1])
            size = '%8.3f' % (size/1024.0)
            param_str= '%d' % (params[0] + params[1])
            if params[1] > 0:
                param_str += '(%d)' % params[1]
            layer = l_type
            if l_act is not None:
                layer += f'({l_act})'
            layers_info[i] = (layer, l_name, param_str, shape, MACs, size)
        # print table
        table = PrettyTable()
        table.field_names = ['Layer(activation)', 'Name', '#Param(NT)', 'Shape', 'MACs', 'Size (KiB)']
        table.align['Layer(activation)'] = 'l'
        table.align['Name'] = 'l'
        table.align['#Param(NT)'] = 'r'
        table.align['MACs'] = 'r'
        table.align['Size (KB)'] = 'r'
        table.align['#Param'] = 'r'
        table.add_rows(layers_info)
        print(table)
        total_p = '%d' % (total_params[0] + total_params[1])
        if total_params[1] > 0:
            total_p += '(%d)' % total_params[1]
        print('Total params (NT)....: ' + total_p)
        print('Total size in KiB....: %.3f' % (total_size/1024.0))
        print('Total MACs operations: %.0f \n' % (total_MACs))
