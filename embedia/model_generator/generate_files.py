import os
import re
import numpy as np
from embedia.model_generator.project_options import (
        ModelDataType,
        ProjectType,
        DebugMode,
        BinaryBlockSize
    )
from embedia.utils import file_management
from embedia.layers.data_layer import DataLayer
from embedia.layers.activation.activation import Activation
from embedia.model_generator.project_options import BinaryBlockSize


def multi_replace(adict, text):
    # Create a regular expression from all of the dictionary keys
    regex = re.compile("|".join(map(re.escape, adict.keys())))

    # For each match, look up the corresponding value in the dictionary
    return regex.sub(lambda match: adict[match.group(0)], text)


def indent(multi_ln_code, level=1, spaces=4):

    # remove the whitespaces at the end of each line
    code = re.sub(r'[ \t]+(?=\n)', '', multi_ln_code)

    # check if the code has a trailing new line
    has_nl = code.endswith("\n")

    return re.sub("^", level*spaces*' ', code, flags=re.MULTILINE)


def generate_embedia_library(layers_embedia, src_folder, options):

    embedia_files = dict()

    filenames = os.listdir(src_folder)
    for filename in filenames:
        embedia_files[filename] = file_management.read_from_file(src_folder+filename)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
    # Prepare includes

    if options.data_type == ModelDataType.BINARY:

        if options.tamano_bloque == BinaryBlockSize.Bits8:
            tam_block = 8
        elif options.tamano_bloque == BinaryBlockSize.Bits16:
            tam_block = 16
        elif options.tamano_bloque == BinaryBlockSize.Bits32:
            tam_block = 32
        else:
            tam_block = 64

        if options.project_type == ProjectType.ARDUINO:
            includes_h = '#include "Arduino.h"\n'
            includes_h += f'\n#define tamano_del_bloque {tam_block}\n'

        else:
            includes_h = '#include <stdlib.h>\n'
            includes_h += f'\n#define tamano_del_bloque {tam_block}\n'

        embedia_files['embedia.h'] = multi_replace({'{includes}': includes_h}, embedia_files['embedia.h'])

    else:

        if options.project_type == ProjectType.ARDUINO:
            includes_h = '#include "Arduino.h"\n'
        else:
            includes_h = '#include <stdlib.h>\n'

        embedia_files['embedia.h'] = multi_replace({'{includes}': includes_h}, embedia_files['embedia.h'])


    return embedia_files


# Warning TO DO: check width vs height order
def get_input_const(input_shape):
    if len(input_shape) == 3:
        return {'INPUT_CHANNELS': input_shape[2], 'INPUT_WIDTH': input_shape[0], 'INPUT_HEIGHT': input_shape[1]}
    elif len(input_shape) == 2:
        return {'INPUT_WIDTH': input_shape[0], 'INPUT_HEIGHT': input_shape[1]}
    elif len(input_shape) == 1:
        return {'INPUT_LENGTH': input_shape[0]}

    return None


def generate_embedia_model(layers_embedia, src_folder, model_name, options):

    def format_model_name(model_name):
        model_name = model_name.lower()
        if not model_name.endswith('model'):
            model_name += '_model'
        return model_name

    filename = format_model_name(model_name)

    src_h = os.path.join(src_folder, 'model/model.h')
    src_c = os.path.join(src_folder, 'model/model.c')

    includes = f'#include "{filename}.h"\n'
    if options.debug_mode != DebugMode.DISCARD:
        includes += '#include "embedia_debug.h"\n'

    model_name_h = filename.upper()
    # macros_first_shape = layers_embedia[0].get_macros_first_shape()
    input_data_type = layers_embedia[0].get_input_data_type()
    output_data_type = layers_embedia[-1].get_output_data_type()
    input_shape = layers_embedia[0].get_input_shape()

    # prepare input dimension constant
    input_dict = get_input_const(input_shape)
    input_const = ""
    total_size = 1
    for k in input_dict:
        total_size *= input_dict[k]
        input_const += f'#define {k} {input_dict[k]}\n'
    input_const += f'\n#define INPUT_SIZE {total_size}\n'

    prototypes_init = ""
    var = ""
    init = ""
    functions_init = ""
    predict = ""

    data_layers_input = [{'type': input_data_type, 'var_name': 'input'}, ]
    data_layers_output = []
    layer_id = -1

    for layer in layers_embedia:
        layer_id += 1

        # Initialization
        if isinstance(layer, DataLayer):
            prototypes_init += layer.prototypes_init()
            var += layer.var()
            init += layer.init()
            functions_init += layer.functions_init()

        # layer section of predict function

        predict += f'\n//*************** LAYER {layer_id} **************//'
        predict += f'\n// Layer name: {layer.name}\n'

        if not layer.inplace_output:

            input_layer_type = layer.get_input_data_type()
            if data_layers_input[-1]['type'] != input_layer_type:
                var_input = f'input{len(data_layers_input)}'
                predict += f'{input_layer_type} {var_input};\n'
                data_layers_input.append({'type': input_layer_type, 'var_name': var_input})

            if layer != layers_embedia[0]:
                predict += f'{data_layers_input[-1]["var_name"]} = {data_layers_output[-1]["var_name"]};\n'

            output_layer_type = layer.get_output_data_type()
            if data_layers_output == [] or data_layers_output[-1]['type'] != output_layer_type:
                var_output = f'output{len(data_layers_output)}'
                predict += f'{output_layer_type} {var_output};\n'
                data_layers_output.append({'type': output_layer_type, 'var_name': var_output})

        param_in = data_layers_input[-1]['var_name']
        param_out = data_layers_output[-1]['var_name']
        predict += f'{layer.predict(param_in, param_out)}\n'

        # Add activation function
        act_fn = layer.activation_function(var_output)
        if act_fn != '':
            if not isinstance(layer, Activation):
                predict += f'// Activation layer for {layer.name}\n'
            predict += f'{act_fn}\n'

        # Add debug function if is enabled
        if options.debug_mode != DebugMode.DISCARD:
            dbg_fn = layer.debug_function(var_output)
            predict += f'// Debug function for layer {layer.name}\n'
            predict += f'{dbg_fn}\n'

    # indent code
    predict = indent(predict)
    if output_data_type == 'data1d_t':
        predict_class = 'return argmax(*results);'
    else:
        predict_class = '''//TO DO: argmax with data2d_t and data3d_t
    return -1; '''

    h = file_management.read_from_file(src_h).format(
            model_name_h=model_name_h,
            input_const=input_const,
            input_data_type=input_data_type,
            output_data_type=output_data_type
        )

    c = file_management.read_from_file(src_c).format(
            includes=includes,
            filename=filename,
            prototypes_init=prototypes_init,
            var=var,
            init=init,
            predict=predict,
            predict_class=predict_class,
            functions_init=functions_init,
            input_data_type=input_data_type,
            output_data_type=output_data_type,
            output_name=var_output
        )

    return (h, c, filename)


def generate_embedia_main(layers_embedia, src_folder, filename, options):

    src_c = os.path.join(src_folder, 'main/main_')

    # Prepare includes
    if options.project_type == ProjectType.ARDUINO:
        src_c += "arduino.c"
        includes_c = '#include "Arduino.h"\n'
        baud_rate = str(options.baud_rate)
    else:
        src_c += "c.c"
        includes_c = '#include <stdio.h>\n'
        baud_rate = "\n"

    includes_c += '#include "embedia.h"\n'
    includes_c += '#include "'+filename+'.h"\n'

    example_var_name = 'sample_data'

    if options.example_data is None:
        main_code = ''
    else:
        includes_c += '#include "example_file.h"\n'
        main_code = f'''
    // sample intitialization
    input.data = {example_var_name};
'''

    main_code += '''
    // model initialization
    model_init();
'''

    # prepare data for model input and output
    input_data_type = layers_embedia[0].get_input_data_type()
    output_data_type = layers_embedia[-1].get_output_data_type()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado no se usa igual por ahora

    # (FLOAT, FIXED32, FIXED16, FIXED8, BINARY) = (0,1,2,3,4)
    if options.data_type == ModelDataType.FLOAT or options.data_type == ModelDataType.BINARY:
        model_data_type = 'float'
    else:
        model_data_type = 'fixed'


    input_const = get_input_const(layers_embedia[0].get_input_shape())
    input_dim = ''
    for k in input_const:
        input_dim += f'{k}, '

    input_data = f'''{input_data_type} input = {{ {input_dim} NULL}};\n'''
    output_data = f'''{output_data_type} results;\n'''

    main_code += '''

    // make model prediction
    // uncomment corresponding code

    int prediction = model_predict_class(input, &results);

    // print predicted class id'''

    if options.project_type == ProjectType.ARDUINO:
        main_code += '''
    Serial.print("Prediction class id: ");
    Serial.println(prediction);
'''
        if options.example_data is not None:
            main_code += '''
    Serial.print("   Example class id: ");
    Serial.println(sample_data_id);
'''
    else:
        main_code += '''
    printf("Prediction class id: %d\\n", prediction);
'''
        if options.example_data is not None:
            main_code += '''
    printf("   Example class id: %d\\n", sample_data_id);
'''

    main_code += '''
    /*

    model_predict(input, &results);
    printf("prediccion: %5f", results.data[0]);
    */

'''

    c = file_management.read_from_file(src_c).format(includes=includes_c,
                                                     input_data=input_data,
                                                     output_data=output_data,
                                                     baud_rate=baud_rate,
                                                     main_code=main_code)

    # load and generate data example if it corresponds
    if options.example_data is not None:
        h = generate_examples(src_folder, example_var_name, options)
    else:
        h = None

    return (h, c)


def generate_codeblock_project(project_name, files, src_folder):

    included_files = ''
    for filename in files:
        if filename[-2:].lower() == '.c':
            included_files += f'''
        <Unit filename="{filename}">
            <Option compilerVar="CC" />
        </Unit>'''
        elif filename[-2:].lower() == '.h':
            included_files += f'''
        <Unit filename="{filename}" />'''

    src_cbp = os.path.join(src_folder, 'main/codeblock_project.cbp')
    content = file_management.read_from_file(src_cbp)

    content = multi_replace({'{project_name}': project_name, '{included_files}': included_files}, content)

    return content


def data_to_array_str(data, macro_converter, clip=120):
    output = ''
    cline = '  '
    for i in data.flatten():
        cline += macro_converter(str(i)) + ', '
        if len(cline) > clip:
            output += cline + '\n'
            cline = '  '
    output += cline
    return output[:-2]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
def generate_examples(src_folder, var_name, options):

    if options.data_type == ModelDataType.FLOAT or options.data_type == ModelDataType.BINARY:
        def conv(s):
            return s
        data_type = 'float'
    else:
        def conv(s):
            return f"FL2FX({s})"
        data_type = 'fixed'

    src_h = os.path.join(src_folder, 'main/example_file.h')
    smp = options.example_data
    ids = options.example_ids

    if smp.shape[0] != ids.shape[0]:
        raise Exception("The number of examples does not match the number of classes")

    examples = f'''
#define MAX_SAMPLE {smp.shape[0]-1}

#define SELECT_SAMPLE 0

'''
    if not isinstance(smp, np.ndarray):
        smp = np.array(smp)
    for i in range(len(smp)):
        data = smp[i].flatten()
        # print(data)
        id = ids[i]
        examples += f'''#if SELECT_SAMPLE == {i}
uint16_t {var_name}_id = {id};

static {data_type} {var_name}[]= {{
{data_to_array_str(data, conv)}
}};

#endif
'''

    content = file_management.read_from_file(src_h).format(examples=examples)

    return content
