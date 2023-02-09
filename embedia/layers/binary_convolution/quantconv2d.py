
from embedia.layers.data_layer import DataLayer
from embedia.model_generator.project_options import ModelDataType
from embedia.model_generator.project_options import BinaryBlockSize
from embedia.utils.binary_helper import BinaryGlobalMask

import numpy as np

import larq as lq

import math


class QuantConv2D(DataLayer):
    """
    The Conv2D convolutional layer is a layer that requires additional data
    (weights to be initialized) in addition to the input data. For this reason
    it inherits from DataLayer which generates C code automatically with the
    variable that will store the data, the declaration of the prototype of the
    initialization function and the call to it.
    Normally the programmer must implement two methods. The first one is
    "functions_init" which returns the implementation of the initialization
    function in C code, retrieving the layer information and dumping it into
    the structure (defined in embedia.h") in an appropriate way. The second one
    is "predict" where the programmer must invoke the EmbedIA function
    (implemented in "embedia.c") that must perform the processing of the layer.
    To avoid overlapping names, both the function name and the variable name
    are generated automatically using the layer name. The same happens with the
    data type of the structure to be completed whose name comes from the name
    of the Python class that implements the layer.
    Ex: As this class is called Conv2D, the type of the additional structure
    will be called "conv2d_datat" and must be defined previously in the
    "embedia.h" file.
    If the name of the layer is conv2d0, it will automatically be generated in
    the C file of the model, the declaration of the variable
    "conv2d_datat conv2d0_data", the prototype of the initialization function
    "conv2d_datat init_conv2d0_data(void)" and the invocation
    "conv2d0_data = init_conv2d0_data()". This way of naming must be taken into
    account in the implementation of the initialization function in the
    "functions_init" method    """

    def __init__(self, model, layer, options=None, **kwargs):
        super().__init__(model, layer, options, **kwargs)
        self.input_data_type = "data3d_t"
        self.output_data_type = "data3d_t"

        # assign properties to be used in "functions_init"
        self.weights = self._adapt_weights(layer.get_weights()[0])
        self.biases = layer.get_weights()[1]

        # verificamos a que caso de los tres corresponde...
        with lq.context.quantized_scope(True):
            if (layer.get_config()['input_quantizer'] is None) and (layer.get_config()['kernel_quantizer'] is None):

                # es una conv normal
                self.tipo_conv = 0

            elif (layer.get_config()['input_quantizer'] is None) and (layer.get_config()['kernel_quantizer'] is not None):
                if (layer.get_config()['kernel_quantizer']['class_name'] == 'SteSign'):
                    # conv binaria entrada no binarizada
                    self.tipo_conv = 1
                else:
                    print("Error: No support for layer with this arguments")
                    raise "Error: No support for layer with this arguments"

            elif (layer.get_config()['input_quantizer'] is not None) and (layer.get_config()['kernel_quantizer'] is not None):
                if (layer.get_config()['input_quantizer']['class_name'] == 'SteSign') and (layer.get_config()['kernel_quantizer']['class_name'] == 'SteSign'):
                    # conv pura binaria
                    self.tipo_conv = 2
                else:
                    print("Error: No support for layer with this arguments")
                    raise "Error: No support for layer with this arguments"
            else:
                print("Error: No support for layer with this arguments")
                raise "Error: No support for layer with this arguments"

    def var(self):
        if self.tipo_conv == 0:

            return f"conv2d_layer_t {self.name}_data;\n"

        else:

            return f"quantconv2d_layer_t {self.name}_data;\n"

    def prototypes_init(self):
        if self.tipo_conv == 0:

            return f"conv2d_layer_t init_{self.name}_data(void);\n"
        else:
            return f"quantconv2d_layer_t init_{self.name}_data(void);\n"

    def _adapt_weights(self, weights):
        _row, _col, _chn, _filt = weights.shape
        arr = np.zeros((_filt, _chn, _row, _col))
        for row, elem in enumerate(weights):
            for column, elem2 in enumerate(elem):
                for channel, elem3 in enumerate(elem2):
                    for filters, value in enumerate(elem3):
                        arr[filters, channel, row, column] = value
        return arr

    def calculate_MAC(self):
        # layer dimensions
        n_filters, n_channels, n_rows, n_cols = self.weights.shape

        # estimate amount multiplication and accumulation operations

        out_size = self.get_output_size()
        # MACs =  (n_rows * n_cols *  n_filters) * in_size
        MACs = out_size*n_cols*n_rows*n_channels

        return MACs

    def calculate_memory(self, types_dict):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """

        # layer dimensions
        n_filters, n_channels, n_rows, n_cols = self.weights.shape

        # base data type in bits: float, fixed (32/16/8), binary 1
        dt_size = ModelDataType.get_size(self.options.data_type)

        # EmbedIA filter structure size
        if(self.tipo_conv == 0):  # conv2d float
            sz_filter_t = types_dict['filter_t']
            dt_size = dt_size*32
        else:  # semi-cuantizada o full binary
            sz_filter_t = types_dict['quant_filter_t']
            if self.options.tamano_bloque == BinaryBlockSize.Bits8:
                dt_size = dt_size*8
            elif self.options.tamano_bloque == BinaryBlockSize.Bits16:
                dt_size = dt_size*16
            elif self.options.tamano_bloque == BinaryBlockSize.Bits32:
                dt_size = dt_size*32
            else:
                dt_size = dt_size*64

        if(self.tipo_conv == 0):
            mem_size = (n_channels * n_rows * n_cols *
                        dt_size / 8 + sz_filter_t) * n_filters
        else:
            mem_size = (math.ceil((n_channels * n_rows * n_cols) /
                        dt_size) * dt_size / 8 + sz_filter_t) * n_filters

        return mem_size

    def functions_init(self):
        """
        Generate C code with the initialization function of the additional
        structure (defined in "embedia.h") required by the layer.
        Note: it is important to note the automatically generated function
        prototype (defined in the DataLayer class).

        Returns
        -------
        str
            C function for data initialization
        """

        # contemplamos los tres casos posibles

        if self.tipo_conv == 0:
            # conv normal

            (data_type, macro_converter) = self.model.get_type_converter()

            n_filters, n_channels, n_rows, n_cols = self.weights.shape
            assert n_rows == n_cols  # WORKING WITH SQUARE KERNELS FOR NOW
            kernel_size = n_rows  # Defining kernel size

            ret = ""
            struct_type = 'conv2d_layer_t'
            name = self.name+'_data'
            text = f'''

    {struct_type} init_{name}(void){{

            static filter_t filters[{n_filters}];
            '''
            for i in range(n_filters):
                o_weights = ""
                for ch in range(n_channels):
                    for f in range(n_rows):
                        o_weights += '\n    '
                        for c in range(n_cols):
                            o_weights += f'''{macro_converter(self.weights[i, ch, f, c])}, '''
                    o_weights += '\n'

                o_code = f'''
            static const {data_type} weights{i}[]={{ {o_weights}
            }};
            static filter_t filter{i} = {{{n_channels}, {kernel_size}, weights{i}, {macro_converter(self.biases[i])}}};
            filters[{i}]=filter{i};
                '''
                text += o_code
            text += f'''
            conv2d_layer_t layer = {{{n_filters},filters}};
            return layer;
            }}
            '''
            ret += text
            # fin caso 0

        else:

            # conv binaria entrada no binary o binaria pura
            (data_type, macro_converter) = self.model.get_type_converter()

            n_filters, n_channels, n_rows, n_cols = self.weights.shape
            assert n_rows == n_cols  # WORKING WITH SQUARE KERNELS FOR NOW
            kernel_size = n_rows  # Defining kernel size

            largo_total = (n_rows)*(n_channels)*(n_cols)

            if self.options.tamano_bloque == BinaryBlockSize.Bits8:
                xBits = 8
                block_type = 'uint8_t'
            elif self.options.tamano_bloque == BinaryBlockSize.Bits16:
                xBits = 16
                block_type = 'uint16_t'
            elif self.options.tamano_bloque == BinaryBlockSize.Bits32:
                xBits = 32
                block_type = 'uint32_t'
            else:
                xBits = 64
                block_type = 'uint64_t'

            ret = ""
            struct_type = 'quantconv2d_layer_t'
            name = self.name+'_data'
            text = f'''

          {struct_type} init_{name}(void){{

            static quant_filter_t filtros_b[{n_filters}];
            '''
            for i in range(n_filters):
                cont = 0
                suma = 0
                o_weights = ""
                for ch in range(n_channels):
                    for f in range(n_rows):
                        for c in range(n_cols):
                            num = self.weights[i, ch, f, c]
                            if xBits == 16:
                                if num == 1.0:
                                    suma += (BinaryGlobalMask.get_mask_16()
                                             )[cont]
                            elif xBits == 32:
                                if num == 1.0:
                                    suma += (BinaryGlobalMask.get_mask_32()
                                             )[cont]
                            elif xBits == 64:
                                if num == 1.0:
                                    suma += (BinaryGlobalMask.get_mask_64()
                                             )[cont]
                            else:
                                if num == 1.0:
                                    suma += (BinaryGlobalMask.get_mask_8()
                                             )[cont]

                            if cont == xBits-1 or ((ch+1)*(f+1)*(c+1) == largo_total):

                                o_weights += f'''{macro_converter(suma)},'''
                                cont = 0
                                suma = 0

                            else:
                                cont += 1

                o_weights = o_weights[:-1]  # remuevo la ultima coma
                o_code = f'''
            static const {block_type} weights{i}[]={{{o_weights}
            }};
            static quant_filter_t filter{i} = {{{n_channels}, {kernel_size}, weights{i}, {macro_converter(self.biases[i])}}};
            filtros_b[{i}]=filter{i};
              '''
                text += o_code
            text += f'''
            {struct_type} layer = {{{n_filters},filtros_b}};
            return layer;
          }}
            '''

            ret += text

        return ret

    def predict(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "embedia.c" and by convention should be called
        "class name" + "_layer".
        For example, for the EmbedIA Conv2D class associated to the Keras
        Conv2D layer, the function "conv2d_layer" must be implemented in
        "embedia.c"

        Parameters
        ----------
        input_name : str
            name of the input variable to be used in the invocation of the C
            function that implements the layer.
        output_name : str
            name of the output variable to be used in the invocation of the C
            function that implements the layer.

        Returns
        -------
        str
            C code with the invocation of the function that performs the
            processing of the layer in the file "embedia.c".

        """

        if len(self.get_input_shape()) >= 3 and self.model.firstLayerOfItsclass(self):
            code = f'''    // convert image for first EmbedIA Conv2d layer
    image_adapt_layer({input_name}, &{output_name});
    {input_name} = {output_name};

 '''
        else:
            code = ''

        if self.tipo_conv == 0:
            # conv normal
            code += f'''conv2d_layer({self.name}_data, {input_name}, &{output_name});
    '''

        elif self.tipo_conv == 1:
            # conv binaria entrada no binarizada
            code += f'''quantconv2d_input_not_binary_layer({self.name}_data, {input_name}, &{output_name});
    '''

        else:
            # conv binaria pura
            code += f'''quantconv2d_layer({self.name}_data, {input_name}, &{output_name});'''

        return code
