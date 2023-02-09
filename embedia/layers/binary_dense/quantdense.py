from embedia.layers.data_layer import DataLayer
from embedia.utils.c_helper import declare_array
from embedia.utils.c_helper import declare_array2
from embedia.model_generator.project_options import BinaryBlockSize
from embedia.model_generator.project_options import ModelDataType
import larq as lq
import math


class QuantDense(DataLayer):
    """
    The Dense layer is a layer that requires additional data (weights to be
    initialized) in addition to the input data. For this reason it inherits
    from DataLayer which generates C code automatically with the variable that
    will store the data, the declaration of the prototype of the initialization
    function and the call to it.
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
    Ex: As this class is called Dense, the type of the additional structure
    will be called "dense_datat" and must be defined previously in the
    "embedia.h" file.
    If the name of the layer is dense0, it will automatically be generated in
    the C file of the model, the declaration of the variable
    "dense_datat dense0_data", the prototype of the initialization function
    "dense_datat init_dense0_data(void)" and the invocation
    "dense0_data = init_dense0_data()". This way of naming must be taken into
    account in the implementation of the initialization function in the
    "functions_init" method
    """

    def __init__(self, model, layer, options=None, **kwargs):
        super().__init__(model, layer, options, **kwargs)
        self.input_data_type = "data1d_t"
        self.output_data_type = "data1d_t"

        # assign properties to be used in "functions_init"
        self.weights = layer.get_weights()[0]
        self.biases = layer.get_weights()[1]

        # verificamos a que caso corresponde
        with lq.context.quantized_scope(True):
            if (layer.get_config()['input_quantizer'] is None) and (layer.get_config()['kernel_quantizer'] is None):
                # es una desnse normal
                self.tipo_densa = 0
            elif (layer.get_config()['input_quantizer'] is None) and (layer.get_config()['kernel_quantizer'] is not None):

                # entrada no binaria
                print(
                    f"Error: No support for layer {layer} with this arguments")
                raise f"Error: No support for layer {layer} with this arguments"

            elif (layer.get_config()['input_quantizer'] is not None) and (layer.get_config()['kernel_quantizer'] is not None):
                if (layer.get_config()['input_quantizer']['class_name'] == 'SteSign') and (layer.get_config()['kernel_quantizer']['class_name'] == 'SteSign'):
                    # dnse pura binaria
                    self.tipo_densa = 1
                else:
                    print("Error: No support for layer {layer} with this arguments")
                    raise "Error: No support for layer {layer} with this arguments"
            else:
                print("Error: No support for layer {layer} with this arguments")
                raise "Error: No support for layer {layer} with this arguments"

    def var(self):
        if self.tipo_densa == 0:

            return f"dense_layer_t {self.name}_data;\n"

        else:

            return f"quantdense_layer_t {self.name}_data;\n"

    def prototypes_init(self):
        if self.tipo_densa == 0:

            return f"dense_layer_t init_{self.name}_data(void);\n"
        else:
            return f"quantdense_layer_t init_{self.name}_data(void);\n"

    def calculate_MAC(self):
        """
        calculates amount of multiplication and accumulation operations
        Returns
        -------
        int
            amount of multiplication and accumulation operations

        """
        # layer dimensions
        (n_input, n_neurons) = self.weights.shape

        MACs = n_input * n_neurons

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
        (n_input, n_neurons) = self.weights.shape

        # base data type in bits: float, fixed (32/16/8), binary 1
        dt_size = ModelDataType.get_size(self.options.data_type)

        # neuron structure size
        if(self.tipo_densa == 0):  # densa float
            sz_neuron_t = types_dict['neuron_t']
            dt_size = dt_size*32
        else:
            sz_neuron_t = types_dict['quant_neuron_t']
            if self.options.tamano_bloque == BinaryBlockSize.Bits8:
                dt_size = dt_size*8
            elif self.options.tamano_bloque == BinaryBlockSize.Bits16:
                dt_size = dt_size*16
            elif self.options.tamano_bloque == BinaryBlockSize.Bits32:
                dt_size = dt_size*32
            else:
                dt_size = dt_size*64

        if self.tipo_densa == 0:
            mem_size = (n_input * dt_size/8 + sz_neuron_t) * n_neurons
        else:
            mem_size = (math.ceil(n_input/dt_size) *
                        dt_size/8 + sz_neuron_t) * n_neurons

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

        weights = self.weights
        biases = self.biases
        name = self.name

        # contemplar los dos casos
        if self.tipo_densa == 0:
            # densa normal

            struct_type = 'dense_layer_t'
            (data_type, macro_converter) = self.model.get_type_converter()

            (n_input, n_neurons) = weights.shape

            init_dense_layer = f'''
    {struct_type} init_{name}_data(void){{

        static neuron_t neurons[{n_neurons}];
    '''
            o_code = ""

            for neuron_id in range(n_neurons):

                o_weights = declare_array(
                    f'static  {data_type}', f'weights{neuron_id}', macro_converter, weights[:, neuron_id])

                o_code += f'''
        {o_weights};
        static const neuron_t neuron{neuron_id} = {{weights{neuron_id}, {macro_converter(biases[neuron_id])}}};
        neurons[{neuron_id}]=neuron{neuron_id};
    '''
            init_dense_layer += o_code

            init_dense_layer += f'''
        dense_layer_t layer= {{{n_neurons}, neurons}};
        return layer;
    }}
    '''

        else:
            # densa full binaria
            struct_type = 'quantdense_layer_t'
            (data_type, macro_converter) = self.model.get_type_converter()

            (n_input, n_neurons) = weights.shape
            toti = weights[:, 0].size

            lista_contadores = [0, 0, 0]  # suma,cont,cont2

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

            init_dense_layer = f'''
    {struct_type} init_{name}_data(void){{

        static quant_neuron_t neurons[{n_neurons}];
    '''
            o_code = ""

            for neuron_id in range(n_neurons):

                lista_contadores[0] = 0  # suma
                lista_contadores[1] = 0  # cont
                lista_contadores[2] = 0  # cont2

                o_weights = declare_array2(
                    toti, xBits, lista_contadores, f'static  {block_type}', f'weights{neuron_id}', macro_converter, weights[:, neuron_id])

                o_code += f'''
        {o_weights};
        static const quant_neuron_t neuron{neuron_id} = {{weights{neuron_id}, {macro_converter(biases[neuron_id])}}};
        neurons[{neuron_id}]=neuron{neuron_id};
    '''
            init_dense_layer += o_code

            init_dense_layer += f'''
        quantdense_layer_t layer= {{{n_neurons}, neurons}};
        return layer;
    }}
    '''

        return init_dense_layer

    def predict(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "embedia.c" and by convention should be called
        "class name" + "_layer".
        For example, for the EmbedIA Dense class associated to the Keras
        Dense layer, the function "dense_layer" must be implemented in
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

        if self.tipo_densa == 0:
            # densa normal
            return f'''dense_layer({self.name}_data, {input_name}, &{output_name});'''

        elif self.tipo_densa == 1:
            # densa binaria
            return f'''quantdense_layer({self.name}_data, {input_name}, &{output_name});'''
