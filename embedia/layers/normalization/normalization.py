from embedia.layers.data_layer import DataLayer
from embedia.utils.c_helper import declare_array
from embedia.model_generator.project_options import ModelDataType

class Normalization(DataLayer):

    """
    The normalization layer is a layer that requires additional data
    (coefficients and averages values to be initialized) in addition to the
    input data. For this reason it inherits from DataLayer which generates C
    code automatically with the variable that will store the data, the
    declaration of the prototype of the initialization function and the call
    to it.
    Normally the programmer must implement two methods. The first one is
    "functions_init" which returns the implementation of the initialization
    function in C code, retrieving the layer information and dumping it into
    the structure (defined in embedia.h") appropriately. The second one is
    "predict", where the programmer must invoke the function EmbedIA function
    (implemented in "embedia.c") that should perform the layer processing.
    To avoid overlapping names, both the function name and the variable name
    are automatically generated using the layer name. The same is true for the
    data type of the structure to be completed whose name comes from the name
    of the Python class that implements the layer.
    Ex: Since this class is called Normalization, the type of the additional
    structure must be called "normalization_datat" and must be previously
    defined in the "embedia.h" file.
    If the name of the layer is normalization0, it will be automatically
    generated in the model's C file, the declaration of the variable
    "normalization_datat normalization0_data", the prototype of the
    initialization function "normalization_datat init_normalization0_data(void)"
    and the invocation "normalization0_data = init_normalization0_data()".
    This way of naming must be taken into account in the implementation of the
    initialization function in the "functions_init" method.
    This class implements the operation of normalizations that use averages and
    coefficients ([average-value]/coefficient ). The classes that inherit from
    this one must only fill the values of the "sub_values" and "div_values"
    properties in their constructor.
"""

    # these properties must be defined in the constructor of subclass
    sub_values = None
    div_values = None

    # Constructor receives sklearn normalization object (Scaler) in layer
    def __init__(self, model, layer, options=None, **kwargs):
        super().__init__(model, layer, options, **kwargs)

        # As the name generated automatically depends on the class name and the
        # same structure is used for all normalizations, the EmbedIA data type
        # name is forced in this class.
        self.struct_data_type = 'normalization_t'

        # Name of EmbedIA normalization function declared in "embedia.h".
        # Subclass must assign the propertly name
        # self.norm_function_name = '?'

    def get_input_shape(self):
        """
        Returns the shape of the input data. This method is redefined because
        SKLearn "Scalers" do not have the "input_shape" property of the Keras
        layers on which the original implementation is based.

        Returns
        -------
        n-tuple
            shape of the input data
        """
        if self.div_values is None:
            return self.sub_values.shape
        return self.div_values.shape

    def get_output_shape(self):
        """
        Returns the shape of the output data.

        Returns
        -------
        n-tuple
            shape of the output data
        """
        return self.get_input_shape()

    def calculate_MAC(self):
        """
        calculates amount of multiplication and accumulation operations
        Returns
        -------
        int
            amount of multiplication and accumulation operations

        """
        MACs = self.get_input_shape()[0]

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
        n_input = self.get_input_shape()[0]

        # neuron structure size
        sz_struct_t = types_dict[self.struct_data_type]

        # base data type in bits: float, fixed (32/16/8)
        dt_size = ModelDataType.get_size(self.options.data_type)
        if self.options.data_type == ModelDataType.BINARY:
            dt_size = 32

        mem_size = n_input * dt_size/8 + sz_struct_t

        return mem_size

    def functions_init(self):

        (data_type, macro_converter) = self.model.get_type_converter()
        name = self.name
        struct_type = self.struct_data_type
        sub_var_name = 'sub_val'
        div_var_name = 'inv_div_val'

        sub_val = self.sub_values
        div_val = self.div_values

        # generate values for substraction.
        # Params: data type, var name, macro, array/list of values
        o_sub_val = declare_array(f'static const {data_type}', sub_var_name, macro_converter, sub_val)

        # inverted values in order to multiply instead of divide
        div_val = ['1/%f' % v for v in div_val]

        # prepare values for division (multiplication of inverse values)
        o_div_val = declare_array(f'static const {data_type}', div_var_name, macro_converter, div_val)

        init_layer = f'''
{struct_type} init_{name}_data(void){{

    {o_sub_val};
    {o_div_val};

    static const {struct_type} norm = {{ {sub_var_name}, {div_var_name} }};
    return norm;
}}
'''
        return init_layer

    def predict(self, input_name, output_name):
        return f'''{self.norm_function_name}({self.name}_data, {input_name}, &{output_name});'''
