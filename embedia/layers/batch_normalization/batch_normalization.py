from embedia.layers.data_layer import DataLayer
from embedia.utils.c_helper import declare_array
from embedia.model_generator.project_options import ModelDataType


class BatchNormalization(DataLayer):

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

    # Constructor receives batch normalization object in layer
    def __init__(self, model, layer, options=None, **kwargs):
        super().__init__(model, layer, options, **kwargs)

        self.gamma = layer.get_weights()[0]
        self.beta = layer.get_weights()[1]
        self.moving_mean = layer.get_weights()[2]
        self.moving_variance = layer.get_weights()[3]
        self.epsilon = layer.epsilon

        # As the name generated automatically depends on the class name and the
        # same structure is used for all normalizations, the EmbedIA data type
        # name is forced in this class.
        self.struct_data_type = 'batch_normalization_layer_t'

        # EmbedIA function saves output into input
        self.inplace_output = True

    def calculate_memory(self, types_dict):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """

        # layer dimensions
        # batch norm has 4 data array: beta, gamma, moving mean and moving
        # variance
        n_features = len(self.gamma)
        n_arrays = 4 - 1  # gamma is omited by optimization

        # neuron structure size
        sz_batch_norm_t = types_dict['batch_normalization_layer_t']

        # base data type: float, fixed (32/16/8)
        dt_size = ModelDataType.get_size(self.options.data_type)
        if self.options.data_type == ModelDataType.BINARY:
            dt_size = 32
        mem_size = n_arrays * n_features * dt_size/8 + sz_batch_norm_t

        return mem_size

    def functions_init(self):

        (data_type, macro_converter) = self.model.get_type_converter()
        name = self.name
        struct_type = self.struct_data_type
        mov_mean_name = 'mov_mean'
        inv_gamma_dev_name = 'inv_gamma_dev'
        beta_name = 'beta'
        gamma_name = 'gamma'
        length = len(self.moving_mean)

        # Params: data type, var name, macro, array/list of values
        array_type = f'static const {data_type}'

        # gamma can be eliminated due to the optimization performed below
        # o_gamma = declare_array(array_type, gamma_name, macro_converter, self.gamma)
        o_beta = declare_array(array_type, beta_name, macro_converter, self.beta)
        o_mov_mean = declare_array(array_type, mov_mean_name, macro_converter, self.moving_mean)

        # Optimization to avoid a multiplication, a division and square root
        # calculation in the microcontroller
        # epsilon is small value to avoid division by zero
        #gamma_variance = np.array([(gamma[i] / sqrt(moving_variance[i] + epsilon)) for i in range(gamma.size)])
        inv_gamma_dev = ['%f/sqrt(%f+%f)' % (self.gamma[i], self.moving_variance[i], self.epsilon) for i in range(self.gamma.size)]

        # get inverse of standard dev (square root of moving variance)
        o_inv_mov_std = declare_array(array_type, inv_gamma_dev_name, macro_converter, inv_gamma_dev)


        init_layer = f'''
{struct_type} init_{name}_data(void){{

    {o_beta};
    {o_mov_mean};
    {o_inv_mov_std};

    static const {struct_type} norm = {{ {length}, {beta_name}, {mov_mean_name}, {inv_gamma_dev_name} }};
    return norm;
}}
'''
        return init_layer

    def predict(self, input_name, output_name):
        dim = len(self.get_input_shape())
        return f'''batch_normalization{dim}d_layer({self.name}_data, &{output_name});'''
