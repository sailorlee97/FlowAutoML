import regex as re
import numpy as np
import tensorflow.keras.backend as K
from embedia.layers.activation.activation_functions import ActivationFunctions


class UnsupportedFeatureError(Exception):
    def __init__(self, obj, feature):
        super().__init__(
            f"EmbedIA feature ({feature}) not implemented for {str(type(obj))}"
            )
        self.object = obj


class LayerInfo(object):
    """
    This class defines a container for layer information:
        name, type name, activation function name, #params, MACs, output shape
    """

    def __init__(self, embedia_layer, types_dict):
        self.set_layer(embedia_layer, types_dict)

    def set_layer(self, layer, types_dict):
        self.layer = layer
        self.types_dict = types_dict

        self._update_properties()

    def _update_properties(self):
        k_layer = self.layer.layer
        self.class_name = k_layer.__class__.__name__
        self.layer_name = k_layer.name
        if hasattr(k_layer, 'activation') and k_layer.activation is not None:
            activation = ActivationFunctions(None, k_layer.activation)
            self.activation = activation.get_function_name()
        else:
            self.activation = None

        self.output_shape = self.layer.get_output_shape()

        trainable = int(np.sum([K.count_params(p) for p in k_layer.trainable_weights]))

        non_trainable = int(np.sum([K.count_params(p) for p in k_layer.non_trainable_weights]))
        self.params = (trainable, non_trainable)

        self.macs_ops = self.layer.calculate_MAC()

        self.memory = self.layer.calculate_memory(self.types_dict)


class Layer(object):
    """
    This class defines the basic behavior of an EmdedIA layer/element. It
    defines methods to obtain information related to the inputs and outputs of
    the layer/element such as its shape, number of elements, EmbedIA data type.
    It also implements methods to generate the C code necessary for activation
    functions, debugging function and invocation of the C function associated
    to the layer/element.
    """
    # these properties must be defined in the constructor of subclass
    input_data_type = ""   # C type of the layer input variable
    output_data_type = ""  # C type of the layer output variable

    def __init__(self, model, layer, options=None, **kwargs):
        """
        Constructor that receives:
            - EmbedIA Model
            - object (Keras, SkLearn, etc.) associated to the EmbedIA layer
            - EmbedIA project options
        Parameters
        ----------
        layer : object
            layer/object is associated to this EmbedIA layer/element. For
            example, it can receive a Keras layer or a SkLearn scaler.
        options : ModelDataType, optional
            options for EmbedIA project. It contains information about the type
            of representation data, normalization object, debugging level, etc.
            The default is None.
        Returns
        -------
        None.
        """
        super().__init__(**kwargs)
        self.model = model
        self.layer = layer

        # assign layer name from layer/element associated, if it's possible
        self.name = model.get_unique_name(layer)

        self.options = options  # Configuration options

        # When the value of this property is "true" it indicates that the
        # layer/element can process the output result on the same input
        # parameter. A typical case are layers that perform normalization
        # or activation functions.
        self.inplace_output = False

        self.set_default_layer_types()

    def get_type_name(self):
        """
        get the name of this class. Usefull for automatic name generation
        Returns
        -------
        str
            name of class
        """
        return type(self).__name__

    def get_embedia_type_name(self):
        """
        generates an automatic EmbedIA name for data type assosiated to the
        class that implements layer/element function. This name must exits in
        "embedia.h" and has snake case format
        Returns
        -------
        str
            Embedia type name for layer/element.
        """
        name = self.get_type_name()
        # name in snake case
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        # snake case but skip it in 1D, 2D, etc. to avoid 1_d, 2_d
        name = re.sub(r'(\d)_d', r'\1d', name)

        return name

    def set_default_layer_types(self):
        """
        generates de default EmbedIA data type names for input and output
        parameters of the layer/element object and then sets to the
        corresponding properties
        Returns
        -------
        None.
        """
        self.input_data_type = 'data%dd_t' % len(self.get_input_shape())
        self.output_data_type = 'data%dd_t' % len(self.get_output_shape())

    def get_input_data_type(self):
        """
        returns the C data type used as input by the EmbedIA layer/element.
        Returns
        -------
        str
            type name of the input for layer/element
        """

        return self.input_data_type

    def get_output_data_type(self):
        """
        get the C data type used as output by the EmbedIA layer/element.
        Returns
        -------
        str
            type name of the output for layer/element
        """
        return self.output_data_type

    def get_input_shape(self):
        """
        get the shape of the input of the EmbedIA layer/element.
        Returns
        -------
        n-tuple
            returns the input shape of the layer/element.
        """
        if self.inplace_output:
            prev_layer = self.model.get_previous_layer(self)
            if prev_layer is not None:
                return prev_layer.get_output_shape()

        s = self.layer.input_shape
        if len(s) >= 1 and s[0] is None:
            return s[1:]
        return s

    def get_output_shape(self):
        """
        get the shape of the output of the EmbedIA layer/element.
        Returns
        -------
        n-tuple
            returns the output shape of the layer/element.
        """
        if self.inplace_output:
            prev_layer = self.model.get_previous_layer(self)
            if prev_layer is not None:
                return prev_layer.get_output_shape()

        s = self.layer.output_shape
        if len(s) >= 1 and s[0] is None:
            return s[1:]
        return s

    def get_input_size(self):
        """
        obtains the number of input elements  of the EmbedIA layer/element,
        regardless of the shape.
        Returns
        -------
        int
            number of input elements
        """

        s = 1
        for d in self.get_input_shape():
            s *= d
        return s

    def get_output_size(self):
        """
        obtains the number of output elements  of the EmbedIA layer/element,
        regardless of the shape.
        Returns
        -------
        int
            number of output elements
        """

        s = 1
        for d in self.get_output_shape():
            s *= d
        return s

    def calculate_MAC(self):
        """
        calculates amount of multiplication and accumulation operations
        Returns
        -------
        int
            amount of multiplication and accumulation operations

        """
        return 0

    def calculate_memory(self, types_dict):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """

        return 0

    def get_info(self, types_dict):
        """
        Gets info of the layer/object
        Returns
        -------
        LayerInfo object
            information of layer: name, type name, #params, MACs, output shape,
            activation function name

        """

        return LayerInfo(self, types_dict)

    def predict(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be implemented in
        "embedia.c" and by convention should be called "class name" + "_layer".
        For example, for the EmbedIA Dense class associated to the Keras Dense
        layer, the function "dense_layer" must be implemented in "embedia.c"
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
        return ''

    def activation_function(self, param):
        """
        generates C code with the invocation of the associated activation
        function that must be implemented in the file "embedia.c".
        Parameters
        ----------
        param : str
            name of the variable to be used in the invocation of the C
            function that implements the activation function.
        Returns
        -------
        str
            C code with the invocation of the function that performs the
            processing of the layer in the file "embedia.c".
        """
        if not hasattr(self.layer, 'activation') or self.layer.activation is None:
            return ''

        act_fncs = ActivationFunctions(self.model, self.layer.activation)

        return act_fncs.predict(f'{param}.data', self.get_output_size())


    def debug_function(self, param):
        """
        generates C code with the debug function invocation with the output
        result of the layer to be implemented in the file "embedia_debug.c".
        The default behavior generates an invocation to a function whose name
        is composed of "print_"+"type of data to print" (data1d_t, data2d_t or
        data3d_t).
        Parameters
        ----------
        param : str
            name of the variable to be used in the invocation of the C
            function that implements the debug function.
        Returns
        -------
        str
            C code with the invocation of the function that performs the
            processing of the layer in the file "embedia.c".
        """
        name = self.name
        dbg_fn = 'print_' + self.output_data_type
        return f'''{dbg_fn}("{name}", {param});'''
