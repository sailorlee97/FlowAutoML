from embedia.layers.layer import Layer


class DataLayer(Layer):
    """
    This class defines the behavior of an embedia layer/element that requires
    the definition of complementary data structures for its operation. It
    automatically creates the declaration of a variable, the declaration of an
    initialization function prototype and the initialization call. The layer
    name will be used to define the name of the structure, the variable, and
    the initialization function.
    Subclasses can define in their constructor the property "struct_data_type"
    with the type of the structure required by the function that implements the
    layer/element in C (by default it uses as convention the lowercase class
    name with the suffix "_layer_t". Both the structure type and the layer
    function must be previously implemented in the "embedia.c" and "embedia.h"
    files, respectively.
    Example: given a Dense layer with the name "dense0", it will take as
     "dense_layer_t" as value of the property "struct_data_type" and will
     generate automatically in the file "your_model_name_model.c":
        - the declaration of the variable "dense_layer_t dense0_data"
        - the prototype of the function "dense_layer_t dense0_data_init()"
        - the initialization "dense0_data = dense0_data_init()"
    It remains for the user to implement the "functions_init" and "predict"
    methods. In the first one must generate the code of the function
    "dense0_data_init" to complete the data of the structure. In the second
    one, must generate the code to invoke the function in the "embedia.c" file,
    which is responsible for implementing the operation of the layer/element.
    """

    def __init__(self, model, layer, options=None, **kwargs):
        super().__init__(model, layer, options, **kwargs)

        # this type automatic defined in "struct_data_type" must exists in
        # "embedia.h"
        self.struct_data_type = self.get_struct_data_type()

    def get_struct_data_type(self):
        """
        gets automatic embedia name for structure associated with layer/element
        Returns
        -------
        str
            embedia type name for layer/element.
        """
        return self.get_embedia_type_name() + '_layer_t'

    def var(self):
        """
        Generates C code with the declaration of a variable of the type
        indicated by the property "struct_data_type" and with the name
        "layer name "+"_data" (e.g.: "dense_data_t dense0_data;").
        Parameters
        ----------
            None
        Returns
        -------
        str
            C code declaring a variable that stores the additional data
            required (e.g. neuron weights) to perform the function of the
            layer/element
        """

        return f'{self.struct_data_type} {self.name}_data;\n'

    def prototypes_init(self):
        """
        Generates C code with the declaration of the prototype of the function
        that must initialize the data structure required by the layer
        (e.g. neuron weights). The return value must match the type
        indicated by the property "struct_data_type" and function name must
        have the name "init_"+"layer name "+"_data" and defined in "embedia.c"
        file (e.g.: "dense_data_t init_dense0_data();").
        Returns
        -------
        str
            C code with the function prototype for data initialization
        """

        return f'{self.struct_data_type} init_{self.name}_data(void);\n'

    def init(self):
        """
        Generates C code with the implementation of the function that must
        initialize the data structure required by the layer (e.g. neuron
        weights). The return value must match the type indicated by the
        property "struct_data_type". The function must be named
        "init_"+"layer name "+"_data" and must be implemented in "embedia.c"
        file (e.g.: "dense_data_t init_dense0_data() {....}").
        Note that data types such as arrays must be declared as "static" to
        persist in memory after the function has been invoked.
        Returns
        -------
        str
            DESCRIPTION.
        """
        return f'    {self.name}_data = init_{self.name}_data();\n'

    def functions_init(self):
        """
        this python method must implement the C function defined as prototype
        in "prototypes_init" in order to return the structure defined in
        "struct_data_type" property filled with the layer data returns a string
        with the C implementation of the funcion declared in "prototypes_init"m
        ethod. This function must return a filled structure of type defined in
        "struct_data_type" property with de layer data.
        It should be noted that array type data must be declared as static.
        For mor details se implementation of Dense layer.
        Parameters
        ----------
            None
        Returns
        -------
        str
            C code with the implementation of the function that fills the
            structure required by the layer/element embedia.
        """
        return ''

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
    