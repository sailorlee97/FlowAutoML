from embedia.layers.layer import Layer
from embedia.layers.activation.activation_functions import ActivationFunctions


class Activation(Layer):
    """
    Normally all layers can directly incorporate activation functions. However,
    sometimes this functionality can appear as a independent layer. The EmbedIA
    activation layer is associated with the Keras Activation layer/object
    """

    def __init__(self, model, layer, options=None, **kwargs):
        super().__init__(model, layer, options, **kwargs)

        # saves output into output of previous layer
        self.inplace_output = True

    def predict(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be implemented in
        "embedia.c" and by convention should be called
        "class name" + "_activation" or "function name" + "_activation".
        For example, for the Keras Sigmoid Activation, the function
        "sigmoid_activation" must be implemented in "embedia.c"
        Parameters
        ----------
        input_name : str
            name of the input variable to be used in the invocation of the C
            function that implements the layer. Not used in activation
            functions since the output_name variable is directly modified
        output_name : str
            name of the output variable to be used in the invocation of the C
            function that implements the layer.
        Returns
        -------
        str
            C code with the invocation of the activation function in the file
            "embedia.c" that performs the processing of the layer
        """
        output_size = self.get_output_size()

        if hasattr(self.layer, 'activation'):
            activation = self.layer.activation
        else:
            activation = self.layer
        act_fncs = ActivationFunctions(self.model, activation)

        return act_fncs.predict(f'{output_name}.data', output_size)
