from embedia.layers.layer import Layer

class DummyLayer(Layer):
    """
    This class defines the behavior of an EmdedIA layer/element that does not
    play any purpose in the inference process
    """

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
        super().__init__(model, layer, options, **kwargs)


    def predict(self, input_name, output_name):
        # Nothing to do
        return ''

    def activation_function(self, param):
        # Nothing to do
        return ''

    def debug_function(self, param):
        # Nothing to do
        return ''
