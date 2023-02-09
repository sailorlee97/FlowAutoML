

class ActivationFunctions:

    def __init__(self, model, activation, **kwargs):
        self.model = model
        self.activation = activation

    def _base_params(self, output_name, output_size):
        return f'{output_name}, {output_size}'

    def _tanh_params(self, output_name, output_size):
        return self._base_params(output_name, output_size)

    def _sigmoid_params(self, output_name, output_size):
        return self._base_params(output_name, output_size)

    def _softmax_params(self, output_name, output_size):
        return self._base_params(output_name, output_size)

    def _softsign_params(self, output_name, output_size):
        return self._base_params(output_name, output_size)

    def _relu_params(self, output_name, output_size):
        return self._base_params(output_name, output_size)

    def _leakyrelu_params(self, output_name, output_size):
        (data_type, macro_converter) = self.model.get_type_converter()
        extra_param = f', {macro_converter(self.activation.alpha)}'
        return self._base_params(output_name, output_size) + extra_param

    def get_params(self, fnc_name, output_name, output_size):
        method_name = f'_{fnc_name}_params'
        method = getattr(self, method_name)
        return method(output_name, output_size)

    def get_function_name(self):
        # activation functions may be objects or functions
        if hasattr(self.activation, '__name__'):
            return self.activation.__name__.lower()
        return self.activation.__class__.__name__.lower()

    def predict(self, output_name, var_output_size):
        fnc_name = self.get_function_name()
        if fnc_name == 'linear':
            return ''
        params = self.get_params(fnc_name, output_name, var_output_size)
        return f'{fnc_name}_activation({params});'
