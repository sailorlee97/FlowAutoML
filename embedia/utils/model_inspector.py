# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 00:45:28 2022

@author: cesar
"""

from tensorflow.keras import backend as K
import numpy as np


class ModelInspector(object):
    """
    This class obtains the output of each layer of a model from a given input
    data. It allows retrieving the layer information with its output,
    converting it to a string or saving it to disk.
    """

    _model = None

    def __init__(self, model=None):
        """
        Initialize the model inspector

        Parameters
        ----------
        model : TYPE, optional
            Tensorflow/Keras model. The default is None.

        Returns
        -------
        None.

        """
        self.setModel(model)

    def setModel(self, model):
        """
        Set model property to inspect

        Parameters
        ----------
        model : Tensorflow/Keras model

        Returns
        -------
        None.

        """
        self._model = model

    def inspect(self, inp_data):
        """
        Obtains information from each layer along with its output from a
        specified data input.

        Parameters
        ----------
        inp_data : array
            input data to feed the model and obtain the outputs.

        Returns
        -------
        layers_outputs : list of tuple (layer, data_output)
            for each layer of the model, a tuple is generated
            (layer, data_output) and added in a list.

        """

        inp_data = self._prepare_inp_data(inp_data)

        layers_outputs = []
        for layerIndex, layer in enumerate(self._model.layers):
            func = K.function([self._model.get_layer(index=0).input], layer.output)
            layers_outputs.append((layer, func([inp_data])))

        return layers_outputs

    def as_string(self, inp_data, col_sz=10, margin='  ', ln_break=80):
        """
        Constructs a string with the output data of each layer of the model
        from a specified input data.

        Parameters
        ----------
        inp_data : array with input data
            Array with input data.
        col_sz : int, optional
            Size in characters of the column into which each output value must
            fit. The default is 10.
        margin : String, optional
            String to indent each line of the output data. The default is '  '.
        ln_break : int, optional
            Maximum length of each line of text. Lines are broken if adding a
            new value exceeds this limit. If a value of 0 is specified, the
            lines are not broken. The default is 80.

        Returns
        -------
        String
            String with the output data of each layer of the model. For each
            layer is provided:
            - a header with the name and type of the layer and its output form.
            - the sequence of output values.

        """

        inp_data = self._prepare_inp_data(inp_data)

        layers_outputs = self.inspect(inp_data)

        return self._layers_to_string(layers_outputs, col_sz, margin, ln_break)

    def print(self, inp_data, col_sz=10, margin='  ', ln_break=80):
        """
        prints the output data of each layer of the model from a specified
        input data.

        Parameters
        ----------
        inp_data : array with input data
            Array with input data.
        col_sz : int, optional
            Size in characters of the column into which each output value must
            fit. The default is 10.
        margin : String, optional
            String to indent each line of the output data. The default is '  '.
        ln_break : int, optional
            Maximum length of each line of text. Lines are broken if adding a
            new value exceeds this limit. If a value of 0 is specified, the
            lines are not broken. The default is 80.

        Returns
        -------
        None.

        """
        print(self.as_string(inp_data, col_sz, margin, ln_break))

    def save(self, filename ,inp_data, col_sz=10, margin='  ', ln_break=80):
        """
        saves the output data of each layer of the model from a specified input
        data.

        Parameters
        ----------
        filename : String
            filename to save the ouput data of each layer of the model.
        inp_data : array with input data
            Array with input data.
        col_sz : int, optional
            Size in characters of the column into which each output value must
            fit. The default is 10.
        margin : String, optional
            String to indent each line of the output data. The default is '  '.
        ln_break : int, optional
            Maximum length of each line of text. Lines are broken if adding a
            new value exceeds this limit. If a value of 0 is specified, the
            lines are not broken. The default is 80.

        Returns
        -------
        None.

        """
        f = open(filename, 'w')
        f.write(self.as_string(
            inp_data,
            col_sz=col_sz,
            margin=margin,
            ln_break=ln_break))
        f.close()

    def _prepare_inp_data(self, inp_data):
        """
        Internal method. Ensure to set input shape to match input layer shape.

        Parameters
        ----------
        inp_data : array
            A rray of values. Input data for input layer.

        Returns
        -------
        inp_data : array
            array of values with same input shape of input layer.

        """
        if len(self._model.input_shape) > len(inp_data.shape):
            inp_data = np.array([inp_data])
        return inp_data

    def _data_to_str(self, data, col_sz=10, margin=' ', ln_break=80):
        """
        Internal method. Generates formatted string from data array

        Parameters
        ----------
        data : array
            Array of values. Input data for input layer.
        col_sz : int, optional
            Size in characters of the column into which each output value must
            fit. The default is 10.
        margin : String, optional
            String to indent each line of the output data. The default is '  '.
        ln_break : int, optional
            Maximum length of each line of text. Lines are broken if adding a
            new value exceeds this limit. If a value of 0 is specified, the
            lines are not broken. The default is 80.

        Returns
        -------
        String
            Returns a string formatted with the values provided by the
            data parameter

        """
        if data.shape[0] == 1:
            data = data[0]
        if len(data.shape) == 3:
            data = self.reconvert3d(data)
            return self._data3d_to_str(data, col_sz, margin, ln_break)
        elif len(data.shape) == 2:
            return self._data2d_to_str(data, col_sz, margin, ln_break)
        else:
            return self._data1d_to_str(data, col_sz, margin, ln_break)

    def _data3d_to_str(self, data, col_sz, margin, ln_break):
        """
        Internal method. Generates formatted string from 3D data array

        Parameters
        ----------
        data : array
            3D array of values. Input data for input layer.
        col_sz : int, optional
            Size in characters of the column into which each output value must
            fit. The default is 10.
        margin : String, optional
            String to indent each line of the output data. The default is '  '.
        ln_break : int, optional
            Maximum length of each line of text. Lines are broken if adding a
            new value exceeds this limit. If a value of 0 is specified, the
            lines are not broken. The default is 80.

        Returns
        -------
        String
            Returns a string formatted with the values provided by the
            data parameter

        """
        output = ''
        for k in range(data.shape[0]):
            (a, b, c, d) = (k+1, data.shape[0], data.shape[1], data.shape[2])
            output += '\nDIM %d/%d [%d x %d]:' % (a, b, c, d)
            output += self._data2d_to_str(data[k], col_sz, margin, ln_break)
        return output

    def _data2d_to_str(self, data, col_sz, margin, ln_break):
        """
        Internal method. Generates formatted string from 2D data array

        Parameters
        ----------
        data : array
            2D array of values. Input data for input layer.
        col_sz : int, optional
            Size in characters of the column into which each output value must
            fit. The default is 10.
        margin : String, optional
            String to indent each line of the output data. The default is '  '.
        ln_break : int, optional
            Maximum length of each line of text. Lines are broken if adding a
            new value exceeds this limit. If a value of 0 is specified, the
            lines are not broken. The default is 80.

        Returns
        -------
        String
            Returns a string formatted with the values provided by the
            data parameter

        """
        output = '\n'
        for j in range(data.shape[0]):
            output += self._data1d_to_str(data[j], col_sz, margin, ln_break)

        return output

    def _data1d_to_str(self, data, col_sz, margin, ln_break):
        """
        Internal method. Generates formatted string from 1D data array

        Parameters
        ----------
        data : array
            1D array of values. Input data for input layer.
        col_sz : int, optional
            Size in characters of the column into which each output value must
            fit. The default is 10.
        margin : String, optional
            String to indent each line of the output data. The default is '  '.
        ln_break : int, optional
            Maximum length of each line of text. Lines are broken if adding a
            new value exceeds this limit. If a value of 0 is specified, the
            lines are not broken. The default is 80.

        Returns
        -------
        String
            Returns a string formatted with the values provided by the
            data parameter

        """
        line = margin
        output = '\n'
        for i in range(data.shape[0]):
            value = self._format_value(data[i], col_sz)
            if ln_break <= 0 or len(line) + len(value) < ln_break:
                line += value
            else:
                output += line+'\n'
                line = margin + value
        output += line+'\n'

        return output

    def _format_value(self, value, col_sz, sep=' '):
        """
        format value as string

        Parameters
        ----------
        value : float
            Float value to format.
        col_sz : int
            Length in characters to limit the value to be converted to
            (including separator string) .
        sep : String, optional
            Value separation string. The default is ' '.

        Returns
        -------
        String
            Formated string of "value" .

        """
        # dig_prec = col_sz - sign -dot -digit -sep
        fmt = '{:>%d.%df}' % (col_sz - len(sep), col_sz - 3 - len(sep))

        return fmt.format(value) + sep

    def _layers_to_string(self, layers_outputs, col_sz, margin, ln_break):
        """
        Constructs a formatted string with the layer's output data of the model

        Parameters
        ----------
        layers_outputs : array
            array of layer's outputs.
        col_sz : int, optional
            Size in characters of the column into which each output value must
            fit. The default is 10.
        margin : String, optional
            String to indent each line of the output data. The default is '  '.
        ln_break : int, optional
            Maximum length of each line of text. Lines are broken if adding a
            new value exceeds this limit. If a value of 0 is specified, the
            lines are not broken. The default is 80.

        Returns
        -------
        output : String
            Returns a string formatted from the layer's outputs of the model.

        """
        ln_len = 80 if ln_break <= 0 else ln_break
        output = ''

        for layer, layer_output in layers_outputs:
            dims = [str(d) for d in layer.output_shape if d is not None]
            if len(dims) == 1:
                dims.insert(0, '1')
            layer_type = type(layer).__name__

            output += ('=' * ln_len)+'\n'
            output += '%s (%s) [%s]:\n' % (layer.name, layer_type, ' x '.join(dims))
            output += self._data_to_str(layer_output, col_sz, margin, ln_break)

        return output

    def reconvert3d(self, data):
        n_data = np.empty((data.shape[2], data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    n_data[k, i, j] = data[i, j, k]

        return n_data
    