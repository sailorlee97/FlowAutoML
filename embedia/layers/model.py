from embedia.layers.layers_implemented import dict_layers
from collections import defaultdict
from embedia.model_generator.project_options import ModelDataType
import regex as re
import pycparser as pcp
from embedia.model_generator.project_options import BinaryBlockSize


class UnsupportedLayerError(Exception):
    types_dict = {}

    def __init__(self, obj):
        super().__init__(f"EmbedIA layer/element not implemented for {str(type(obj))}")
        self.object = obj


class Model(object):
    types_dict = {}

    def __init__(self, options):
        self.options = options
        self.clear_names()

    def set_layers(self, layers, options_array=None):
        # options es la generica del proyecto
        # options_array es un vector con opciones para cada clase

        embedia_layers = []

        try:
            # external normalizar to the model? => add as first layer
            if self.options.normalizer is not None:
                obj = self.options.normalizer
                ly = dict_layers[type(obj)](self, obj, self.options)
                embedia_layers.append(ly)

            for layer in layers:
                obj = layer
                ly = dict_layers[type(layer)](self, layer, self.options)
                embedia_layers.append(ly)
        except KeyError:
            raise UnsupportedLayerError(obj)

        self.embedia_layers = embedia_layers
        return embedia_layers

    def get_previous_layer(self, layer):
        try:
            idx = self.embedia_layers.index(layer)
        except ValueError:
            return None
        if idx == 0:
            return None
        return self.embedia_layers[idx-1]


    def clear_names(self):
        self.names = defaultdict(lambda: 0)

    def get_unique_name(self, obj):
        if hasattr(obj, "name"):
            name = obj.name
        elif hasattr(obj, "__name__"):
            name = obj.__name__
        else:
            name = obj.__class__.__name__

        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        num = self.names[name]
        self.names[name] += 1
        if num == 0:
            return name
        return name+str(num)

    def get_type_converter(self):
        """
        returns a tuple with the name of the type used (float or float) in the
        data representation (e.g. neuron weights) together with the conversion
        macro to be invoked to transform a float value to the data type

        Parameters
        ----------
        data_type : ModelDataType
            variable with the data type used in the data representation
            (float, fixed8, fixed16, fixed32)

        Returns
        -------
        tuple (str, str)
            tuple with type and macro convertion for C.

        """

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
        if self.options.data_type == ModelDataType.FLOAT or self.options.data_type == ModelDataType.BINARY:
            def macro_converter(v):
                return v
            data_type = 'float'       #binary layers dont use data_type, use block_type
        else:
            def macro_converter(v):
                return f'''FL2FX({v})'''
            data_type = 'fixed'

        return (data_type, macro_converter)

    def _build_types_size_dict(self, embedia_decl):
        # prepare to extract declaration of structures

        # get code to first function definition in order to includes structures
        if (self.options.data_type == ModelDataType.BINARY):
            start = embedia_decl.find('endif')
            start = start + 5
        else:
            start = embedia_decl.find('typedef')
        end = embedia_decl.find('void')
        embedia_decl = embedia_decl[start:end]

         # remove comments, pycparser doesnt support them
        pattern = re.compile(
                r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                re.DOTALL | re.MULTILINE
            )
        embedia_decl = re.sub(pattern, '', embedia_decl)

        # add dummy base data type declaration into EmbedIA code
        embedia_decl = """
typedef char uint8_t;
typedef char uint16_t;
typedef char uint32_t;
typedef char uint64_t;
typedef char xBITS;
typedef char fixed;
typedef char dfixed;

""" + embedia_decl

        parser = pcp.CParser()

        code = parser.parse(embedia_decl)

        if(self.options.tamano_bloque==BinaryBlockSize.Bits8):
            bytes_size = 1
        elif(self.options.tamano_bloque==BinaryBlockSize.Bits16):
            bytes_size = 2
        elif(self.options.tamano_bloque==BinaryBlockSize.Bits32):
            bytes_size = 4
        else:
            bytes_size = 8
        if(self.options.data_type == ModelDataType.FIXED8):
            bytes_size2 = 1
            bytes_size3 = 2
        elif(self.options.data_type == ModelDataType.FIXED16):
            bytes_size2 = 2
            bytes_size3 = 4
        else:
            bytes_size2 = 4
            bytes_size3 = 8
        # base types sizes in bytes
        self.types_dict = {
            'uint8_t': 1,
            'uint16_t': 2,
            'uint32_t': 4,
            'uint64_t': 8,
            'float': 4,
            'xBITS': bytes_size,
            'fixed': bytes_size2,
            'dfixed': bytes_size3
        }

        for node in code.ext:
            if type(node) is pcp.c_ast.Typedef and not node.name in  self.types_dict:
                dt_type = node.name
                dt_size = self._explore_type(node)
                self.types_dict[dt_type] = dt_size

        return self.types_dict

    def _explore_type(self, node):
        if type(node.type.type) is pcp.c_ast.Struct:
            size = 0
            for d in node.type.type.decls:
                size += self._explore_type(d)
            return size
        elif type(node.type) is pcp.c_ast.PtrDecl:
            return 4  # always return 4 bytes for now
        elif node.type.type.names[0] in self.types_dict:
            return self.types_dict[node.type.type.names[0]]

        raise Exception(node.type)

    def get_layers_info(self, embedia_decl):
        if len(self.types_dict) == 0:
            self._build_types_size_dict(embedia_decl)

        layers_info = []
        for layer in self.embedia_layers:
            info = layer.get_info(self.types_dict)
            l_type = info.class_name
            l_name = info.layer_name
            l_act = info.activation
            params = info.params
            shape = info.output_shape
            MACs = info.macs_ops
            size = info.memory

            layers_info.append((l_name, l_type, l_act, params, shape, MACs, size))

        return layers_info

    def firstLayerOfItsclass(self, embedia_layer):
        for layer in self.embedia_layers:
            if type(embedia_layer) is type(layer):
                return embedia_layer == layer
        return False