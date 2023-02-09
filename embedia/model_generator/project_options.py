class ModelDataType:
    (FLOAT, FIXED32, FIXED16, FIXED8, BINARY) = (0, 1, 2, 3, 4)
    LAST = BINARY
    SIZES = [32, 32, 16, 8, 1]

    def get_size(dt):
        if dt <= ModelDataType.LAST:
            return ModelDataType.SIZES[dt]
        return None


class ProjectType:
    (C, CPP, ARDUINO, CODEBLOCK) = (0, 1, 2, 3)


class ProjectFiles:
    (LIBRARY, MAIN, MODEL) = (1, 2, 4)
    (ALL) = {LIBRARY, MAIN, MODEL}


class DebugMode:
    (DISCARD, DISABLED, HEADERS, DATA) = (-1, 0, 1, 2)


class BinaryBlockSize:
    (Bits8, Bits16, Bits32, Bits64) = (0, 1, 2, 3)


class ProjectOptions:
    def __init__(self):
        self.embedia_folder = None
        self.project_type = ProjectType.C
        self.data_type = ModelDataType.FLOAT
        self.baud_rate = 9600
        self.example_data = None
        self.example_labels = None
        self.files = ProjectFiles.ALL
        self.debug_mode = DebugMode.DISABLED
        self.clean_output = False
        self.normalizer = None
        self.tamano_bloque = BinaryBlockSize.Bits8
