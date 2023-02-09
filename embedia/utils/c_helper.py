from embedia.utils.binary_helper import BinaryGlobalMask

def declare_array(dt_type, var_name, dt_conv, data_array, limit=80):

    if data_array is None:
        val = 'NULL'
    else:
        val = ''
        line = ''
        for v in data_array:
            line += f'''{dt_conv(v)}, '''
            if len(line) >= limit:
                val += line + '\n    '
                line = ''
        if line != '':
            val += line
        val = val[:-2]  # remove last comma and space

    if var_name is None or var_name == '':
        code = ''  # only values
    else:
        code = f'{dt_type} {var_name}[] ='
    code += f'''{{
    {val}
    }}'''

    return code


def declare_array2(toti,xBits,lista_contadores,dt_type, var_name, dt_conv, data_array, limit=80):

    if data_array is None:
        val = 'NULL'
    else:
        val = ''
        line = ''
        for v in data_array:
            
            lista_contadores[2] = lista_contadores[2] + 1
            
            if xBits==16:
                if v == 1.0:  
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_16())[lista_contadores[1]]
            elif xBits==32:
                if v == 1.0: 
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_32())[lista_contadores[1]]
            elif xBits==64:
                if v == 1.0: 
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_64())[lista_contadores[1]]
            else:
                if v == 1.0: 
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_8())[lista_contadores[1]]
            
            if lista_contadores[1] == xBits-1 or (lista_contadores[2] == toti):
                
                line += f'''{dt_conv(lista_contadores[0])}, '''
                if len(line) >= limit:
                    val += line + '\n    '
                    line = ''
                lista_contadores[1] = 0
                lista_contadores[0] = 0
                
            else:
                lista_contadores[1] = lista_contadores[1] +1
            
            
        if line != '':
            val += line
        val = val[:-2]  # remove last comma and space

    if var_name is None or var_name == '':
        code = ''  # only values
    else:
        code = f'{dt_type} {var_name}[] ='
    code += f'''{{
    {val}
    }}'''

    return code

