def save_to_file(filename, content):
    file = open(filename, 'w', encoding='utf-8')
    file.write(content)
    file.close()
    
def read_from_file(filename):
    file = open(filename, 'r', encoding='utf-8')
    # content = file.readlines()
    content = file.read()
    file.close()  
    return content