import os
import pandas as pd

def list_files_in_directory(directory):
    files = []
    contador = 0
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            os.rename(os.path.join(directory, filename), os.path.join(directory, f'{contador}.png'))
            files.append(f'{contador}.png')
            contador += 1
    return files

def save_files_to_excel(files, output_filename):
    df = pd.DataFrame(files, columns=['Filename'])
    df.to_excel(output_filename, index=False)

# Especifica la ruta del directorio
directory = 'C:\\Users\\leuis\\Desktop\\Projectes_Personals\\pokemon_NN\\test'

# Especifica el nombre del archivo Excel de salida
output_filename = 'files_list.xlsx'

# Listar los archivos en el directorio
files = list_files_in_directory(directory)

# Guardar los nombres de los archivos en un archivo Excel
save_files_to_excel(files, output_filename)

print(f'Lista de archivos guardada en {output_filename}')