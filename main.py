import os
from PIL import ImageGrab, Image
from pynput import keyboard
import uuid
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import io, color

"""

DEFINIR ARQUITECTURA DE LA RED NEURONAL

"""

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(768, 128)  # Capa completamente conectada (256 entradas, 128 salidas)
        self.fc2 = nn.Linear(128, 64)   # Capa completamente conectada (128 entradas, 64 salidas)
        self.fc3 = nn.Linear(64, 18)    # Capa completamente conectada (64 entradas, 18 salidas para 18 neuronas)

        self._initialize_weights() # Inicializar los pesos de las capas completamente conectadas

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Aplicar la función de activación ReLU a la primera capa oculta
        x = F.relu(self.fc2(x))  # Aplicar la función de activación ReLU a la segunda capa oculta
        x = self.fc3(x)          # La salida es el resultado de la tercera capa completamente conectada
        return x

    # Inicializar los pesos de las capas completamente conectadas
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

model = SimpleNN()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()  # Entropía cruzada como función de pérdida para problemas de clasificación
optimizer = optim.Adam(model.parameters(), lr=0.001)  # ADAM como optimizador con tasa de aprendizaje de 0.001

"""

DATOS DE ENTRENAMIENTO

"""
labels = []
inputs = []

"""

ENTRENAMIENTO DE LA RED NEURONAL

"""

# Bucle de entrenamiento
def entrenamiento(model, criterion, optimizer, inputs, labels, epochs=100):
    for epoch in range(epochs):
        model.train()  # Modo de entrenamiento
        optimizer.zero_grad()  # Reiniciar los gradientes acumulados en el optimizador
        outputs = model(inputs)  # Propagar hacia adelante (forward pass)
        loss = criterion(outputs, labels)  # Calcular la pérdida
        loss.backward()  # Propagar hacia atrás (backward pass) para calcular los gradientes
        optimizer.step()  # Actualizar los parámetros del modelo

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')  # Imprimir la pérdida en cada época
"""

EVALUAR RED NEURONAL
"""
def evaluar(model, inputs, labels):
    model.eval()  # Modo de evaluación
    with torch.no_grad():  # Deshabilitar el cálculo de gradientes
        outputs = model(inputs)
        print("Forma de las predicciones (outputs):", outputs.shape)
        print("Forma de las etiquetas:", labels.shape)
        
        _, predicted = torch.max(outputs, 1)
        print("Forma de las predicciones (predicted):", predicted.shape)


        total = labels.size(0)

        correct = (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100}%')

def create_histogram(image):
    # Convertir la imagen a escala de grises
    gray_image = color.rgb2gray(np.array(image))
    # Calcular HOG
    hog_descriptor = hog(gray_image, orientations=4, pixels_per_cell=(16, 16),
                                    cells_per_block=(4, 4), visualize=False)
    return hog_descriptor

# (left, top, right, bottom)
bboxSimple = [800, 232, 903, 345]
bboxDouble2 = [754, 376, 846, 484]

keyboard_thread = None

def on_press(key):
    global bbox

    id = uuid.uuid4()

    try:
        if key.char == 'q':
            screenshot = ImageGrab.grab(bboxSimple)
            screenshot.save(f'.\\test\\screenshot_{id}.png')
            print('Screenshot saved')

        elif key.char == 'w':
            id2 = uuid.uuid4()

            screenshot = ImageGrab.grab(bboxSimple)
            screenshot1 = ImageGrab.grab(bboxDouble2)
            screenshot.save(f'.\\test\\screenshot_{id}.png')
            screenshot1.save(f'.\\test\\screenshot_{id2}.png')

            print('Screenshot saved')

    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        return False

def keyboard_listener_thread():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def main():
    keyboard_thread = threading.Thread(target=keyboard_listener_thread)
    keyboard_thread.start()

    keyboard_thread.join()

def load_csv():
    global labels, inputs

    df = pd.read_excel("files_list.xlsx")

    label_columns = df.columns[1:19]
    labels = df[label_columns].fillna(0).values.tolist()

    inputs = df['Filename'].values.tolist()
    histograms = []

    for i in range(len(inputs)):
        path = os.path.join('.\\test', inputs[i])
        image = Image.open(path)

        histograms.append(create_histogram(image))

    histograms = np.array(histograms)

    inputs = torch.tensor(histograms, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.1, random_state=36)

    # Convertir labels a long para CrossEntropyLoss
    labels_train = torch.argmax(labels_train, dim=1).long()
    labels_test = torch.argmax(labels_test, dim=1).long()
    
    print("Inicio de entrenamiento")
    entrenamiento(model, criterion, optimizer, inputs_train, labels_train)

    print("Inicio de evaluación")
    evaluar(model, inputs_test, labels_test)

if __name__ == '__main__':

    load_csv()

