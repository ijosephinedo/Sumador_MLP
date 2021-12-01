import numpy as np
from random import random
import matplotlib.pyplot as plt


class MLP(object):
    """Clase para el Perceptrón Mulicapa
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor del MLP. Toma el número de entradas, capas ocultas y número de salidas

        Args:
            num_inputs (int): Número de entradas
            hidden_layers (list): Lista de capas ocultas
            num_outputs (int): Número de salidas
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # Crea una representación generérica de las capas
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # Creación de pesos random
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # Creación de bias random
        bias = []
        for i in range(len(layers) - 1):
            b = np.random.rand(layers[i + 1])
            bias.append(b)
        self.bias = bias

        # Variable para salvar los deltas
        deltas = []
        for i in range(len(layers)-1):
            d = np.zeros(layers[i+1])
            deltas.append(d)
        self.deltas = deltas

        # Variable para salvar las derivadas
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # Variable para salvar las activaciones o nets
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


    def forward_propagate(self, inputs):
        """Calcula la propagación hacia adelante basado en las entradas

        Args:
            inputs (ndarray): Entradas
        Returns:
            activations (ndarray): Valores de sálida
        """

        # Las activacones de la capa de entrada son las entradas mismas
        activations = inputs

        # Se salvan las entradas en la primer capa
        self.activations[0] = activations

        # Iteración a través de las capas de la red
        for i, w in enumerate(self.weights):
            # Calculo de las Net a través de la multiplicación matrical y la resta del bias
            net_inputs = np.dot(activations, w) - self.bias[i]

            # Aplicación de la función de transferencia
            activations = self._tanh(net_inputs)

            # Se salvan las activaciones para el Backprop
            self.activations[i + 1] = activations

        # Se retornan las activaciones de salida
        return activations


    def back_propagate(self, error):
        """Algoritmo de Backprop
        Args:
            error (ndarray): Error del backpop
        Returns:
            error (ndarray): Error final de salida
        """

        # Iteración hacia atrás de las capas
        for i in reversed(range(len(self.derivatives))):

            # Activaciones de la capa previa
            activations = self.activations[i+1]

            # Aplicación de la función tanh
            delta = error * self._tanh_derivada(activations)

            # Se guarda el valor de los deltas para la actualización de los bias
            self.deltas[i] = delta

            # Reordenamiento del arreglo 2D 
            delta_re = delta.reshape(delta.shape[0], -1).T

            # Activacones de la capa actual
            current_activations = self.activations[i]

            # Reordenamiento de las activaciones como un vector columna
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # Se guardan las derivadas después de la multiplicacioón matricial
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # Backprop del error
            error = np.dot(delta, self.weights[i].T)


    def entrenamiento(self, inputs, targets, epochs, learning_rate):
        """Modelo de entrenamiento forward y backward
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Número de epocas que queremos que entrene
            learning_rate (float): Tasa de aprendizaje
        """
        # Loop para las épocas
        for i in range(epochs):
            sum_errors = 0

            # Iteración de todos los valores de entrada
            for j, input in enumerate(inputs):
                target = targets[j]

                # Activaciones de la red
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # Desarrollo del Gradiente Descendente
                self.gradient_descent(learning_rate)

                # Cálculo del error cuadrático medio 
                sum_errors += self._mse(target, output)

            # Época compleada e informe del error
            print("Error: {}, Época: {}".format(sum_errors / len(items), i+1))

        print("Entrenamiento completado!")
        print("Weights: " + str(self.weights))
        print("Bias:" + str(self.bias))


    def gradient_descent(self, learningRate=1):
        """Aplicación del gradiente descendente
        Args:
            learningRate (float): Que tan rápido aprende
        """
        # Actualización de los pesos
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate
            bias = self.bias[i]
            deltas = self.deltas[i]
            bias -= deltas * learningRate


    def graphs(self):
        """ Función para graficar el espacio de características

        Args:

        Returns: 
            Muestra las gráficas
        """
        x = np.arange(-0.5, 1, 0.01)
        w = self.weights[0]
        b = self.bias[0]
        y1 = (-w[0,0] * x + b[0]) / w[1,0]
        y2 = (-w[0,1] * x + b[1]) / w[1,1]
        y3 = (-w[0,2] * x + b[2]) / w[1,2]
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9, 3))
        ax0.plot(x, y1)
        ax1.plot(x, y2)
        ax2.plot(x, y3)
        plt.show()

    def _tanh(self, x):
        """Función de activación tangente hiperbólica
        Args:
            x (float): Valor a ser procesado
        Returns:
            y (float): Salida
        """
        return np.tanh(x)


    def _tanh_derivada(self, x):
        """Derivada de la funció tangente hiperbólica
        Args:
            x (float): Valor a ser procesado
        Returns:
            y (float): Salida
        """
        return (1.0 - x * x)


    def _mse(self, target, output):
        """Función del error cuadrático medio
        Args:
            target (ndarray): Valor esperado
            output (ndarray): Valores predecidos
        Returns:
            (float): Salida
        """
        return np.average((target - output) ** 2)


if __name__ == "__main__":

    # create a dataset to train a network for the sum operation
    items = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print(items)

    targets = np.array([[0, 0], [0, 1], [0, 1], [1, 0]])
    print(targets)

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [2], 2)

    # train network
    mlp.entrenamiento(items, targets, 2000, 0.5)

    # create dummy data
    input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target = np.array([0, 0])

    # get a prediction
    output = mlp.forward_propagate(input)

    print()
    print("{} + {} = {}, Co = {}".format(input[0,0], input[0,1], output[0,1], output[0,0]))
    print("{} + {} = {}, Co = {}".format(input[1,0], input[1,1], output[1,1], output[1,0]))
    print("{} + {} = {}, Co = {}".format(input[2,0], input[2,1], output[2,1], output[2,0]))
    print("{} + {} = {}, Co = {}".format(input[3,0], input[3,1], output[3,1], output[3,0]))
    
    # mlp.graphs()

