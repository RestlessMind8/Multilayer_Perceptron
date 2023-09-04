import numpy as np
import random as rd
import matplotlib.pyplot as plt
import dataset_config as dt

samplesByAttribute = 50 # amostras por atributo
v_out = 10 # quantidade de saídas da rede
samples = samplesByAttribute * v_out # quantidade total de amostras
inputs = 11 # as entradas correspondem aos atributos do vinho
neurons = 55  # quantidade de neurônios da rede
limiar = 0.0
alfa = 0.005 # taxa de aprendizagem
tolerated_error = 0.5
cycles = []
errorByCycle = []

# obtendo as matrizes de dados
x = dt.getDataset()
targets = dt.getTargets()
order = dt.getOrderArray()

# gerando os pesos sinapticos das ligações entre a camada de entrada e a camada intermediária 
weights = np.zeros((inputs, neurons)) #vanterior
for i, j in np.ndindex(weights.shape):
    weights[i][j] = rd.uniform(-2.0, 2.0)

# gerando os bias das ligações entre a camada de entrada e da camada intermediária 
bias = np.zeros((1, neurons)) #v0anterior
for i in range(neurons):
    bias[0][i] = rd.uniform(-2.0, 2.0)

# gerando os pesos sinapticos das ligações entre a camada intermediária e a camada de saída 
w_weights = np.zeros((neurons, v_out)) #wanterior
for i, j in np.ndindex(w_weights.shape):
    w_weights[i][j] = rd.uniform(-2.0, 2.0)

# gerando os bias das ligações entre a camada intermediária e a camada de saída  
w_bias = np.zeros((1, v_out)) #w0anterior
for i in range(v_out):
    w_bias[0][i] = rd.uniform(-2.0, 2.0)

# matrizes de atualização de pesos e valores de saída da rede
weights_new = np.zeros((inputs, neurons)) #vnovo
bias_new = np.zeros((1, neurons)) #v0novo
w_weights_new = np.zeros((neurons, v_out)) #wnovo
w_bias_new = np.zeros((1, v_out)) #w0novo

# inicializando as matrizes de treinamento
zin = np.zeros((1, neurons))
z = np.zeros((1, neurons))
deltaK = np.zeros((v_out, 1))
deltaW0 = np.zeros((v_out, 1))
delta = np.zeros((1 ,neurons))
xaux = np.zeros((1 ,inputs))
h = np.zeros((v_out, 1))
target = np.zeros((v_out, 1))
delta2 = np.zeros((neurons, 1))
cycle = 0 
total_error = 10000

# treinamento
while tolerated_error < total_error:
    total_error = 0
    for default in range(samples):
        for j in range(neurons):
            zin[0][j] = np.dot(x[default, :], weights[:, j]) + bias[0][j]
        z = np.tanh(zin)
        yin = np.dot(z, w_weights) + w_bias
        y = np.tanh(yin)
        for m in range(v_out):
            h[m][0] = y[0][m]
            target[m][0] = targets[m][order[default]]
        total_error += np.sum(0.5 * ((target - h) ** 2))

        # obtendo matrizes para atualização dos pesos
        deltaK = (target - h) * (1 + h) * (1 - h)
        deltaW = alfa * (np.dot(deltaK, z))
        deltaW0 = alfa * delta
        deltaIn = np.dot(np.transpose(deltaK), np.transpose(w_weights))
        delta = deltaIn * (1 + z) * (1 - z)
        for m in range(neurons):
            delta2[m][0] = delta[0][m]
        for k in range(inputs):
            xaux[0][k] = x[default][k]
        deltaV = alfa * np.dot(delta2, xaux)
        deltaV0 = alfa * delta
        
        # atualizações dos pesos
        weights_new = weights + np.transpose(deltaV)
        bias_new = bias + np.transpose(deltaV0)
        w_weights_new = w_weights + np.transpose(deltaW)
        w_bias_new = w_bias + np.transpose(deltaW0)
        
        weights = weights_new
        bias = bias_new
        w_weights = w_weights_new
        w_bias = w_bias_new
    
    cycle += 1
    cycles.append(cycle)
    errorByCycle.append(total_error)
    print(f'Ciclo: {cycle}, {total_error}')
    
# Gráfico de ciclos x erro
plt.plot(cycles, errorByCycle)
plt.xlabel('Cycle')
plt.ylabel('Error')
plt.show()


# teste manual
















