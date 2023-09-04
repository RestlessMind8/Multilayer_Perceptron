import numpy as np
import csv

samplesByAttribute = 50 # amostras por atributo
v_out = 10 # quantidade de saídas da rede
samples = samplesByAttribute * v_out # quantidade total de amostras
inputs = 11 # as entradas correspondem aos atributos do vinho
order = np.zeros(samples)

# carregando o arquivo targets
with open('targets.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)    
    targets = list(csv_reader)
    csv_file.close    

# montando o arquivo de amostras de treinamento

with open('winequality-red.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)
    data = list(csv_reader)
       
    # convertendo os dados de string para lista
    dataList = []
    for row in data:        
        dataList.append(row[0].split(';'))       

    
    # Encontrando os mínimos e máximos de cada coluna
    minn = [float('inf')] * (len(dataList[0]) - 1)
    maxx = [float('-inf')] * (len(dataList[0]) - 1)
        
    for linha in dataList:         
        for i, valor_str in enumerate(linha):     
            if i < 11:          
                valor_float = float(valor_str)  
                if valor_float < minn[i]:
                    minn[i] = valor_float
                if valor_float > maxx[i]:
                    maxx[i] = valor_float    

    # normalizando os dados
    normalized_dataset = []
    cont = 0
    for linha in dataList:        
        for i, valor_str in enumerate(linha):
            valor_float = float(valor_str) 
            if i < 11:                
                linha[i] = (valor_float - minn[i]) / (maxx[i] - minn[i])
            elif cont < samples:
                order[cont]
                cont += 1
        normalized_dataset.append(linha)

    # montando a matriz x
    x = np.zeros((samples, inputs))  
    for i in range(samples):
        for j in range(inputs):            
            x[i][j] = normalized_dataset[i][j]

    csv_file.close


def getDataset():
    return x

def getTargets():
    return targets

def getOrderArray():
    return order.astype('int')
    

    
                    