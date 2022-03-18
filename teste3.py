import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

print('Carregando Arquivo de teste')
arquivo = np.load('teste3.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])





regr = MLPRegressor(hidden_layer_sizes=(1100),  # qtd neuronios
                    max_iter=5000, # quantas vezes os neuronios vão se conectar
                    activation='tanh', #{'identity', 'logistic', 'tanh', 'relu'}, # tipo do grafico
                    solver='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change=200) # qtd de iteracoes
print('Treinando RNA')
regr = regr.fit(x,y)



print('Preditor')
y_est = regr.predict(x)






plt.figure(figsize=[14,7])

#plot curso original
plt.subplot(1,3,1)
plt.plot(x,y)

#plot aprendizagem
plt.subplot(1,3,2)
plt.plot(regr.loss_curve_)

#plot regressor
plt.subplot(1,3,3)
plt.plot(x,y,linewidth=1,color='yellow')
plt.plot(x,y_est,linewidth=2)




plt.show()
