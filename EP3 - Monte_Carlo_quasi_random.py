#Bibliotecas
import random, chaospy
import numpy as np
import pandas as pd
from scipy.stats import beta


#Variáveis
RG = 390000000
NUSP = 10300000
a = RG / 10 ** len(str(RG)) # 0.RG
b = NUSP / 10 ** len(str(NUSP)) # 0.NUSP 

#Função
def f(x):
    return np.exp(-a*x) * np.cos(b*x)

#Função aproximada de f(x)
def g_function(x):
    return np.exp(-a*x)

#Gerador de númeoros quasi-aleatórios
def Quasi_Random_Generator(n):

    distribution = chaospy.J(chaospy.Uniform(0,1))
    x = distribution.sample(n,rule='Halton')
    
    return x

#Crud
def Crud(n=150):
    
    #O erro começa valendo 10 para que o loop rode antes de calcular o erro real
    erro = 10
    
    while erro >= 0.01:
        
        contador_Crud = []
        
        for i in range(1,n+1):
            
            #Gerador de números quasi-aleatórios
            x = Quasi_Random_Generator(n)[random.randint(0,n-1)]
            contador_Crud.append(f(x))
                
            #Calcula o erro padrão
            erro = np.std(contador_Crud)/np.sqrt(i)
            
            #Verifica se o erro é menor que 1% depois de 2 iterações
            if i > 2 :
                if erro < 0.01:
                    break

            resultado = np.mean(contador_Crud)
    
    return resultado


def Hit_or_miss(n=200):
    
    #O erro começa valendo 10 para que o loop rode antes de calcular o erro real
    erro = 10
    
    while erro >= 0.01:
        
        contador_HM = []
        
        for i in range(1,n+1):
            
            # h(x, y) = I(f(x) ≥ y) 
            h = 0
            for k in range(i):
                h += f(Quasi_Random_Generator(n)[random.randint(0,n-1)]) > Quasi_Random_Generator(n)[random.randint(0,n-1)]
            
            contador_HM.append(h/i)
              
            #Calcula o erro padrão
            erro = np.std(contador_HM)/np.sqrt(i)
            
            #Verifica se o erro é menor que 1% depois de 10 iterações
            if i > 10:
                if erro < 0.01:
                    break
                    
            resultado = np.mean(contador_HM)

            
    return resultado

def Importance_Sampling(n=100):
    
    erro = 10
    
    while erro >= 0.01:
        
        contador_IS = []
        
        for i in range(1,n+1):
            
            #Gerador de quasi-aleatório com distribuição beta
            quasi = chaospy.Beta(0.9,1,0,1).sample(100,rule='Halton')
            p = np.array(random.choices(quasi,k=100)) #Escolhe um valor da lista quasi
            x = beta.pdf(p,0.9,1)
            
            contador_IS.append(np.mean(f(p)/x))
            
            #Calcula o erro padrão
            erro = np.std(contador_IS)/np.sqrt(i)
            
            #Verifica se o erro é menor que 1% a partir da 2 iteração
            if i > 10:
                if erro < 0.01:
                    break
                    
            resultado = np.mean(contador_IS)
    
            
    return resultado


def Control_Variate(n=150):
    
    erro = 10
    
    while erro >= 0.01:
        
        contador_f = []
        contador_g = []
        
        for i in range(1,n+1):
            
            #Calcula a aproximação pelo método Crud para as funções f(x) e g(x)
            x  = Quasi_Random_Generator(n)[random.randint(0,n-1)]
            contador_f.append(f(x))
            contador_g.append(g_function(x))
            
            #Calcula o erro padrão
            erro = np.std(contador_f)/np.sqrt(i)
            
            #Verifica se o erro é menor que 1% depois de 10 iterações
            if i > 1:
                if erro < 0.01:
                    break
                    
            saida = np.mean(contador_f)
            
    media_controle = np.mean(contador_g)
    #A função cov retorna a matriz de covariancia contendo 
    #[[correlação da primeira matriz,correlação entre as duas matrizes],[correlação das duas matrizes, correlação da segunda matriz]]
    c = -np.cov(contador_f,contador_g)/np.var(contador_g)

    segunda_parte = c[0][1] * (media_controle - 0.824)
    
    resultado = saida + segunda_parte
    
    return resultado


def main():
    
    #Roda cada método uma vez
    print('\n Rodando cada método uma vez \n')
    print('===================================================')
    print('Crud: ',Crud())
    print('Hit or miss: ', Hit_or_miss())
    print('Importance Sampling:', Importance_Sampling())
    print('Control Variate: ',Control_Variate())
    print('===================================================')
    
    
    CRUD = []
    HM = []
    IS = []
    CV = []
    
    #Roda cada método 5 vezes e salva o resultado em uma lista
    for i in range(5):
        CRUD.append(Crud())
        HM.append(Hit_or_miss())
        IS.append(Importance_Sampling())
        CV.append(Control_Variate())
    
    #Lista de métodos
    list_metodos = {'Método Crud':CRUD,
                    'Método Hit or miss': HM,
                    'Método Importance Sampling': IS,
                    'Método Control Variate':CV}
    
    print('\n\n Rodando cada métodos 5 vezes \n')
    print('===================================================')
    
    
    #Estatísticas dos métodos
    for nome,testes in list_metodos.items():
        print(nome,'\n')
        print('Resultado médio',np.mean(testes))
        print('Desvio padrão',np.std(testes))
        print('Máximo',np.max(testes))
        print('Mínimo',np.min(testes))
        print('Mediana',np.median(testes))
        print('===================================================')
    

main()

    
        
