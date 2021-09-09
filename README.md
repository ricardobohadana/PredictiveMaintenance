# Repositório Predictive Maintenance
## Aplicação de Algoritmos de Machine Learning para Apoio a Manutenção Preditiva
Esse repositório faz parte do trabalho de conclusão de curso <strong>"Aplicação de Algoritmos de Machine Learning para Apoio a Manutenção Preditiva"</strong> a ser apresentado como pré-requisito para conclusão do curso de graduação em engenharia elétrica na UFF.

## Proposta
Construção de modelo computacional baseado no aprendizado de máquina capaz de apoiar a implementação de uma estratégia de manutenção preditiva

## Dados 
Os dados utilizados estão disponíveis no repositório (por enquanto), apesar de seu tamanho grande. Eles consistem em amostras horárias de variáveis de estado de 100 máquinas de diferentes idades, além de indicação se a mesma opera com erro ou se falhou na última hora/intervalo. 

## Bibliotecas utilizadas:
<ul>
  <li>Numpy</li>
  <li>Pandas</li>
  <li>TensorFlow</li>
  <li>Keras</li>
  <li>Scikit-learn</li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
</ul>

## Arquitetura do modelo desenvolvido:
O modelo desenvolvido é basicamente uma rede neural com duas camadas escondidas (apesar de apenas uma camada escondida ser suficiente para solução de problemas não-lineares.<br>
  Na primeira camada temos as 20 variáveis de entrada.<br>
  Na segunda camada temos 256 neurônios.<br>
  Na terceira camada temos 16 neurônios.<br>
  Na quarte e última camada temos 1 neurônio de saída.<br>
