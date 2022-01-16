# Repositório Predictive Maintenance
## Desenvolvimento de Solução Baseada em Machine Learning para Apoio a Manutenção Preditiva
Esse repositório faz parte do trabalho de conclusão de curso <strong>"Desenvolvimento de Solução Baseada em Machine Learning para Apoio a Manutenção Preditiva"</strong> a ser apresentado como pré-requisito para conclusão do curso de graduação em engenharia elétrica na UFF.

## Proposta
Construção de modelo computacional baseado no aprendizado de máquina capaz de apoiar a implementação de uma estratégia de manutenção preditiva, com o objetivo de alertar sobre iminência de falha ao engenheiro de manutenção responsável pela supervisão dos equipamentos.

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
A arquitetura foi escolhida com base em análises que estão melhor descritas e explicadas no Trabalho de Conclusão de Curso.
O modelo desenvolvido é basicamente uma rede neural com duas camadas escondidas (apesar de apenas uma camada escondida ser suficiente para solução de problemas não-lineares.<br>
  Na primeira camada temos as 16 variáveis de entrada.<br>
  Na segunda camada (única camada escondida) temos 16 neurônios.<br>
  Na quarte e última camada temos 1 neurônio de saída.<br>

## Arquivo principal:
#### Último pipeline para replicar o desenvolvimento foi publicado às 12:59 do dia 02 de dezembro de 2021 sob o nome de pipeline_2.py.
