


1.  Inicializar os pesos com valores arbitrários não nulos.
2.  Apresentar um padrão de entrada x(n) e propagá-lo até a saída da rede.     
3.  Calcular os erros instantâneos na saída da rede, ek(n).
4.  Calcular os gradientes locais dos neurônios da camada de saída, δsk(n).
5.  Ajustar os pesos da camada de saída pela expressão:

                        wskj (n + 1) = wskj (n) + η δsk(n) .ij (n) 

6. Calcular os gradientes locais dos neurônios da camada oculta, δoj (n).
7. Ajustar os pesos da camada oculta pela expressão:

                         woji (n + 1) = woji (n) + η δoj (n) .xi (n)

8. Repetir os passos de 2 a 7 para todos os padrões de treinamento (1 época)
9. Calcular o erro médio quadrado (EMQ) para o arquivo de treinamento.
10. Se o EMQ for maior que o valor desejado, repetir o passo 8.


# a funcao local gradiente esta ok, verificar quem chama esta funcao

function lg = localgradienteO(d, output)
	(d - output)*output*(1-output)
end
