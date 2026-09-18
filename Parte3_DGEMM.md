# Entrega Final - Blocking, OpenMP e consolidacao

Repositorio: https://github.com/Ezequielsj/DGEMM

## Integrantes

- Ezequiel de Jesus Santos - 121056350
- Kelly Pinheiro Soares - 125170716
- Luiza Teixeira Barcellos Rosauro de Almeida - 126423803
- Lucas Pereira Pacheco de Medeiros - 126436084

## 1. Objetivo

Esta entrega final acrescenta localidade de cache por blocking e paralelismo entre blocos com OpenMP. Tambem consolida a progressao observada nas duas entregas parciais e registra a avaliacao do efeito do numero de threads.

## 2. Blocking ou tiling

O Chapter5 divide as matrizes em blocos de `32 x 32`. Os lacos externos percorrem os blocos das dimensoes `i`, `j` e `k`, mantendo uma porcao menor das matrizes em uso antes de avancar. A versao preserva AVX2, FMA e quatro acumuladores do Chapter4.

A hipotese experimental e que blocos menores reduzem a quantidade de dados reutilizados fora da cache, melhorando a localidade temporal e espacial. Nesta implementacao, `N` precisa ser multiplo de 32.

## 3. OpenMP

O Chapter6 aplica `#pragma omp parallel for` ao percurso dos blocos da dimensao `j`. Como cada thread trabalha com colunas diferentes de `C`, os blocos distribuidos nao devem escrever na mesma regiao durante uma multiplicacao.

Foram testadas 1, 2 e 4 threads com a variavel `OMP_NUM_THREADS`. A medicao do Chapter6 usa tempo de parede por meio de `omp_get_wtime()`.

## 4. Compilacao e execucao

```text
gcc -O3 -Wall -mavx2 -mfma -o Chapter5/parte3_blocking.exe Chapter5/main_algorithm.c
gcc -O3 -Wall -mavx2 -mfma -fopenmp -o Chapter6/parte3_openmp.exe Chapter6/main_algorithm.c
```

Todos os testes desta parte usaram `N=32` e janela de 0,5 segundos. O valor de `N=32` foi escolhido para validar rapidamente o fluxo; por ser pequeno, ele tambem evidencia o custo de criar e sincronizar equipes OpenMP.

## 5. Resultados

| Versao | Threads | N | Multiplicacoes | Tempo total (s) | GFLOPS | Speedup vs. blocking |
|---|---:|---:|---:|---:|---:|---:|
| Blocking | 1 | 32 | 194382 | 0,50 | 25,48 | 1,00x |
| OpenMP | 1 | 32 | 76749 | 0,50 | 10,06 | 0,39x |
| OpenMP | 2 | 32 | 16057 | 0,50 | 2,10 | 0,08x |
| OpenMP | 4 | 32 | 8666 | 0,50 | 1,14 | 0,04x |

Os valores foram calculados por `2 * N^3 * multiplicacoes / (tempo * 10^9)`. Cada configuracao terminou com sucesso. O compilador foi executado novamente depois da correcao das mensagens de tamanho, sem erros de compilacao.

## 6.1 Validacao numerica

O programa [validate_dgemm.c](validate_dgemm.c) calcula uma referencia escalar independente com as mesmas matrizes de entrada e compara as versoes otimizadas. Para `N=32`, AVX2, AVX2 + FMA + unrolling, blocking e OpenMP obtiveram erro maximo `0,000e+000`, todos com status `PASS` e tolerancia `1e-12`.

## 6. Discussao

O blocking apresentou o maior throughput neste teste. A versao OpenMP nao superou o blocking para `N=32`; com mais threads, o resultado caiu. Isso nao contradiz a expectativa de paralelismo: neste tamanho, o trabalho total e pequeno e o custo de criar, distribuir e sincronizar as threads pode ser maior que o ganho obtido com a divisao do trabalho.

Portanto, nao e correto concluir que OpenMP sempre piora o DGEMM. A conclusao limitada e que, para a configuracao testada, o tamanho do problema nao foi suficiente para amortizar o custo do paralelismo. A campanha final deve testar `N` maiores, como `64`, `128` e `256`, alem de repetir cada configuracao mais vezes.

## 7. Limitacoes e pendencias

- Ainda nao foi feita uma varredura de varios tamanhos de bloco.
- A comparacao de threads foi feita em `N=32`, que e pequeno para avaliar escalabilidade.
- A validacao numerica foi feita para `N=32`; tamanhos maiores ainda podem ser adicionados.
- MKL e PyTorch/CUDA nao foram incluidos porque suas dependencias nao estao disponiveis neste ambiente.
- Os resultados desta parte sao uma primeira campanha e devem ser interpretados junto com as repeticoes e tamanhos maiores da versao final.

## 8. Conclusao

O experimento confirmou a etapa de blocking e mostrou que o paralelismo precisa ser avaliado considerando o tamanho do problema. Para matrizes pequenas, a sobrecarga do OpenMP pode dominar; para matrizes maiores, a distribuicao dos blocos pode se tornar vantajosa. Essa diferenca entre as configuracoes e parte do resultado da investigacao, e nao deve ser substituida por uma conclusao baseada apenas no melhor numero.

## Referencia

PATTERSON, David A.; HENNESSY, John L. *Computer Organization and Design: The Hardware/Software Interface, RISC-V Edition*. Morgan Kaufmann.
