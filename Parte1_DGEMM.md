# Parte 1 - Investigacao inicial do DGEMM

Repositorio: https://github.com/Ezequielsj/DGEMM

## Integrantes

- Ezequiel de Jesus Santos - 121056350
- Kelly Pinheiro Soares - 125170716
- Luiza Teixeira Barcellos Rosauro de Almeida - 126423803
- Lucas Pereira Pacheco de Medeiros - 126436084

## 1. Objetivo

Nesta primeira parte, estabelecemos uma referencia para as etapas seguintes: o produto de matrizes quadradas em precisao dupla, inicialmente descrito por uma implementacao direta e depois executado em C sem instrucoes vetoriais ou paralelismo explicito.

O objetivo nao e buscar o melhor desempenho ainda. A funcao desta etapa e fixar o problema, a forma de medicao e uma primeira referencia para comparar as otimizacoes posteriores.

## 2. Problema DGEMM

Para matrizes quadradas `A`, `B` e `C` de dimensao `N`, o calculo realizado e:

`C[i][j] = C[i][j] + A[i][k] * B[k][j]`

para todos os indices `i`, `j` e `k`. Uma multiplicacao completa realiza aproximadamente `2 * N^3` operacoes de ponto flutuante.

Nesta implementacao, as matrizes sao armazenadas em ordem column-major, coerente com os indices usados no programa C. O algoritmo possui tres lacos aninhados e nao utiliza AVX2, FMA, blocking ou OpenMP.

## 3. Ambiente

- Processador: Intel Core i5-10210U, 4 nucleos e 8 threads.
- Memoria: aproximadamente 8 GB de RAM.
- Sistema operacional: Windows.
- Compilador: GCC 8.1.0 (MinGW-w64).
- Flags: `-O3 -Wall`.
- Ferramenta de build: compilacao direta com GCC, pois `make` nao esta disponivel no ambiente.

## 4. Medicao inicial

O executavel foi compilado e executado com:

```text
gcc -O3 -Wall -o Chapter2/parte1_baseline.exe Chapter2/main_algorithm.c
Chapter2/parte1_baseline.exe 32 0.5
```

Foram feitas tres repeticoes para `N=32`, usando a mesma janela de 0,5 segundos. Os tamanhos `N=64` e `N=128` permanecem como referencias preliminares de uma execucao curta.

| Implementacao | Repeticao | N | Janela solicitada (s) | Multiplicacoes | Tempo acumulado (s) | GFLOPS |
|---|---:|---:|---:|---:|---:|---:|
| C escalar | 1 | 32 | 0,50 | 18673 | 0,50 | 2,45 |
| C escalar | 2 | 32 | 0,50 | 20470 | 0,51 | 2,63 |
| C escalar | 3 | 32 | 0,50 | 19268 | 0,50 | 2,53 |
| C escalar | preliminar | 64 | 0,20 | 914 | 0,21 | 2,28 |
| C escalar | preliminar | 128 | 0,20 | 92 | 0,20 | 1,93 |

O GFLOPS foi calculado por `2 * N^3 * multiplicacoes / (tempo * 10^9)`. Como o programa contabiliza a duracao de cada multiplicacao e encerra somente depois de ultrapassar a janela solicitada, o tempo acumulado pode ser ligeiramente maior que o alvo.

Para as tres repeticoes de `N=32`, a media foi `2,53 GFLOPS`, a mediana foi `2,53 GFLOPS` e o desvio padrao amostral foi `0,09 GFLOPS`.

## 5. Interpretacao preliminar

Os resultados de `N=32` agora possuem tres repeticoes e sao uma referencia mais estavel para a Parte 2. Os valores de `N=64` e `N=128` continuam preliminares. A implementacao foi executada com sucesso, mas a validacao numerica automatica do conteudo de `C` ainda sera acrescentada antes da conclusao final.

A principal conclusao desta etapa e que a implementacao escalar compila e executa corretamente no ambiente atual, oferecendo uma base concreta para comparar vetorizacao, desenrolamento de lacos, blocking e paralelismo. A referencia usada na validacao numerica foi calculada por uma versao escalar independente.

## 6. Proxima etapa

A Parte 2 acrescentara instrucoes SIMD com AVX2, FMA e desenrolamento de lacos. A comparacao devera manter o mesmo tamanho de matriz e criterio de medicao sempre que a implementacao permitir.

## Requisitos e limitacoes do ambiente

Os testes foram realizados em Windows, com Intel Core i5-10210U, 4 nucleos, 8 threads, aproximadamente 8 GB de RAM, GCC 8.1.0 e Python 3.7.8. O compilador suportou AVX2, FMA e OpenMP. O comando `make` nao estava instalado, por isso a compilacao foi feita diretamente com GCC. MKL, PyTorch e CUDA nao estavam disponiveis no ambiente.

## Referencia

PATTERSON, David A.; HENNESSY, John L. *Computer Organization and Design: The Hardware/Software Interface, RISC-V Edition*. Morgan Kaufmann.
