# Investigação de desempenho do DGEMM

Repositório: https://github.com/Ezequielsj/DGEMM

## Integrantes

- Ezequiel de Jesus Santos - 121056350
- Kelly Pinheiro Soares - 125170716
- Luiza Teixeira Barcellos Rosauro de Almeida - 126423803
- Lucas Pereira Pacheco de Medeiros - 126436084

## 1. Introdução

O DGEMM (Double-precision General Matrix Multiplication) calcula o produto de duas matrizes de ponto flutuante em dupla precisão. A operação possui custo computacional cúbico, aproximadamente $2N^3$ operações de ponto flutuante para matrizes quadradas de dimensão $N$.

Este trabalho investiga como decisões de implementação e recursos da arquitetura do computador alteram o desempenho dessa operação. O projeto anterior será usado somente como referência para as técnicas, enquanto os resultados deste relatório serão obtidos em uma nova rodada de experimentos. Este documento consolida as duas etapas parciais e a entrega final.

## 2. Objetivos

- Implementar e comparar versões progressivamente otimizadas do DGEMM.
- Avaliar o efeito do tamanho da matriz sobre o desempenho.
- Medir o impacto de localidade de memória, vetorização, blocking e paralelismo.
- Validar a corretude numérica antes de comparar os tempos.
- Relacionar os resultados observados com cache, SIMD e número de threads.

## 3. Ambiente experimental

- Processador: Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz
- Número de núcleos e threads: 4 núcleos e 8 threads
- Memória RAM: aproximadamente 8 GB
- Sistema operacional: Windows
- Compilador e versão: GCC 8.1.0 (MinGW-w64)
- Flags de compilação: `-O3 -Wall`; versões SIMD usam `-mavx2 -mfma`
- GPU: NVIDIA GeForce MX110 e Intel UHD Graphics
- Versão do driver da GPU: 451.67
- Observação: o comando `make` não está disponível no ambiente atual.

Cada implementação deve ser executada com os mesmos dados de entrada, período de aquecimento e número de repetições. Para cada configuração, registrar o tempo, a mediana, a dispersão e o desempenho em GFLOPS.

A métrica principal será:

$$
\mathrm{GFLOPS} = \frac{2N^3}{t \times 10^9}
$$

onde $N$ é a dimensão da matriz e $t$ é o tempo necessário para uma multiplicação.

Tamanhos avaliados nesta versão: $N \in \{32, 64, 128\}$. As comparações entre técnicas SIMD, blocking e OpenMP foram feitas com $N=32$, respeitando os múltiplos exigidos por cada implementação.

Para cada resultado, foi usada tolerância `1e-12` no teste independente [validate_dgemm.c](validate_dgemm.c). Em `N=32`, as versões AVX2, FMA/unrolling, blocking e OpenMP obtiveram erro máximo `0,000e+000`.

## 5. Implementações avaliadas

### 5.1 Baseline

A Parte 1 apresenta o baseline em C escalar, com três laços aninhados e sem SIMD ou OpenMP. Os resultados estão em [Parte1_DGEMM.md](Parte1_DGEMM.md) e [resultados_parte1.csv](resultados_parte1.csv).

### 5.2 Reorganização dos laços

As matrizes C usadas nos capítulos em C seguem o layout column-major. A ordem dos laços foi mantida compatível com os índices desse armazenamento para permitir a progressão até SIMD e blocking.

### 5.3 Vetorização SIMD

O Chapter3 processa quatro valores `double` por registrador AVX2. O Chapter4 acrescenta FMA e quatro acumuladores SIMD. A análise e os dados estão em [Parte2_DGEMM.md](Parte2_DGEMM.md) e [resultados_parte2.csv](resultados_parte2.csv).

### 5.4 Blocking ou tiling

O Chapter5 usa blocos de `32 x 32` para aumentar a reutilização dos dados na cache. A hipótese e os resultados estão em [Parte3_DGEMM.md](Parte3_DGEMM.md).

### 5.5 Paralelismo

O Chapter6 usa OpenMP sobre os blocos e foi testado com 1, 2 e 4 threads. Em `N=32`, a sobrecarga do paralelismo dominou e o aumento de threads reduziu o throughput observado.

### 5.6 Bibliotecas de referência

MKL e PyTorch/CUDA não foram incluídos porque as dependências correspondentes não estão disponíveis no ambiente atual.

## 6. Resultados

As tabelas das três partes estão nos respectivos relatórios e os dados brutos estão em `resultados_parte1.csv`, `resultados_parte2.csv` e `resultados_parte3.csv`. O notebook [DGEMM_analise.ipynb](DGEMM_analise.ipynb) reúne a leitura dos dados e os gráficos preliminares. O script [analisar_resultados.py](analisar_resultados.py) reproduz as estatísticas pela linha de comando e separa os resultados por tamanho de matriz e número de threads.

### 6.1 Efeito do tamanho da matriz

No baseline, foram observados `2,45`, `2,63` e `2,53 GFLOPS` nas três repetições de `N=32`; as execuções preliminares de `N=64` e `N=128` produziram `2,28` e `1,93 GFLOPS`.

### 6.2 Efeito do tamanho do bloco

Nesta versão foi avaliado o bloco de `32 x 32`. Uma varredura de tamanhos alternativos permanece como melhoria possível.

### 6.3 Efeito do número de threads

Com `N=32`, o blocking obteve `25,48 GFLOPS`, enquanto OpenMP obteve `10,06`, `2,10` e `1,14 GFLOPS` com 1, 2 e 4 threads, respectivamente.

### 6.4 Comparação entre implementações

No experimento SIMD, o baseline obteve `2,53 GFLOPS`, AVX2 obteve `12,05 GFLOPS` e AVX2 + FMA + unrolling obteve `28,03 GFLOPS` em média.

## 7. Discussão

Interpretar os resultados observando:

- quando o problema deixa de ser limitado principalmente pela computação;
- quando o acesso à memória e a cache passam a dominar;
- quais otimizações apresentam ganho consistente;
- quais configurações não melhoram o desempenho e por quê;
- como a variabilidade das medições afeta as conclusões.

## 8. Conclusão

Os resultados preliminares mostram ganhos consistentes com AVX2, FMA, unrolling e blocking. O OpenMP não apresentou ganho em `N=32`, indicando que a sobrecarga de threads precisa ser amortizada por problemas maiores. A corretude numérica foi validada em `N=32` com erro máximo zero. As principais limitações restantes são a campanha reduzida de tamanhos e a indisponibilidade de MKL e PyTorch/CUDA.

## 9. Requisitos e limitações do ambiente

Os experimentos foram realizados em Windows, com Intel Core i5-10210U, 4 nucleos, 8 threads, aproximadamente 8 GB de RAM, GCC 8.1.0 e Python 3.7.8. O ambiente suportou AVX2, FMA e OpenMP. Como `make` nao estava instalado, os programas C foram compilados diretamente com GCC. MKL, PyTorch e CUDA nao estavam disponiveis e, portanto, nao foram incluidos na comparacao principal.

## 10. Referências

- PATTERSON, David A.; HENNESSY, John L. *Computer Organization and Design: The Hardware/Software Interface, RISC-V Edition*. Morgan Kaufmann.
- Documentação das bibliotecas utilizadas e outras fontes consultadas.
