# Investigação de desempenho do DGEMM

Repositório: https://github.com/Ezequielsj/DGEMM

## Integrantes

- Ezequiel de Jesus Santos - 121056350
- Kelly Pinheiro Soares - 125170716
- Luiza Teixeira Barcellos Rosauro de Almeida - 126423803
- Lucas Pereira Pacheco de Medeiros - 126436084

## Resumo executivo

Este projeto investiga o desempenho da operação DGEMM em uma plataforma real, com foco em como diferentes técnicas de otimização afetam a eficiência computacional. A análise parte de uma implementação escalar e avança para versões que empregam vetorização SIMD, desenrolamento de laços, blocking de cache e paralelismo via OpenMP.

A principal contribuição do trabalho é mostrar que o ganho de desempenho não decorre apenas da complexidade do algoritmo, mas da forma como o código explora a hierarquia de memória, os registradores e as instruções do processador. Os resultados indicam que otimizações bem escolhidas podem aumentar o throughput em ordens de grandeza, embora o benefício dependa do tamanho do problema e do custo de sincronização e overhead das threads.

Este documento representa a versão final do relatório, consolidando os objetivos, a metodologia, os resultados e as conclusões em uma apresentação coerente para entrega acadêmica.

## 1. Introdução

O DGEMM (Double-precision General Matrix Multiplication) calcula o produto de duas matrizes de ponto flutuante em dupla precisão. A operação possui custo computacional cúbico, aproximadamente $2N^3$ operações de ponto flutuante para matrizes quadradas de dimensão $N$.

Este trabalho investiga como decisões de implementação e recursos da arquitetura do computador alteram o desempenho dessa operação. O enfoque principal não é apenas produzir uma multiplicação de matrizes funcional, mas entender como a reorganização do código, o uso de instruções SIMD e o paralelismo afetam a vazão de processamento e o tempo de execução em uma plataforma real.

A hipótese central deste estudo é que a otimização do código, quando alinhada com as características da microarquitetura, produz ganhos de desempenho significativos em relação à implementação escalar. No entanto, esses ganhos não são universais: dependem do tamanho do problema, da organização dos acessos à memória, da localidade dos dados e da disponibilidade de paralelismo em hardware.

## 2. Objetivos

- Implementar e comparar versões progressivamente otimizadas do DGEMM.
- Avaliar o efeito do tamanho da matriz sobre o desempenho.
- Medir o impacto de localidade de memória, vetorização, blocking e paralelismo.
- Validar a corretude numérica antes de comparar os tempos.
- Relacionar os resultados observados com cache, SIMD e número de threads.
- Analisar quando uma otimização aumenta o desempenho e quando a sobrecarga deixa de compensar o ganho.

## 3. Metodologia experimental

A metodologia adotada neste projeto busca comparar todas as versões em condições equivalentes de execução. Para cada implementação, o programa foi executado com a mesma base de entrada, mesmo tamanho de matriz, mesma janela de tempo e mesma métrica de análise. As versões foram compiladas com flags de otimização apropriadas e os resultados foram registrados em CSV para posterior processamento.

A métrica principal foi o desempenho em GFLOPS, calculado por:

$$
\mathrm{GFLOPS} = \frac{2N^3 \times m}{t \times 10^9}
$$

onde $N$ é a dimensão da matriz, $m$ é o número de multiplicações realizadas e $t$ é o tempo total de execução em segundos.

Essa abordagem permite avaliar não apenas o tempo absoluto de execução, mas também a eficiência da implementação em termos de operações por segundo. Em conjunto com a validação numérica, a metodologia garante que o ganho de desempenho observado seja acompanhado pela preservação da correção da operação.

## 4. Ambiente experimental

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

## 5. Metodologia de reprodução

As versões C foram compiladas diretamente no GCC porque o comando `make` não estava disponível:

```text
gcc -O3 -Wall -o Chapter2/parte1_baseline.exe Chapter2/main_algorithm.c
gcc -O3 -Wall -mavx2 -mfma -o Chapter3/parte2_simd.exe Chapter3/main_algorithm.c
gcc -O3 -Wall -mavx2 -mfma -o Chapter4/parte2_unroll.exe Chapter4/main_algorithm.c
gcc -O3 -Wall -mavx2 -mfma -o Chapter5/parte3_blocking.exe Chapter5/main_algorithm.c
gcc -O3 -Wall -mavx2 -mfma -fopenmp -o Chapter6/parte3_openmp.exe Chapter6/main_algorithm.c
```

As medições principais usaram `N=32`, janela de 0,5 segundos e três repetições para as versões da Parte 2. A Parte 1 também possui referências preliminares com `N=64` e `N=128`. Na Parte 3, OpenMP foi comparado com 1, 2 e 4 threads.

A métrica utilizada foi:

$$
\mathrm{GFLOPS} = \frac{2N^3 \times m}{t \times 10^9}
$$

onde `m` é o número de multiplicações realizadas e `t` é o tempo acumulado em segundos. Os dados brutos estão nos arquivos CSV próprios do projeto.

## 6. Implementações avaliadas

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

## 7. Resultados

As tabelas das três partes estão nos respectivos relatórios e os dados brutos estão em `resultados_parte1.csv`, `resultados_parte2.csv` e `resultados_parte3.csv`. O notebook [DGEMM_analise.ipynb](DGEMM_analise.ipynb) reúne a leitura dos dados e os gráficos preliminares. O script [analisar_resultados.py](analisar_resultados.py) reproduz as estatísticas pela linha de comando e separa os resultados por tamanho de matriz e número de threads.

### 6.1 Efeito do tamanho da matriz

No baseline, foram observados `2,45`, `2,63` e `2,53 GFLOPS` nas três repetições de `N=32`; as execuções preliminares de `N=64` e `N=128` produziram `2,28` e `1,93 GFLOPS`.

### 6.2 Efeito do tamanho do bloco

Nesta versão foi avaliado o bloco de `32 x 32`. Uma varredura de tamanhos alternativos permanece como melhoria possível.

### 6.3 Efeito do número de threads

Com `N=32`, o blocking obteve `25,48 GFLOPS`, enquanto OpenMP obteve `10,06`, `2,10` e `1,14 GFLOPS` com 1, 2 e 4 threads, respectivamente.

### 6.4 Comparação entre implementações

No experimento SIMD, o baseline obteve `2,53 GFLOPS`, AVX2 obteve `12,05 GFLOPS` e AVX2 + FMA + unrolling obteve `28,03 GFLOPS` em média.

### 6.5 Validação numérica

As versões otimizadas foram comparadas com uma referência escalar independente no arquivo [validate_dgemm.c](validate_dgemm.c). Para `N=32`, AVX2, FMA/unrolling, blocking e OpenMP apresentaram erro máximo `0,000e+000`, com status `PASS` e tolerância `1e-12`.

## 8. Discussão

Os resultados indicam que AVX2 aumentou significativamente o desempenho em relação ao baseline, enquanto FMA e unrolling trouxeram ganho adicional. Esse comportamento é esperado, pois a operação de matrizes exige alta intensidade de cálculo e a vetorização permite processar múltiplos valores `double` em paralelo, reduzindo o número de instruções necessárias para completar o mesmo trabalho.

O blocking também apresentou bom resultado, confirmando a importância da localidade de referência e da reutilização de dados na cache. Em vez de acessar a memória de forma indiscriminada, a implementação em blocos reduz a distância entre os dados necessários em um trecho da computação e melhora a eficiência do uso da hierarquia de memória.

Já o OpenMP não compensou para `N=32`, pois a matriz pequena não forneceu trabalho suficiente para amortizar o custo de criação, distribuição e sincronização das threads. Esse resultado reforça a ideia de que paralelismo não é uma otimização automática: ele só traz benefício quando a carga computacional é suficientemente grande para justificar a sobrecarga de gerenciamento das threads.

As conclusões são específicas da máquina testada. O tamanho `N=32` é pequeno para avaliar escalabilidade OpenMP, e MKL, PyTorch e CUDA não foram executados por falta das dependências correspondentes. Portanto, o estudo deve ser interpretado como uma investigação de comportamento sob um ambiente concreto, e não como regra universal para todas as arquiteturas.

Para uma campanha futura, seria interessante testar matrizes maiores e mais tamanhos de bloco, além de medir a sensibilidade da implementação em relação ao número de threads e ao nível de cache disponível.

A leitura dos resultados deve considerar também os seguintes pontos:

- quando o problema deixa de ser limitado principalmente pela computação;
- quando o acesso à memória e a cache passam a dominar;
- quais otimizações apresentam ganho consistente;
- quais configurações não melhoram o desempenho e por quê;
- como a variabilidade das medições afeta as conclusões.

## 9. Conclusão

Os resultados preliminares mostram ganhos consistentes com AVX2, FMA, unrolling e blocking. Esses ganhos confirmam a hipótese central do projeto: quando a implementação se alinha ao modelo de execução da microarquitetura, o desempenho do DGEMM pode aumentar de maneira muito expressiva.

O OpenMP não apresentou ganho em `N=32`, indicando que a sobrecarga de threads precisa ser amortizada por problemas maiores. Esse comportamento é coerente com a teoria de paralelismo: para matrizes pequenas, o custo de sincronização e distribuição do trabalho pode superar o ganho de throughput.

A corretude numérica foi validada em `N=32` com erro máximo zero, o que garante que os ganhos observados não foram obtidos à custa de resultados incorretos. As principais limitações restantes são a campanha reduzida de tamanhos e a indisponibilidade de MKL e PyTorch/CUDA. Mesmo assim, os experimentos deixam evidente que as otimizações estudadas têm impacto real sobre a eficiência do DGEMM.

## 10. Requisitos e limitações do ambiente

Os experimentos foram realizados em Windows, com Intel Core i5-10210U, 4 nucleos, 8 threads, aproximadamente 8 GB de RAM, GCC 8.1.0 e Python 3.7.8. O ambiente suportou AVX2, FMA e OpenMP. Como `make` nao estava instalado, os programas C foram compilados diretamente com GCC. MKL, PyTorch e CUDA nao estavam disponiveis e, portanto, nao foram incluidos na comparacao principal.

## 11. Referências

- PATTERSON, David A.; HENNESSY, John L. *Computer Organization and Design: The Hardware/Software Interface, RISC-V Edition*. Morgan Kaufmann.
- Documentação das bibliotecas utilizadas e outras fontes consultadas.
