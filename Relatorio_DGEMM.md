# Investigação de desempenho do DGEMM

## Links do projeto no GitHub

- Repositório principal: [DGEMM](https://github.com/Ezequielsj/DGEMM)
- Relatório final: [Relatorio_DGEMM.md](https://github.com/Ezequielsj/DGEMM/blob/master/Relatorio_DGEMM.md)
- Resultados da Parte 1: [resultados_parte1.csv](https://github.com/Ezequielsj/DGEMM/blob/master/resultados_parte1.csv)
- Resultados da Parte 2: [resultados_parte2.csv](https://github.com/Ezequielsj/DGEMM/blob/master/resultados_parte2.csv)
- Resultados da Parte 3: [resultados_parte3.csv](https://github.com/Ezequielsj/DGEMM/blob/master/resultados_parte3.csv)
- Dados brutos da campanha ampliada: [resultados_campanha.csv](https://github.com/Ezequielsj/DGEMM/blob/master/resultados_campanha.csv)
- Script de análise: [analisar_resultados.py](https://github.com/Ezequielsj/DGEMM/blob/master/analisar_resultados.py)
- Benchmark reprodutível: [benchmark_dgemm.c](https://github.com/Ezequielsj/DGEMM/blob/master/benchmark_dgemm.c)
- Kernels compartilhados: [dgemm_kernels.h](https://github.com/Ezequielsj/DGEMM/blob/master/dgemm_kernels.h)
- Validação numérica: [validate_dgemm.c](https://github.com/Ezequielsj/DGEMM/blob/master/validate_dgemm.c)

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

O projeto contém uma campanha preliminar, coletada no Windows, e uma campanha ampliada pareada, executada no WSL 2. Na campanha ampliada, as variantes de cada dimensão usam as mesmas matrizes determinísticas, inicializam `C` com zero, recebem um aquecimento e cinco repetições medidas. Todas usam o mesmo relógio monotônico de parede, compilador e métrica; cada linha bruta e sua validação ficam registradas em CSV.

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
- Campanha preliminar: Windows e GCC 8.1.0 (MinGW-w64)
- Campanha ampliada: Ubuntu 24.04 no WSL 2, GCC 13.3.0, 4 núcleos/8 threads expostos e flags `-O3 -mavx2 -mfma -fopenmp`
- GPU: NVIDIA GeForce MX110 e Intel UHD Graphics
- Versão do driver da GPU: 451.67
- A CPU exposta ao WSL reporta suporte a AVX2 e FMA.

Na campanha ampliada foram usados `N=128` e `N=256`; os blocos avaliados foram `8`, `16`, `32` e `64`, todos divisores de ambos os tamanhos. OpenMP foi medido com 1, 2 e 4 threads. O benchmark registra cinco observações por configuração, além de um aquecimento não registrado.

A métrica principal será:

$$
\mathrm{GFLOPS} = \frac{2N^3}{t \times 10^9}
$$

onde $N$ é a dimensão da matriz e $t$ é o tempo necessário para uma multiplicação.

Os dados preliminares cobrem `N=32`, `64` e `128`; a campanha pareada ampliada compara as variantes em `N=128` e `N=256`. As variantes sem blocking separam AVX2, FMA e unrolling; blocking e OpenMP foram comparados para quatro tamanhos de bloco.

O validador [validate_dgemm.c](validate_dgemm.c) executa os mesmos kernels compartilhados chamados pelos executáveis dos Chapters 2 a 6, contra uma referência escalar independente. Foram validados `N=32`, `128` e `256`, todos os blocos válidos e 1, 2 e 4 threads; todos passaram com erro máximo zero e tolerância absoluta `1e-10`.

## 5. Metodologia de reprodução

Os executáveis, o benchmark e o validador podem ser compilados com os alvos do Makefile:

```text
make ch2 ch3 ch4 ch5 ch6 benchmark validate
```

O benchmark ampliado é reproduzido por:

```text
./benchmark_dgemm.exe resultados_campanha.csv 5 1
./validate_dgemm.exe
```

Os argumentos do benchmark são caminho do CSV, número de repetições e número de aquecimentos. A campanha usa seed `20261003 + N`; ela registra separadamente AVX2, AVX2 + FMA, AVX2 + FMA + unrolling x4, blocking com cada bloco e OpenMP com cada combinação de bloco e threads.

A métrica utilizada foi:

$$
\mathrm{GFLOPS} = \frac{2N^3 \times m}{t \times 10^9}
$$

onde `m` é o número de multiplicações realizadas e `t` é o tempo acumulado em segundos. Os dados brutos estão nos arquivos CSV próprios do projeto.

## 6. Implementações avaliadas

### 6.1 Baseline

A Parte 1 apresenta o baseline em C escalar, com três laços aninhados e sem SIMD ou OpenMP. Os resultados estão em [Parte1_DGEMM.md](Parte1_DGEMM.md) e [resultados_parte1.csv](resultados_parte1.csv).

### 6.2 Reorganização dos laços

As matrizes C usadas nos capítulos em C seguem o layout column-major. A ordem dos laços foi mantida compatível com os índices desse armazenamento para permitir a progressão até SIMD e blocking.

### 6.3 Vetorização SIMD

O Chapter3 processa quatro valores `double` por registrador AVX2. O Chapter4 acrescenta FMA e quatro acumuladores SIMD. A análise e os dados estão em [Parte2_DGEMM.md](Parte2_DGEMM.md) e [resultados_parte2.csv](resultados_parte2.csv).

### 6.4 Blocking ou tiling

O Chapter5 usa blocking parametrizado e a campanha avalia blocos de `8`, `16`, `32` e `64`. O kernel bloqueado mantém AVX2 e FMA com um acumulador vetorial para que o tamanho de bloco seja comparado sem adicionar unrolling nessa variante. Os resultados estão em [Parte3_DGEMM.md](Parte3_DGEMM.md).

### 6.5 Paralelismo

O Chapter6 distribui os blocos de colunas com OpenMP; a campanha mede 1, 2 e 4 threads nos tamanhos `N=128` e `N=256`.

### 6.6 Bibliotecas de referência

MKL e PyTorch não foram avaliados neste trabalho. O WSL 2 com Ubuntu 24.04 está instalado, mas CUDA no WSL não foi validado: o driver NVIDIA do Windows é a versão 451.67, abaixo da versão R495 indicada pela NVIDIA para CUDA no WSL, e a GeForce MX110, baseada em Maxwell, não tem suporte oficial nesse ambiente. Assim, os resultados apresentados se restringem às implementações C executadas na CPU; não há comparação com PyTorch nem com GPU.

## 7. Resultados

As tabelas preliminares estão nos relatórios parciais. A campanha ampliada contém 200 observações em [resultados_campanha.csv](resultados_campanha.csv); o script [analisar_resultados.py](analisar_resultados.py) calcula média, mediana, desvio padrão, mínimo e máximo por variante, dimensão, bloco e número de threads.

### 7.1 Efeito do tamanho da matriz

Na campanha preliminar do Windows, o baseline obteve `2,45`, `2,63` e `2,53 GFLOPS` nas três repetições de `N=32`; as execuções preliminares de `N=64` e `N=128` produziram `2,28` e `1,93 GFLOPS`. Esses valores não pertencem à campanha pareada do WSL.

### 7.2 Campanha ampliada: variantes SIMD

Medianas em GFLOPS das cinco repetições, no WSL 2:

| N | Escalar | AVX2 | AVX2 + FMA | AVX2 + FMA + unrolling x4 |
|---:|---:|---:|---:|---:|
| 128 | 1,91 | 7,77 | 7,88 | 20,98 |
| 256 | 1,06 | 4,86 | 4,21 | 9,60 |

FMA isolado não apresentou ganho claro em relação a AVX2 nesta medição; a versão com quatro acumuladores teve mediana maior nos dois tamanhos.

### 7.3 Campanha ampliada: blocking e OpenMP

Medianas em GFLOPS. A coluna Blocking usa uma thread; as colunas OpenMP mostram a mesma variante com o número indicado de threads.

| N | Bloco | Blocking, 1 thread | OpenMP, 1 thread | OpenMP, 2 threads | OpenMP, 4 threads |
|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 10,32 | 5,81 | 22,96 | 45,44 |
| 128 | 16 | 11,78 | 11,88 | 21,07 | 47,34 |
| 128 | 32 | 10,07 | 6,47 | 23,39 | 46,40 |
| 128 | 64 | 8,69 | 9,08 | 18,12 | 17,85 |
| 256 | 8 | 9,04 | 9,93 | 12,78 | 27,69 |
| 256 | 16 | 9,75 | 9,77 | 21,25 | 28,53 |
| 256 | 32 | 7,57 | 8,49 | 18,55 | 22,69 |
| 256 | 64 | 6,95 | 8,60 | 18,08 | 35,87 |

O melhor resultado depende do tamanho e do bloco: em `N=128`, bloco 16 com quatro threads obteve mediana de `47,34 GFLOPS`; em `N=256`, bloco 64 com quatro threads obteve `35,87 GFLOPS`. A dispersão de cada grupo está nos dados brutos e deve ser considerada ao interpretar diferenças próximas.

### 7.4 Validação numérica

O validador usa os mesmos kernels compartilhados que os executáveis dos Chapters. Em `N=32`, `128` e `256`, todas as variantes, blocos válidos e configurações de 1, 2 e 4 threads obtiveram `PASS`; o erro absoluto máximo registrado foi zero, com tolerância `1e-10`.

Os resultados preliminares de `N=32` no Windows foram mantidos nos CSVs das partes como registros históricos. Eles foram medidos com outra campanha/ambiente e não devem ser comparados diretamente aos resultados pareados no WSL 2.

## 8. Discussão

Na campanha ampliada, AVX2 melhorou as medianas em relação ao escalar nos dois tamanhos. FMA sem unrolling teve efeito pequeno e não consistente, enquanto a combinação com quatro acumuladores atingiu medianas de `20,98 GFLOPS` em `N=128` e `9,60 GFLOPS` em `N=256`. Esses valores são específicos do compilador e ambiente usados.

No blocking sem OpenMP, o bloco 16 teve a maior mediana entre os tamanhos testados tanto em `N=128` quanto em `N=256` (11,78 e 9,75 GFLOPS). Considerando OpenMP, a melhor combinação mudou: bloco 16 com quatro threads em `N=128` e bloco 64 com quatro threads em `N=256`. Isso reforça a necessidade de avaliar conjuntamente dimensão, bloco e paralelismo.

Na campanha preliminar de `N=32`, OpenMP não compensou; em `N=128` e `N=256`, várias combinações de bloco e threads aumentaram o throughput, embora com variabilidade e sensibilidade aos parâmetros. Assim, o resultado da matriz pequena não deve ser generalizado para problemas maiores.

As conclusões são específicas do Intel Core i5-10210U, do GCC 13.3 e da execução no WSL 2. Os testes ampliados mitigam a limitação de `N=32`, mas ainda cobrem apenas duas dimensões e quatro blocos. MKL e PyTorch não foram avaliados, e CUDA no WSL não foi validado pelas limitações do driver e da GeForce MX110.

Como continuação, seria útil repetir a campanha em matrizes ainda maiores, testar mais blocos e coletar execuções em outras arquiteturas para avaliar a generalização dos resultados.

A leitura dos resultados deve considerar também os seguintes pontos:

- quando o problema deixa de ser limitado principalmente pela computação;
- quando o acesso à memória e a cache passam a dominar;
- quais otimizações apresentam ganho consistente;
- quais configurações não melhoram o desempenho e por quê;
- como a variabilidade das medições afeta as conclusões.

## 9. Conclusão

Os resultados pareados mostram ganhos de SIMD e de unrolling em relação ao escalar, além de ganhos de blocking/OpenMP que dependem de `N`, bloco e threads. Isso sustenta a hipótese de que o alinhamento entre implementação e microarquitetura pode elevar o desempenho, mas também evidencia que nenhuma configuração é universalmente superior.

OpenMP reduziu o desempenho na medição preliminar de `N=32`, mas melhorou o throughput em diversas combinações de `N=128` e `N=256`. O resultado depende da quantidade de trabalho disponível e da escolha do bloco.

A corretude foi validada diretamente sobre os kernels compartilhados usados pelos executáveis, para `N=32`, `128` e `256`, com todos os casos testados aprovados. Permanecem como limitações a quantidade de dimensões, a ausência de comparação com MKL/PyTorch e a impossibilidade de validar CUDA no WSL com a configuração atual.

## 10. Requisitos e limitações do ambiente

As medições preliminares foram feitas em Windows com Intel Core i5-10210U e GCC 8.1.0. A campanha ampliada foi executada em Ubuntu 24.04 sob WSL 2, com GCC 13.3.0, 4 núcleos/8 threads e AVX2/FMA/OpenMP disponíveis. CUDA não foi validado: o driver NVIDIA 451.67 é inferior ao R495 indicado para CUDA no WSL e a MX110, baseada em Maxwell, não tem suporte oficial nesse ambiente. Não houve comparação com PyTorch ou GPU.

## 11. Referências

- PATTERSON, David A.; HENNESSY, John L. *Computer Organization and Design RISC-V Edition: The Hardware/Software Interface*. 2nd ed. Morgan Kaufmann, 2021. ISBN 978-0-12-820331-6.
- Documentação das bibliotecas utilizadas e outras fontes consultadas.
