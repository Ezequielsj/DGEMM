# Parte 2 - Vetorizacao e paralelismo de instrucoes no DGEMM

## Links do projeto no GitHub

- Repositório: [DGEMM](https://github.com/Ezequielsj/DGEMM)
- Este relatório: [Parte2_DGEMM.md](https://github.com/Ezequielsj/DGEMM/blob/master/Parte2_DGEMM.md)
- Resultados: [resultados_parte2.csv](https://github.com/Ezequielsj/DGEMM/blob/master/resultados_parte2.csv)
- Campanha ampliada: [resultados_campanha.csv](https://github.com/Ezequielsj/DGEMM/blob/master/resultados_campanha.csv)
- Implementação AVX2: [Chapter3/main_algorithm.c](https://github.com/Ezequielsj/DGEMM/blob/master/Chapter3/main_algorithm.c)
- Implementação AVX2 + FMA + unrolling: [Chapter4/main_algorithm.c](https://github.com/Ezequielsj/DGEMM/blob/master/Chapter4/main_algorithm.c)

## Integrantes

- Ezequiel de Jesus Santos - 121056350
- Kelly Pinheiro Soares - 125170716
- Luiza Teixeira Barcellos Rosauro de Almeida - 126423803
- Lucas Pereira Pacheco de Medeiros - 126436084

## 1. Objetivo

Nesta segunda parte, a implementacao escalar da Parte 1 foi usada como referencia para investigar duas formas de explorar paralelismo dentro do processador: vetorizacao SIMD com AVX2 e aumento do numero de acumuladores por meio de desenrolamento de lacos. As versoes foram compiladas e executadas no mesmo ambiente local.

## 2. Otimizacao AVX2

O Chapter3 usa registradores `__m256d`, que comportam quatro valores `double` de 64 bits. A operacao de quatro elementos de `C` e atualizada simultaneamente com `_mm256_load_pd`, `_mm256_mul_pd`, `_mm256_broadcast_sd`, `_mm256_add_pd` e `_mm256_store_pd`.

As matrizes sao alocadas com alinhamento de 32 bytes, necessario para os carregamentos alinhados usados pelo codigo. Por isso, esta versao exige que `N` seja multiplo de 4.

## 3. FMA e loop unrolling

O Chapter4 preserva a vetorizacao AVX2 e acrescenta:

- quatro acumuladores SIMD (`UNROLL=4`);
- processamento de 16 elementos da dimensao `i` por grupo;
- instrucao `_mm256_fmadd_pd`, que combina multiplicacao e soma em uma unica operacao vetorial.

Essa organizacao reduz parte do custo de controle dos lacos e aumenta o paralelismo de instrucoes disponivel para o processador. Nesta versao, `N` deve ser multiplo de 16.

## 4. Reproducao

Os resultados da tabela preliminar foram coletados no Windows, antes da consolidacao dos kernels compartilhados. Os CSVs originais foram mantidos como registro historico; os executaveis atuais usam os kernels comuns de [dgemm_kernels.h](dgemm_kernels.h).

Para compilar as versoes atuais e repetir a campanha pareada:

```text
make ch3 ch4 benchmark
./benchmark_dgemm.exe resultados_campanha.csv 5 1
```

## 5. Resultados preliminares

| Versao | Repeticoes | N | GFLOPS medio | Mediana | Desvio padrao | Speedup medio |
|---|---:|---:|---:|---:|---:|---:|
| C escalar, Parte 1 | 3 | 32 | 2,53 | 2,53 | 0,09 | 1,00x |
| AVX2 | 3 | 32 | 12,05 | 12,16 | 0,35 | 4,76x |
| AVX2 + FMA + unrolling | 3 | 32 | 28,03 | 28,05 | 1,17 | 11,08x |

Os GFLOPS foram calculados por `2 * N^3 * multiplicacoes / (tempo * 10^9)`. As estatisticas usam as tres repeticoes de cada versao. O speedup medio foi calculado em relacao a media de `2,53 GFLOPS` do C escalar.

## 6. Analise

Com tres repeticoes, a vetorizacao AVX2 atingiu media de `12,05 GFLOPS`, aproximadamente `4,76x` o baseline. A combinacao de FMA e quatro acumuladores atingiu `28,03 GFLOPS`, aproximadamente `11,08x` o baseline e cerca de `2,33x` a media do AVX2.

Esse ganho e consistente com a ideia de SIMD: ao processar quatro valores em paralelo, a implementacao reduz a quantidade de instrucoes de carga, multiplicacao e soma que o processador precisa executar. A adicao de FMA e desenrolamento de laços aumenta ainda mais o paralelismo de instrucoes e reduz a sobrecarga de controle do loop, o que explica a melhora significativa observada em relacao ao AVX2 puro.

Esses valores sao preliminares e pertencem a outra campanha. A validacao numerica atual foi ampliada: [validate_dgemm.c](validate_dgemm.c) testa os mesmos kernels usados pelos executaveis finais em `N=32`, `128` e `256`. A campanha pareada tambem mede FMA sem unrolling separadamente e esta resumida abaixo.

A execucao confirmou que as duas implementacoes otimizadas compilam e funcionam no ambiente atual. O teste independente [validate_dgemm.c](validate_dgemm.c) tambem confirmou erro maximo `0` para AVX2 e AVX2 + FMA + unrolling, usando tolerancia `1e-12`. O blocking e o OpenMP sao apresentados na [Parte 3](Parte3_DGEMM.md).

## 7. Campanha ampliada: FMA e unrolling separados

Para separar os efeitos de FMA e desenrolamento, a campanha complementar mede AVX2 sem FMA, AVX2 + FMA sem unrolling e AVX2 + FMA com quatro acumuladores. Foram usados os mesmos dados determinísticos para todas as variantes de cada dimensão, um aquecimento e cinco repetições medidas por configuração. Os tempos foram medidos com o mesmo relógio monotônico de parede.

| N | Variante | Mediana (GFLOPS) | Amostras |
|---:|---|---:|---:|
| 128 | Escalar | 1,91 | 5 |
| 128 | AVX2 | 7,77 | 5 |
| 128 | AVX2 + FMA | 7,88 | 5 |
| 128 | AVX2 + FMA + unrolling x4 | 20,98 | 5 |
| 256 | Escalar | 1,06 | 5 |
| 256 | AVX2 | 4,86 | 5 |
| 256 | AVX2 + FMA | 4,21 | 5 |
| 256 | AVX2 + FMA + unrolling x4 | 9,60 | 5 |

Nesta campanha, FMA sem unrolling não trouxe ganho claro em relação ao AVX2 isolado; já a variante com quatro acumuladores apresentou throughput maior. A interpretação deve considerar a dispersão das cinco medições e que este complemento foi executado no Ubuntu 24.04 sob WSL 2, com GCC 13.3, enquanto os resultados preliminares desta parte foram coletados no Windows com GCC 8.1. Os dois conjuntos não devem ser comparados diretamente como se fossem a mesma execução.

## 8. Proxima etapa

A Parte 3 acrescenta cache blocking/tiling e investiga OpenMP e o numero de threads. Essa etapa e importante porque ela verifica se o ganho de SIMD continua relevante quando a organizacao da memoria e a distribuicao do trabalho passam a ser fatores limitantes.

## Requisitos e limitacoes do ambiente

Os resultados preliminares foram coletados em Windows com Intel Core i5-10210U, 4 nucleos, 8 threads e GCC 8.1.0. A campanha ampliada foi executada no Ubuntu 24.04 sob WSL 2, no mesmo processador exposto ao Linux, com GCC 13.3.0 e suporte a AVX2, FMA e OpenMP. MKL e PyTorch nao foram avaliados. CUDA no WSL nao foi validado: o driver NVIDIA do Windows e a versao 451.67, abaixo da versao R495 indicada pela NVIDIA, e a GeForce MX110, baseada em Maxwell, nao tem suporte oficial nesse ambiente.

## Referencia

PATTERSON, David A.; HENNESSY, John L. *Computer Organization and Design RISC-V Edition: The Hardware/Software Interface*. 2nd ed. Morgan Kaufmann, 2021. ISBN 978-0-12-820331-6.
