# Parte 2 - Vetorizacao e paralelismo de instrucoes no DGEMM

Repositorio: https://github.com/Ezequielsj/DGEMM

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

## 4. Compilacao e execucao

As versoes foram compiladas diretamente com GCC, pois `make` nao esta disponivel:

```text
gcc -O3 -Wall -mavx2 -mfma -o Chapter3/parte2_simd.exe Chapter3/main_algorithm.c
gcc -O3 -Wall -mavx2 -mfma -o Chapter4/parte2_unroll.exe Chapter4/main_algorithm.c
```

Foram feitas tres repeticoes para cada versao, com `N=32` e janela de 0,5 segundos:

```text
Chapter3/parte2_simd.exe 32 0.5
Chapter4/parte2_unroll.exe 32 0.5
```

As duas execucoes terminaram com codigo de saida 0. Tambem foi corrigida a formatacao da mensagem de tamanho das matrizes para evitar avisos do GCC 8 no ambiente Windows.

## 5. Resultados preliminares

| Versao | Repeticoes | N | GFLOPS medio | Mediana | Desvio padrao | Speedup medio |
|---|---:|---:|---:|---:|---:|---:|
| C escalar, Parte 1 | 3 | 32 | 2,53 | 2,53 | 0,09 | 1,00x |
| AVX2 | 3 | 32 | 12,05 | 12,16 | 0,35 | 4,76x |
| AVX2 + FMA + unrolling | 3 | 32 | 28,03 | 28,05 | 1,17 | 11,08x |

Os GFLOPS foram calculados por `2 * N^3 * multiplicacoes / (tempo * 10^9)`. As estatisticas usam as tres repeticoes de cada versao. O speedup medio foi calculado em relacao a media de `2,53 GFLOPS` do C escalar.

## 6. Analise

Com tres repeticoes, a vetorizacao AVX2 atingiu media de `12,05 GFLOPS`, aproximadamente `4,76x` o baseline. A combinacao de FMA e quatro acumuladores atingiu `28,03 GFLOPS`, aproximadamente `11,08x` o baseline e cerca de `2,33x` a media do AVX2.

Esses valores sao uma campanha preliminar: a janela de 0,5 segundos e tres repeticoes permitem comparar as versoes, mas a validacao numerica automatica contra uma referencia ainda precisa ser acrescentada ao experimento final.

A execucao confirmou que as duas implementacoes otimizadas compilam e funcionam no ambiente atual. O teste independente [validate_dgemm.c](validate_dgemm.c) tambem confirmou erro maximo `0` para AVX2 e AVX2 + FMA + unrolling, usando tolerancia `1e-12`. O blocking e o OpenMP sao apresentados na [Parte 3](Parte3_DGEMM.md).

## 7. Proxima etapa

A Parte 3 acrescenta cache blocking/tiling e investiga OpenMP e o numero de threads.

## Requisitos e limitacoes do ambiente

Os testes foram realizados em Windows, com Intel Core i5-10210U, 4 nucleos, 8 threads, aproximadamente 8 GB de RAM e GCC 8.1.0. O ambiente suportou as flags `-mavx2`, `-mfma` e OpenMP. O comando `make` nao estava instalado, e MKL, PyTorch e CUDA nao foram utilizados por indisponibilidade das dependencias.

## Referencia

PATTERSON, David A.; HENNESSY, John L. *Computer Organization and Design: The Hardware/Software Interface, RISC-V Edition*. Morgan Kaufmann.
