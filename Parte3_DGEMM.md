# Entrega Final - Blocking, OpenMP e consolidacao

## Links do projeto no GitHub

- Repositório: [DGEMM](https://github.com/Ezequielsj/DGEMM)
- Este relatório: [Parte3_DGEMM.md](https://github.com/Ezequielsj/DGEMM/blob/master/Parte3_DGEMM.md)
- Resultados: [resultados_parte3.csv](https://github.com/Ezequielsj/DGEMM/blob/master/resultados_parte3.csv)
- Campanha ampliada: [resultados_campanha.csv](https://github.com/Ezequielsj/DGEMM/blob/master/resultados_campanha.csv)
- Implementação com blocking: [Chapter5/main_algorithm.c](https://github.com/Ezequielsj/DGEMM/blob/master/Chapter5/main_algorithm.c)
- Implementação com OpenMP: [Chapter6/main_algorithm.c](https://github.com/Ezequielsj/DGEMM/blob/master/Chapter6/main_algorithm.c)

## Integrantes

- Ezequiel de Jesus Santos - 121056350
- Kelly Pinheiro Soares - 125170716
- Luiza Teixeira Barcellos Rosauro de Almeida - 126423803
- Lucas Pereira Pacheco de Medeiros - 126436084

## 1. Objetivo

Esta entrega final acrescenta localidade de cache por blocking e paralelismo entre blocos com OpenMP. Tambem consolida a progressao observada nas duas entregas parciais e registra a avaliacao do efeito do numero de threads.

## 2. Blocking ou tiling

O Chapter5 usa blocking parametrizado: a campanha avaliou blocos de `8`, `16`, `32` e `64`. O kernel interno usa AVX2 e FMA com um acumulador vetorial; isso permite comparar todos os blocos com a mesma estrutura interna, inclusive o bloco 8. `N` deve ser multiplo do bloco, e o bloco deve ser multiplo de 4.

A hipotese experimental e que o tamanho do bloco altera a reutilizacao de dados na cache. Nesta implementacao, `N` deve ser divisivel pelo tamanho de bloco escolhido, e ambos devem respeitar os multiplos de 4 exigidos pelo kernel AVX2.

## 3. OpenMP

O Chapter6 aplica `#pragma omp parallel for` ao percurso dos blocos da dimensao `j`. Como cada thread trabalha com colunas diferentes de `C`, os blocos distribuidos nao devem escrever na mesma regiao durante uma multiplicacao.

Foram testadas 1, 2 e 4 threads. O laço OpenMP distribui os blocos de colunas, e a campanha inclui o custo de iniciar e sincronizar a regiao paralela dentro do tempo medido.

## 4. Compilacao e execucao

```text
make ch5 ch6 benchmark validate
./Chapter5/program.exe 128 16
./Chapter6/program.exe 128 16 4
./benchmark_dgemm.exe resultados_campanha.csv 5 1
```

Os resultados originais desta parte usaram `N=32` e janela de 0,5 segundos, antes da consolidacao dos kernels compartilhados. Essa dimensao pequena nao permitia avaliar escalabilidade; a campanha ampliada abaixo complementa as medidas historicas com `N=128` e `N=256`.

## 5. Resultados

| Versao | Threads | N | Multiplicacoes | Tempo total (s) | GFLOPS | Speedup vs. blocking |
|---|---:|---:|---:|---:|---:|---:|
| Blocking | 1 | 32 | 194382 | 0,50 | 25,48 | 1,00x |
| OpenMP | 1 | 32 | 76749 | 0,50 | 10,06 | 0,39x |
| OpenMP | 2 | 32 | 16057 | 0,50 | 2,10 | 0,08x |
| OpenMP | 4 | 32 | 8666 | 0,50 | 1,14 | 0,04x |

Os valores foram calculados por `2 * N^3 * multiplicacoes / (tempo * 10^9)`. Cada configuracao terminou com sucesso. O compilador foi executado novamente depois da correcao das mensagens de tamanho, sem erros de compilacao.

## 6. Campanha ampliada: blocking e threads

A campanha foi executada no Ubuntu 24.04 sob WSL 2, com GCC 13.3.0, no Intel Core i5-10210U exposto ao WSL. Para cada `N`, todas as variantes receberam as mesmas matrizes deterministicas; `C` foi zerada a cada execucao. Houve um aquecimento e cinco repeticoes medidas por configuracao. A tabela mostra as medianas em GFLOPS; os dados brutos, incluindo tempos e dispersao, estao no CSV da campanha.

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

No blocking sem OpenMP, bloco 16 teve a maior mediana entre os blocos testados tanto em `N=128` quanto em `N=256` (11,78 e 9,75 GFLOPS). Considerando OpenMP, a melhor combinacao mudou: bloco 16 com quatro threads em `N=128` (47,34 GFLOPS) e bloco 64 com quatro threads em `N=256` (35,87 GFLOPS). Isso mostra por que dimensao, bloco e numero de threads precisam ser avaliados em conjunto. Os resultados sao especificos deste processador e desta execucao em WSL; a variabilidade pode ser consultada no arquivo CSV.

## 7. Validacao numerica

O programa [validate_dgemm.c](validate_dgemm.c) compara as implementacoes compartilhadas efetivamente chamadas pelos executaveis dos Chapters 2 a 6 com uma referencia escalar independente. Foram validados `N=32`, `128` e `256`, todos os blocos compativeis com cada dimensao e 1, 2 e 4 threads; todos os casos passaram com erro maximo `0` e tolerancia absoluta `1e-10`.

## 8. Discussao

Na medicao preliminar de `N=32`, o OpenMP nao superou o blocking e o throughput caiu ao aumentar as threads. A campanha ampliada mostra o quadro complementar: em `N=128` e `N=256`, varias configuracoes com 2 ou 4 threads superaram a variante bloqueada de uma thread. O resultado confirma que a conclusao depende do tamanho do problema e do bloco.

Esse comportamento evidencia que paralelismo e localidade de memoria devem ser avaliados em conjunto. O blocking melhora a reutilizacao de dados na cache e reduz a distancia entre os acessos consecutivos, enquanto o OpenMP acrescenta overhead de sincronizacao e distribuicao de trabalho. Em matrizes pequenas, esse overhead tende a dominar, e o ganho de paralelismo desaparece ou se torna negativo.

Portanto, nao e correto concluir que OpenMP sempre piora o DGEMM. Em `N=32`, havia trabalho insuficiente para amortizar o custo de paralelismo; nos tamanhos maiores, o throughput aumentou em diversas configuracoes. O melhor bloco variou entre `N=128` e `N=256`, evidenciando que nao existe um unico parametro otimo para todos os cenarios.

## 9. Limitacoes e pendencias

- A campanha ampliada avaliou `N=128` e `N=256` e quatro tamanhos de bloco, mas nao cobre todos os tamanhos de matriz possiveis.
- Os resultados de `N=32` sao preliminares e nao servem para avaliar escalabilidade OpenMP; a campanha ampliada testa tambem dimensoes maiores.
- A validacao numerica foi executada nas variantes compartilhadas para `N=32`, `128` e `256`, com todos os blocos validos e 1, 2 e 4 threads.
- MKL e PyTorch nao foram avaliados. Embora o WSL 2 com Ubuntu 24.04 esteja instalado, CUDA no WSL nao foi validado: o driver NVIDIA do Windows e a versao 451.67, abaixo da versao R495 indicada pela NVIDIA, e a GeForce MX110, baseada em Maxwell, nao tem suporte oficial nesse ambiente.
- A campanha cobre dois tamanhos maiores e quatro blocos, mas ainda nao caracteriza matrizes muito maiores nem outras arquiteturas.
- A variabilidade de algumas configuracoes e elevada; por isso, a tabela reporta medianas e o CSV preserva todas as observacoes.

## 10. Conclusao

Os experimentos mostram que blocking e OpenMP dependem dos parametros e do tamanho da matriz. Em `N=128`, bloco 16 com quatro threads obteve mediana de `47,34 GFLOPS`; em `N=256`, bloco 64 com quatro threads obteve `35,87 GFLOPS`. Esses resultados descrevem esta plataforma e campanha; nao devem ser generalizados para outras arquiteturas sem novos testes.

## Requisitos e limitacoes do ambiente

Os resultados iniciais foram obtidos em Windows com GCC 8.1.0; a campanha ampliada foi medida em Ubuntu 24.04 sob WSL 2 com GCC 13.3.0, usando o mesmo Intel Core i5-10210U e suporte a AVX2, FMA e OpenMP. MKL e PyTorch nao foram avaliados. CUDA no WSL nao foi validado pelas limitacoes de driver e de suporte da GPU descritas acima.

## Referencia

PATTERSON, David A.; HENNESSY, John L. *Computer Organization and Design RISC-V Edition: The Hardware/Software Interface*. 2nd ed. Morgan Kaufmann, 2021. ISBN 978-0-12-820331-6.
