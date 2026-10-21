# Migração do núcleo C++ para Rust

Documento de referência da conversão de `src/` (C++/pybind11/OpenMP/Eigen) para
Rust (PyO3/rayon). Objetivo: **nenhuma perda de desempenho nem de fidelidade**;
ganhar onde for possível.

## Sequência escolhida

1. **Etapa A: melhorias no C++**, que viram a nova linha de base de desempenho.
2. **Etapa B: Rust e C++ lado a lado** (`zeit._core` e `zeit._rust` no mesmo
   wheel), módulo a módulo, com testes diferenciais e benchmarks contra a linha
   de base da etapa A.
3. **Etapa C: corte final**, removendo C++ e Eigen quando o último módulo
   passar. Feita na 2.0.0.

## Retrato do ponto de partida (v1.7.0)

- ≈10,5 mil linhas em 21 módulos: `warp` (1356), `landtrendr` (1190), `ccdc`
  (932), `bfast` (604), `mann_kendall` (514), `phenology_*` (~1300), `som`
  (458), `snic` (318), `twdtw` (311), `stl_decompose`, `bfast_lite`,
  `bfast_monitor`, `coded`, `sma`, `whittaker`, `tmask`, `utils`.
- pybind11: ~130 `.def`, 10 classes, um módulo `zeit._core`.
- OpenMP em quase todo módulo (`schedule(dynamic, N)`, buffers por thread,
  `thread_local`, um `critical`).
- Eigen: `VectorXd`/`MatrixXd`; `LDLT` e `ColPivHouseholderQR` em matrizes
  pequenas; `LevenbergMarquardt` + `NumericalDiff` (unsupported) na fenologia;
  `SimplicialLLT` esparso no Whittaker.
- **Contratos de fidelidade bit a bit** (docs/benchmarks/fidelity.md):
  - CCDC: GLMnet Fortran em float32, operação por operação (`-ffp-contract=off`);
  - LandTrendr: anos de vértice 330/330 contra o IDL;
  - SNIC: rótulos idênticos ao `snic.c`;
  - SOM: diferença 0.0 contra o `minisom` (`exp`/`pow` do runtime C, soma em
    pares do NumPy);
  - `warp`: kernel do GDAL na mesma ordem de operações.

A dificuldade não é escrever Rust rápido. É reescrever 10 mil linhas sem mudar
um bit onde isso é contrato.

## Etapa A: melhorias no C++ (nova linha de base)

| Item | Status | Nota |
|---|---|---|
| Harness de benchmark (`benchmarks/bench_core.py`) | feito | Cargas determinísticas, 1 thread (fixada num núcleo P: o i5-13600K é híbrido) e todas; vários backends medidos em turnos no mesmo processo; saída JSON |
| Liberar o GIL nas funções que ainda não liberam | feito | ccdc, utils (medoid), twdtw, phenology, mann_kendall, bfast, bfast_monitor, bfast_lite e o `fit_trajectory_batch` do LandTrendr; `tests/test_gil.py` verifica (falha em 7 de 8 com o v1.7.0) |
| Correções de passagem | feito | LandTrendr usava `omp_set_num_threads` (estado global do OpenMP) em vez de `num_threads`; índices `int` que estouravam acima de 2^31 valores (LandTrendr, TWDTW, medoid) viraram `size_t` |
| TWDTW: tabela do peso temporal uma vez por lote | feito | O C++ montava a tabela (2001 `exp`) a cada pixel; agora uma vez por lote: 191 → 115 ms no benchmark, mesmos bits |
| OpenMP nas wheels de macOS | adiado | O `libomp` do Homebrew obrigaria as wheels a exigir o macOS do runner (14+), ou o `llvm-openmp` do conda-forge; sem como verificar localmente. O rayon resolve o macOS módulo a módulo na etapa B |
| Windows: clang-cl ou `/openmp:llvm` | descartado | O `libomp140.x86_64.dll` do `/openmp:llvm` só vem em `debug_nonredist` (não redistribuível); clang-cl exigiria empacotar o libomp do LLVM com delvewheel. O Rust já traz LLVM no Windows |
| pybind11 → nanobind | descartado | Os bindings serão substituídos por PyO3 na etapa B; seria trabalho jogado fora |

Linha de base medida: `benchmarks/results/win-i5-13600k-cpp-v1.7.0.json` (antes) e
`win-i5-13600k-cpp-baseline.json` (depois). Sem regressão: o código numérico não
mudou, exceto o TWDTW.

## Etapa B: Rust e C++ lado a lado

### Equivalências

| Hoje | Em Rust | Observação |
|---|---|---|
| pybind11 | PyO3 + `numpy` (rust-numpy) | Arrays sem cópia via `PyReadonlyArray` / `as_slice()` |
| `gil_scoped_release` | `py.detach` (antigo `allow_threads`) em **todas** as funções | |
| OpenMP | rayon | Um pool em cache por `n_jobs`; `map_init` no lugar de buffers por thread e `thread_local`; `with_min_len` no lugar de `dynamic, N` |
| `omp critical` | `Mutex` ou redução | Corridas de dados viram erro de compilação |
| `-ffp-contract=off` | padrão do Rust | Rust nunca funde em FMA sem `mul_add` explícito |
| `exp`/`pow` do runtime C | `f64::exp` / `powf` | Mesma libm da plataforma. **Não** usar o crate `libm` |
| Eigen denso | slices + rotinas próprias pequenas | `faer` só onde houver matrizes grandes |
| `LDLT`, `ColPivHouseholderQR` (k pequeno) | port literal do algoritmo do Eigen | faer/nalgebra mudariam o arredondamento |
| `LevenbergMarquardt` (unsupported) | port literal (~600 linhas, MINPACK `lmdif`) | |
| `SimplicialLLT` (Whittaker) | Cholesky banda pentadiagonal | |
| setuptools + pybind11 | setuptools-rust (transição) → maturin (corte final) | |

### Organização

- `rust/zeit-core`: Rust puro, sem PyO3, testável com `cargo test`.
- `rust/zeit-py`: só bindings; compila para `zeit._rust`.
- `setup.py` constrói `zeit._core` (C++) e `zeit._rust` (Rust, via
  setuptools-rust) no mesmo wheel.
- `zeit/_backend.py`: escolhe o backend por módulo. `ZEIT_BACKEND=cpp|rust`
  força um deles; sem a variável, cada módulo usa Rust quando ele já foi
  convertido e aprovado.
- `tests/test_rust_parity.py`: testes diferenciais C++ × Rust, bit a bit onde o
  contrato é exato e em ULPs onde o Eigen decide a ordem das reduções.

### Critério de aprovação de cada módulo

1. Diferencial contra o C++ sobre entradas determinísticas grandes, incluindo
   casos-limite (NaN, séries curtas, empates).
2. Suítes de paridade contra o código dos autores (IDL, Octave, `snic.c`,
   minisom, R) continuam passando. **Elas são o contrato**; o C++ é só a
   referência intermediária.
3. Desempenho ≥ C++ (linha de base da etapa A), em 1 thread e em todas.
4. Suíte inteira (`pytest tests`) verde com `ZEIT_BACKEND=rust`.

### Fases e status

| Fase | Módulos | Status |
|---|---|---|
| 0. Infraestrutura | workspace Cargo, `zeit._rust`, `_backend.py`, testes diferenciais, CI | feito |
| 1. Piloto (ponto de decisão) | `mann_kendall`, `twdtw`, `snic` | feito; suíte verde no CI nos 3 sistemas (os tempos no Linux e no macOS saem no resumo do job Tests) |
| 2. Kernels simples | `utils` (medoid), `whittaker`, `sma`, `coded`, `tmask` (com o `robustfit` do CCDC) | feito |
| 3. Regressão e decomposição | `stl_decompose`, `bfast`, `bfast_monitor`, `bfast_lite` | feito |
| 4. SOM | `som` | feito |
| 5. LandTrendr | `landtrendr` | feito |
| 6. Fenologia | `phenology_*` | feito, bit a bit como os outros módulos (decisão de 2026-10-21): LM do Eigen (`NonLinearOptimization`), `SimplicialLLT` com ordenação AMD |
| 7. CCDC | `ccdc` (GLMnet float32) | feito |
| 8. Warp | `warp` (callback do transformer do GDAL) | feito |

**Ponto de decisão após a fase 1:** seguir só se os três módulos do piloto
tiverem desempenho ≥ C++ nos 3 sistemas e mantiverem a paridade exata.

### Resultados do piloto (Windows, i5-13600K)

Os três módulos dão **os mesmos bits** que o C++ (`tests/test_rust_parity.py`:
NaN, empates, séries curtas, 1 a 9 bandas, float32 e float64, tiles, 1 e várias
threads). Tempos contra a linha de base da etapa A (C++ já melhorado), medidos
em turnos no mesmo processo (`benchmarks/results/win-i5-13600k-pilot.json`):

| Módulo | Threads | C++ | Rust | Rust/C++ |
|---|---|---:|---:|---:|
| Mann-Kendall (50 mil séries × 40) | 1 | 598 ms | 287 ms | 0,48 |
| | todas | 50,8 ms | 26,3 ms | 0,52 |
| TWDTW (10 mil séries × 46, padrão de 23) | 1 | 115 ms | 21,8 ms | 0,19 |
| | todas | 9,4 ms | 2,5 ms | 0,26 |
| SNIC (1024², 4 bandas, tiles de 256) | 1 | 142 ms | 132 ms | 0,93 |
| | todas | 23,3 ms | 21,4 ms | 0,92 |

Suíte inteira: 803 testes com o C++ e os mesmos com `ZEIT_BACKEND=rust`, todos verdes.

Parte do ganho deve vir de o C++ do Windows ser compilado pelo MSVC e o Rust pelo
LLVM; o CI no Linux (GCC) e no macOS (Clang) dirá quanto é da linguagem.
`RUST_BY_DEFAULT` em `zeit/_backend.py` continua vazio até lá: o padrão segue
sendo o C++, e o Rust se escolhe com `ZEIT_BACKEND=rust`.

### Lições do piloto

- **A ordem das somas do Eigen** se reproduz (`rust/zeit-core/src/eigen.rs`):
  pacotes de 2 doubles (SSE2/NEON), dois acumuladores, soma horizontal, cauda
  escalar. Com float32, o `cast<double>()` do Eigen 3.4 não vetoriza, e a soma é
  sequencial. Os dois casos foram confirmados bit a bit.
- **Funções que o Rust não tem** (`erfc`) vêm do runtime C, o mesmo que o C++ usa
  (`rust/zeit-core/src/cmath.rs`). O crate `libm` daria outros bits.
- **setuptools-rust compila em debug nos builds in-place** (`build_ext --inplace`)
  se não houver `debug=False`, e o Rust fica 10× mais lento. Já está no `setup.py`.
- **Medir antes de otimizar.** O SNIC saiu 30% mais lento, e nenhuma das
  hipóteses óbvias (checagens de limites, closures, aliasing) explicava. A causa
  era um desvio imprevisível na escolha do filho no heap: o MSVC gera um `cmov`,
  o LLVM gerou um desvio. A escolha aritmética (`j + right_smaller as usize`)
  deixou o Rust 9% mais rápido que o C++.

### Resultados da fase 2 (Windows, i5-13600K)

Os cinco módulos dão os mesmos bits que o C++. Para isso foi preciso portar o
`ColPivHouseholderQR` do Eigen com seus produtos (`rust/zeit-core/src/eigen/qr.rs`),
peça que serve também a CCDC e BFAST Lite.

Tempos em turnos no mesmo processo, todos os módulos, depois das fases 6 a 8
(`benchmarks/results/win-i5-13600k-rust.json`; 1 thread fixada na CPU 0):

| Módulo | Rust/C++, 1 thread | Rust/C++, todas |
|---|---:|---:|
| Mann-Kendall | 0,48 | 0,48 |
| TWDTW | 0,13 | 0,30 |
| SNIC | 0,86 | 0,81 |
| Whittaker | 0,91 | 0,73 |
| medoid | — | 0,61 |
| SMA | 0,26 | 0,16 |
| CODED | 0,32 | 0,32 |
| Tmask | 0,75 | 0,73 |
| BFAST Monitor | 0,79 | 0,76 |
| BFAST Lite | 0,47 | 0,46 |
| BFAST clássico | 0,27 | 0,12 |
| SOM em lote | 0,70 | 0,80 |
| LandTrendr | 0,73 | 0,61 |
| Fenologia | 0,59 | 0,49 |
| CCDC | 0,33 | 0,28 |
| warp bilinear | 0,89 | 0,84 |
| warp lanczos | 0,89 | 0,93 |
| warp average | 0,84 | 0,77 |

### Lições da fase 2

- **A ordem das somas do Eigen depende da ordem de armazenamento.** Uma soma
  sobre valores contíguos pelo tipo (coluna de matriz col-major, linha de
  row-major, vetor) vai em pacotes; uma soma com passo vai um a um. No GEMV, o
  kernel row-major acumula cada linha em duas pistas (colunas pares e ímpares),
  e o col-major acumula sequencialmente. O SMA usa matrizes row-major, e CCDC,
  CODED e BFAST Lite usam col-major: o port cobre as duas.
- **O tamanho de uma expressão do Eigen é o do operando esquerdo.**
  `X.row(r).dot(beta)` com `beta` um `Vector3d` não é desenrolado (o tamanho da
  linha é dinâmico): a soma é sequencial, não `x0 b0 + (x1 b1 + x2 b2)`.
- **`rowwise().squaredNorm()`** soma pares de linhas pelo caminho vetorizado (a
  primeira coluna, depois blocos de quatro somados em pares) e a última linha
  ímpar pelo escalar. Os dois só diferem a partir de 5 colunas.
- **O alocador pesa.** Módulos que alocam por pixel (QR, robust fit) ficavam até
  3× mais lentos no alocador do sistema, sobretudo no Windows e com muitas
  threads. O `zeit._rust` usa o mimalloc como alocador global, sem mudar nenhum
  valor.
- **O GEMM do Eigen depende da arquitetura.** No x86-64 sem FMA (as wheels de
  Linux e Windows) os blocos de linhas do kernel são de 4; no ARM64 (as de
  macOS), de 6, e lá o `pmadd` dos pacotes é fundido (`vfmaq_f64`, um
  arredondamento só). O próprio C++ dá bits diferentes entre as duas. O port
  segue a arquitetura em que roda (`eigen::MR`, `eigen::pmadd`); no ARM é o CI
  do macOS que confirma.
- **Mesma ordem, outro laço.** Os produtos (`X^T X`, matriz × vetor) ficaram
  rápidos quando várias cadeias de somas independentes passaram a avançar
  juntas, em blocos de 2 linhas × 4 colunas como o kernel do Eigen. Cada
  coeficiente mantém sua sequência exata de operações. Uma cadeia por vez
  ficava presa na latência da soma.
- **Erros dentro de laços paralelos.** No C++, um `x` não crescente no Whittaker
  lança exceção dentro da região OpenMP e derruba o processo (`std::terminate`).
  Em Rust vira `ValueError`.

### Fases 3 a 5 e o que o CI mostrou

BFAST clássico (com o STL), SOM e LandTrendr também dão os mesmos bits que o
C++. No i5-13600K, 1 thread: BFAST 0,27×, SOM em lote 0,76× (0,97× com 20
threads), LandTrendr 0,70× (0,61× com todas).

O CCDC (fase 7) também dá os mesmos bits: 0,33× com 1 thread e 0,27× com
todas no i5-13600K (~0,4× em séries de 20 anos a cada 8 dias). Os termos
harmônicos saem uma vez por lote e a matriz padronizada do lasso uma vez por
ajuste, para todas as bandas. O solver lasso é GPL-2.0, então o `zeit._rust`,
como o `zeit._core`, é distribuído sob GPL v2 (`THIRD_PARTY_NOTICES.md`). Onde
o C++ não tem comportamento definido (`num_c < 1`, `conseq_anom < 1`, arrays
de tamanho errado), o Rust levanta `ValueError` ou devolve zero segmentos.

O CI roda a suíte inteira nos dois núcleos e mede C++ × Rust nos três
sistemas (resumo do job Tests). No Linux o C++ é compilado pelo GCC, bem mais
rápido que o MSVC, e lá o Rust perdia em CODED (1,12×) e TWDTW (1,04×): os dois
foram otimizados (abaixo).

### Lições das fases 3 a 5

- **O Eigen bloqueia a profundidade do GEMM pelo tamanho do L1**, consultado
  com cpuid no x86 e fixo em 16 KB no ARM, onde não há consulta. No macOS isso
  divide históricos acima de ~200 observações; o único teste que falhou no
  macOS era esse. O Rust consulta o cache como o Eigen e divide igual
  (`eigen::l1_cache_size`, `eigen::gemm_kc`). Em CPUs híbridas (núcleos P e E
  com L1 diferentes), o próprio C++ dá resultados que dependem do núcleo em que
  rodou o primeiro produto, para profundidades entre ~400 e ~600.
- **Regiões paralelas curtas e repetidas** (o SOM em lote faz duas por
  iteração) custam caro no rayon, cujas threads dormem entre elas. O
  `parallel::team` mantém uma equipe ativa durante a chamada inteira, com uma
  barreira que gira um pouco antes de dormir, como o OpenMP. Girar demais é
  pior: com 19 threads e outros programas abertos, quem gira rouba CPU de quem
  trabalha.
- **Calcular uma vez o que é igual para todos os pixels.** As datas são as
  mesmas em todos os pixels de um lote: senos e cossenos do CODED e do Tmask
  (e a tabela de peso do TWDTW) saem uma vez por lote, com os mesmos valores.
  Dentro do laço do TWDTW, o tempo decorrido sai uma vez por célula (não uma
  vez por banda), e o caso de uma banda tem caminho próprio: 0,21× → 0,13× no
  Windows, 1 thread.
- **Funções do runtime C com estado global** (o `lgamma` da glibc escreve
  `signgam`) passam pela versão reentrante (`lgamma_r`), que dá o mesmo valor.

### Fases 6 a 8: fenologia, CCDC e warp

Os três dão os mesmos bits que o C++. No i5-13600K, tempo do Rust sobre o do
C++ com 1 thread e com todas: fenologia 0,60 e 0,48; CCDC 0,33 e 0,27; warp
0,83 a 0,94 e 0,80 a 0,97, conforme o método. Com isso, todo o C++ tem
versão em Rust; falta a etapa C.

O que não dá para igualar, porque o C++ não tem resultado definido:

- **Fenologia, série com um único pico:** o `check_GS_HeadTail` lê `pos[-1]`
  e `pos[1]` fora do vetor. O Rust faz o que o C++ quase certamente faz no
  Windows (não acrescenta vale); os testes evitam esse caso.
- **Warp, NaN ou −0/+0 numa janela de mediana ou quartil** (quando não são
  NoData): o C++ depende de como o `std::sort` de cada compilador os ordena.
  O Rust usa uma ordem total fixa.
- **Warp, float fora da faixa para inteiro:** o Rust imita o que os
  compiladores C++ geram no x86-64 (MSVC conferido; GCC pelo CI).
- **Arrays não contíguos na fenologia:** o binding em C++ os lê errado; o Rust
  os torna contíguos.

### Lições das fases 6 a 8

- **O LM da fenologia é o antigo `NonLinearOptimization`** (porta do MINPACK)
  com `NumericalDiff` em diferenças progressivas, passo h = 2⁻²⁶·|x|. A
  aplicação de Q usa os n refletores, não só os pivôs não nulos.
- **Zeros mudam a ordem das somas em pacote.** O termo do gradiente escalado é
  um vetor de m posições com zeros abaixo de n; o `stableNorm` tem de somar as
  m posições. `stableNorm` e `blueNorm`: um bloco de 4096, máximo e depois
  soma de quadrados em pacotes; o `blueNorm` é sequencial, com constantes
  potências de 2. Com NaN, o `fnorm` vira NaN, a norma do gradiente fica 0
  sob `std::max` e o ajuste "converge" no ponto inicial; o Rust reproduz.
- **Cholesky esparso:** a ordenação AMD depende só de n (cache por thread, como
  o `thread_local` do C++); a ordem das entradas em cada coluna permutada
  decide a ordem das atualizações; o solve triangular pula entradas nulas do
  lado direito (o que muda o sinal de zeros); o `setFromTriplets` soma
  duplicatas na ordem de inserção: diagonal = ((w + λ) + 4λ) + λ.
- **`minCoeff`/`maxCoeff`:** no SSE2, min/max com argumentos trocados (o
  acumulador fica no NaN); no NEON, `vminq`/`vmaxq` propagam o NaN.
- **O HANTS monta a matriz densa W = diag(w)** e multiplica por ela: igual a
  w_i·x_i, exceto que um valor não finito numa coluna a torna toda NaN. O Rust
  reproduz isso sem o produto n².
- **Jacobiano numérico mais barato com os mesmos bits:** cada coluna move um só
  parâmetro; os termos `exp`/`pow` que não dependem dele vêm do ponto base, e
  f(x) vem dos resíduos já calculados. Corta um terço das avaliações.
- **Matriz e lasso uma vez por ajuste, não por banda** (CCDC): o produto de
  duas colunas dá os mesmos bits em qualquer ordem dos fatores.
- **Os forks em worktree** às vezes nascem do `main`, não do branch de
  trabalho: conferir a base antes de começar.

### Como trabalhar

Durante a etapa B (até a 1.7.x, com os dois núcleos):

```bash
python setup.py build_ext --inplace          # C++ e Rust (release)
python setup.py build_rust --inplace          # só o Rust
cargo test --manifest-path rust/Cargo.toml -p zeit-core
python -m pytest tests/test_rust_parity.py    # C++ x Rust, bit a bit
ZEIT_BACKEND=rust python -m pytest tests/     # a suíte inteira no Rust
python benchmarks/bench_core.py --backend cpp rust --only snic twdtw mann_kendall
```

Desde a 2.0, só o Rust:

```bash
python setup.py build_rust --inplace --release
cargo test --manifest-path rust/Cargo.toml -p zeit-core --release
python -m pytest tests/
python benchmarks/bench_core.py --compare benchmarks/results/win-i5-13600k-rust.json:cpp new.json
```

## Etapa C: corte final (feita na 2.0.0)

Decidida em 2026-10-21, com todos os módulos em Rust e as mesmas saídas do C++:

- `src/`, `third_party/eigen` e a dependência de pybind11 saíram; o pacote
  constrói só a extensão Rust, com o setuptools-rust (o mesmo `build_wheels.yml`
  com cibuildwheel, que já gerava os wheels com Rust).
- A extensão passou a se chamar `zeit._core`, o nome do núcleo C++: os wrappers
  voltaram a importar `_core` direto, e `zeit/_backend.py` (a escolha por
  `ZEIT_BACKEND`) saiu, com os testes diferenciais (`tests/test_rust_parity.py`),
  que não têm mais contra o que comparar. As suítes de fidelidade contra o
  código dos autores (MATLAB, R, IDL, MiniSom, phenofit) continuam.
- `n_jobs <= 0` segue o `omp_get_max_threads` do OpenMP: o Rust lê
  `OMP_NUM_THREADS` (os tiles limitam cada processo com ela).
- `benchmarks/bench_core.py` mede só o núcleo; os tempos do C++ ficam nos JSON
  de `benchmarks/results/` para `--compare`.
- README, `docs/` (instalação, contribuição, desempenho, tutoriais),
  Dockerfiles, `THIRD_PARTY_NOTICES.md` e classifiers atualizados.

Ficaram para depois (não bloqueiam a 2.0): maturin e wheels abi3 (um wheel por
sistema em vez de um por Python), e `cp314t` (free-threaded), que exige
declarar o módulo como seguro sem o GIL no PyO3 e testar sob ele.

## Prós e contras (resumo)

**Prós:** segurança de memória (panic vira exceção em vez de segfault); corridas
de dados detectadas na compilação; multithread no macOS sem `libomp`; LLVM
também no Windows; GIL liberado em todo lugar; Cargo (testes, criterion,
proptest, fuzzing); sem Eigen vendorizado; abi3 e free-threaded; núcleo
reutilizável fora do Python (CLI estático, R via extendr, WASM no `zeit.plot`).

**Contras e riscos:** reescrever código validado contra os autores; a ordem das
reduções do Eigen não sai de graça (aceitar tolerância nesses caminhos e
revalidar contra os autores, esperando alguns empates trocados); ecossistema
numérico menos maduro (LM, LDLT e QR portados à mão); `warp` mais difícil de
acompanhar o GDAL; `unsafe` nos laços quentes; toolchain Rust no CI e para quem
compila do sdist; período com dois backends; custo de oportunidade.
