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
   passar.

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
| 3. Regressão e decomposição | `stl_decompose`, `bfast`, `bfast_monitor`, `bfast_lite` | em andamento: `bfast_monitor` e `bfast_lite` feitos |
| 4. SOM | `som` | |
| 5. LandTrendr | `landtrendr` | |
| 6. Fenologia | `phenology_*` (port do LM do Eigen) | |
| 7. CCDC | `ccdc` (GLMnet float32) | |
| 8. Warp | `warp` (callback do transformer do GDAL) | |

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

Tempos em turnos no mesmo processo (`benchmarks/results/win-i5-13600k-rust.json`,
todos os módulos convertidos até aqui; com todas as threads a máquina tinha
carga de fundo, e os dois lados ficaram ~2× mais lentos que em outras medições):

| Módulo | Rust/C++, 1 thread | Rust/C++, todas |
|---|---:|---:|
| Mann-Kendall | 0,49 | 0,49 |
| TWDTW | 0,21 | 0,26 |
| SNIC | 0,85 | 0,88 |
| Whittaker | 0,93 | 0,83 |
| medoid | — | 0,64 |
| SMA | 0,23 | 0,28 |
| CODED | 0,76 | 0,71 |
| Tmask | 0,87 | 1,09 (a investigar: 0,94 numa medição sem carga) |
| BFAST Monitor | 0,77 | 0,79 |
| BFAST Lite | 0,48 | 0,52 |

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

### Como trabalhar

```bash
python setup.py build_ext --inplace          # C++ e Rust (release)
python setup.py build_rust --inplace          # só o Rust
cargo test --manifest-path rust/Cargo.toml -p zeit-core
python -m pytest tests/test_rust_parity.py    # C++ x Rust, bit a bit
ZEIT_BACKEND=rust python -m pytest tests/     # a suíte inteira no Rust
python benchmarks/bench_core.py --backend cpp rust --only snic twdtw mann_kendall
```

## Etapa C: corte final

- Remover `src/`, `third_party/eigen` e a dependência de pybind11.
- Passar para maturin; wheels abi3 (confirmar compatibilidade do rust-numpy).
- Habilitar `cp314t` (free-threaded) no CI.
- Atualizar README, `docs/benchmarks/performance.md` e os classifiers.

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
