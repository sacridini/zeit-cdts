# Ideias para o zeit

Ideias de evolução da API, divididas em fases. Cada fase termina com testes passando,
versão nova (minor) e commit. Uma fase concluída é marcada como **Feito** (versão).

## Objetivo: carregar, rodar, exportar

Hoje um fluxo simples (ler um stack anual de NDVI, rodar o LandTrendr e exportar a maior
perda) pede rasterio, montar `years` à mão, carregar `crs`/`transform` até o fim e saber
qual das várias funções de LandTrendr usar (`run_landtrendr`, `run_landtrendr_batch`,
`run_landtrendr_array`, `run_landtrendr_image`, `DataArray.zeit.run_landtrendr`). A meta
é que a georreferência **e o tempo** viajem junto com o dado, como no landschaft, e que
cada algoritmo seja uma função só, que entende o formato da entrada:

```python
import zeit

ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")    # (time, y, x), anos lidos de "yr1985"...
lt = zeit.landtrendr(ndvi)                                # Dataset georreferenciado
events = zeit.extract_events(lt, event_type="loss")
zeit.save_raster(events, "lt_rondonia")                   # um GeoTIFF por métrica
```

## Convenção do cubo

Todo o zeit troca dados no mesmo formato:

- `xarray.DataArray` com dims `(time, y, x)` (um índice) ou `(time, band, y, x)` (várias
  bandas espectrais, ex.: entrada do CCDC); `(y, x)` para um mapa só.
- `time` é uma coordenada `datetime64`. Cada algoritmo converte para o que precisa
  (anos inteiros no LandTrendr, dias ordinais no CCDC, ano fracionário no BFAST).
- Georreferência via `.rio` (CRS, transform, nodata), com coordenadas `x`/`y`.
- O dtype original é mantido (`masked="auto"`): um NDVI Int16 escalado por 10000
  continua Int16; floats têm o nodata como NaN.
- Quando o tempo não pode ser inferido, a dimensão das bandas se chama `band` e os
  algoritmos pedem `years=`/`dates=` com uma mensagem clara.

## Fase 1: `load_raster` — **Feito** (0.27.0)

`zeit.load_raster(source, *, dates=None, start_year=None, band=None, chunks=None,
clip=None, masked="auto", pattern=None, date_format=None, validate=None)` devolve um
cubo georreferenciado. Substitui o `zeit.io.load_raster` atual, que devolve
`(numpy, profile)` (quebra de compatibilidade aceita: zeit ainda é 0.x).

Entradas:

| Entrada | Exemplo | De onde vem o tempo |
|---|---|---|
| GeoTIFF (ou qualquer raster GDAL) multibanda, uma banda por data | stacks do GEE/LT-GEE | `dates=`/`start_year=`, depois a metadata gravada pelo `save_raster` (`ZEIT_TIME`), depois descrição das bandas (`yr1985`, `1985`, `2020-01-15`, `20200115`, `2020_01_15`), depois o `_dates.csv` antigo |
| Pasta, glob ou lista de rasters de uma data cada | ARD local | data no nome do arquivo (padrões comuns, ou `pattern=` com grupos `date`/`band` como no `build_local_cube`) |
| Descrições `data_banda` (`2020-01-15_red`) | export 4D do `save_raster` | vira `(time, band, y, x)` |
| Zarr (`.zarr`) e NetCDF (`.nc`) | `to_zarr_optimized` | coordenada `time` gravada |
| `DataArray`/`Dataset` em memória | `build_time_series`, composites | normaliza dims, georreferência e tempo |
| numpy + `dates`/`start_year` (+ `like=` para georreferenciar) | dados já lidos | dos argumentos |

- `chunks=None` carrega em memória; `chunks="auto"` (ou dict) devolve um cubo dask lazy.
- `clip=` lê só a janela de uma região (bounds ou geometrias).
- `validate="landtrendr"|"ccdc"` mantém os avisos de escala/tamanho do `raster_check`.
- `build_local_cube` passa a ser um atalho para `load_raster(pasta, pattern=...)`.
- Os algoritmos chamam o próprio `load_raster` para aceitar qualquer uma dessas entradas
  (para `DataArray`/`Dataset` em memória ele só normaliza, sem copiar nem carregar).
- Implementado em `zeit/_load.py` (leitura) e `zeit/_time.py` (datas: rótulos, nomes de
  arquivo e conversões para anos, dias ordinais e ano fracionário).
- `load_raster` e `save_raster` exportados no `zeit` (hoje só em `zeit.io`).

## Fase 2: `save_raster` simétrico — **Feito** (0.28.0)

`zeit.save_raster(data, path, *, like=None, crs=None, transform=None, nodata=None,
dtype=None, band_names=None, compress="deflate", driver=None)` devolve o `Path` gravado.

- Aceita `DataArray`, numpy/dask (georreferenciado por `like=`, um raster ou caminho),
  `xr.Dataset` e `dict` de mapas (ex.: o resultado de `extract_events`).
- `Dataset`/`dict`: caminho sem extensão vira pasta com um GeoTIFF por variável;
  caminho `.tif` vira um multibanda com os nomes nas descrições das bandas.
- `time` vai para as descrições das bandas (ISO, `1985-01-01`) e para a metadata
  (`ZEIT_TIME`), para o `load_raster` ler de volta sem arquivo extra. O `_dates.csv`
  deixa de ser escrito por padrão.
- `(time, band, y, x)` vira bandas `data_banda`.
- NaN → nodata em floats; `bool` → `uint8`; inteiros de 64 bits reduzidos ao menor tipo
  que cabe; tiled + predictor; BigTIFF quando precisar; dask escrito bloco a bloco.
- Formato pela extensão (`.tif`, `.nc`, `.zarr`, ...) ou `driver` (`"COG"`).
- Sem `print`; sem georreferência, um aviso em vez de uma matriz identidade silenciosa.
- `reference_cube=` continua aceito como nome antigo de `like=` (DeprecationWarning).
- Implementado em `zeit/_save.py`. O `get_georef` passou a pôr a origem no canto da
  célula (as coordenadas `x`/`y` são centros; antes ficava meio pixel deslocada).

## Fase 3: `zeit.landtrendr` — **Feito** (0.29.0)

Uma função só: `zeit.landtrendr(data, *, years=None, direction="loss", max_segments=6,
... , band=None, chunks=None, n_jobs=-1)`.

- Entende a entrada: série de um pixel (lista/numpy 1D/`Series`), numpy 3D + `years`,
  cubo `DataArray` (em memória ou dask), caminho de arquivo (lido em blocos), `Dataset`
  ou cubo 4D com `band=`.
- Anos tirados de `time` quando houver; `years=` só para numpy ou cubos sem tempo.
- `direction="loss"|"gain"` substitui o `modifier` (±1) e fica gravado no resultado.
- Resultado: `xr.Dataset` georreferenciado com `vertex_year`/`vertex_value`
  `(vertex, y, x)`, `n_vertices` e `rmse` `(y, x)`; `fitted=True` acrescenta a série
  ajustada `(time, y, x)`.
- `zeit.extract_events` aceita esse `Dataset` (usa o `rmse` para o DSNR e a `direction`
  como `event_type` padrão) e devolve um `Dataset` georreferenciado; numpy continua
  aceito.
- O módulo `zeit/landtrendr.py` vira `zeit/_landtrendr.py` (o nome `zeit.landtrendr`
  passa a ser a função). `run_landtrendr`, `run_landtrendr_batch` e
  `run_landtrendr_array` ficam como blocos internos; `run_landtrendr_image` e o
  accessor `run_landtrendr` são substituídos (`DataArray.zeit.landtrendr()` delega para a
  função nova). CLI, exemplos e docs passam a usar `zeit.landtrendr`.
- `nodata="auto"`: o NoData do raster; em dados inteiros sem NoData, `0` (como o Earth
  Engine exporta pixels mascarados; o stack de Rondônia tem 10 mil pixels com anos
  zerados que virariam perdas falsas). `nodata=None` usa só NaN.
- `extract_events` deixou de contar um segmento plano (magnitude 0) como evento, e o
  numba passou a compilar uma vez por processo (antes, a cada chamada).
- `save_raster` de um Dataset/dict grava todas as variáveis juntas, faixa a faixa: com
  um resultado lazy, o LandTrendr roda uma vez só para todos os mapas.
- Implementado em `zeit/_lt.py`; o resultado é idêntico ao do `run_landtrendr_array`
  (testado pixel a pixel no stack de Rondônia).

## Fase 4: `zeit.ccdc` — **Feito** (0.30.0)

Mesmo padrão para o CCDC/COLD: `zeit.ccdc(data, *, qa=None, dates=None, bands=None, ...)`
aceita o cubo `(time, band, y, x)` (em memória ou dask), um raster com bandas
`data_banda`, um raster intercalado por data com `dates`/`bands`, numpy com `dates` ou um
pixel (`DataFrame` indexado por data, ou `DataArray (time, band)`), e tira as datas
ordinais de `time`.

- `qa` pode ser o nome de uma banda do cubo (sai das bandas espectrais), um cubo
  `(time, y, x)` ou `None`; `detection_bands`, `tmask_bands` e `thermal_band` aceitam
  nomes de banda.
- Resultado: `Dataset` com `t_start`/`t_end`/`t_break` (datas, `NaT` além do último
  segmento), `n_segments`, `rmse (segment, band)` e `coefs (segment, band, coef)`.
- `predict_synthetic_image(segments, "2020-07-01")` devolve a imagem `(band, y, x)`
  georreferenciada (também lazy); a chamada numpy antiga continua aceita.
- `save_raster` grava datas como ano decimal e achata dimensões extras em bandas
  nomeadas (`1_blue_a0`); `load_raster` mantém a ordem das bandas de um stack
  `data_banda` (antes o `unstack` ordenava alfabeticamente).
- `run_ccdc_array`/`run_ccdc_image`/accessor `run_ccdc` saíram da API pública; o módulo
  `zeit/ccdc.py` virou `zeit/_ccdc.py`. A CLI `zeit ccdc` lê as datas dos nomes das
  bandas ou de `--dates-file` (o fallback de datas fictícias de 16 dias saiu).
- Implementado em `zeit/_ccdc_api.py`; idêntico ao motor de um pixel, segmento a segmento.

## Fase 5: BFAST, Mann-Kendall e fenologia — **Feito** (0.31.0)

`zeit.bfast_monitor(data, monitor_start)`, `zeit.bfast_lite`, `zeit.bfast`,
`zeit.mann_kendall` e `zeit.phenology` no mesmo padrão: qualquer entrada, tempo tirado do
cubo e saída `Dataset` georreferenciado com uma variável por métrica.

- `start_time` e `frequency` (o eixo `ts` do R) saem das datas: a frequência do
  espaçamento mediano (23 para 16 dias, 12 mensal, 1 anual) e o início da primeira data;
  dados os dois, a série não precisa de datas. `monitor_start` aceita data ou ano decimal.
- Fenologia: `curve="beck"`, `method="threshold"` por nome; a numeração de dias e o
  `base_year` saem das datas (antes o usuário montava `dates_doy` à mão);
  `max_seasons` padrão = número de anos. Corrigido: com `annual=True`, o LOS (uma
  duração) era decodificado como data e caía sempre no primeiro ano; agora vai para o ano
  do POP, como R2/RMSE.
- `nodata="auto"` como no LandTrendr e no CCDC.
- No `bfast`, a métrica `time` virou `break_time` (colidia com a coordenada `time`).
- `run_*_image` e os métodos `run_*` do accessor saíram; `zeit/bfast.py` e
  `zeit/phenology.py` viraram `_bfast.py` e `_phenology.py`; o `raster.py` ficou só com
  os motores numpy do LandTrendr e do CCDC. A CLI grava o mesmo `<prefix>.tif`, com
  `--start-time`/`--frequency` opcionais.
- `save_raster` nomeia as bandas pela coordenada da dimensão (anos, vértices...).
- Implementado em `zeit/_series_api.py`; resultados idênticos aos motores.

## Fase 6: arrumação final — **Feito** (0.32.0)

- `import zeit` de ~19 s para ~2 s: `zeit.ai` (torch + transformers), os construtores de
  cubo STAC (pystac-client, stackstac) e o classificador CCDC (sklearn) carregam no
  primeiro uso, pelo `__getattr__` do módulo (`tests/test_package.py` garante que o
  torch e o stackstac não voltem ao import).
- Página "Upgrading to the one-function API" (`docs/getting-started/migrating.md`) com a
  tabela do que saiu e do que substitui, e quickstart carregar, rodar, exportar no README.
- README, quickstart, conceitos, tutoriais, exemplos e CLI já foram migrados em cada fase;
  `sync_api --check` e `mkdocs build --strict` limpos.

## Fase 7: `zeit.plot` — **Feito** (0.33.0–0.37.0)

Uma função para ver qualquer coisa do zeit: o cubo carregado, um resultado (`Dataset` do
LandTrendr, CCDC, BFAST...), um mapa ou a série de um pixel. O foco são rasters, e
principalmente séries temporais densas e grandes (centenas de datas, milhares de pixels de
lado), que precisam rodar fluidas. Vetores entram como sobreposição.

```python
zeit.plot(ndvi)                          # cubo (time, y, x): slider no tempo, legenda contínua
zeit.plot(loss.yod)                      # mapa de anos: paleta sequencial com anos na legenda
zeit.plot(classes, basemap="satellite")  # categórico, sobre imagem de satélite
zeit.plot(ndvi, window=True)             # janela separada, fora do Jupyter
zeit.plot(ndvi, vector="talhoes.gpkg")   # contornos por cima
```

### O que ela precisa fazer

- **Entender o dado** (a mesma ideia do resto da API): `DataArray`/`Dataset`, numpy,
  caminho de arquivo (via `load_raster(chunks="auto")`), resultado de algoritmo. Um `time`
  vira slider; um `band` com 3 bandas nomeadas (red/green/blue) vira RGB; um `Dataset` vira
  um seletor de variáveis.
- **Escolher as cores sozinha:**
  - inteiros com poucos valores, `bool` ou classes → categórico, cor por valor e legenda
    com os rótulos (do colormap embutido no GeoTIFF, quando houver);
  - floats → contínuo, limites robustos (percentis 2–98, calculados numa amostra e não
    no cubo inteiro);
  - valores em torno de 0 (slope, magnitude, diferença) → divergente centrado em 0;
  - datas e anos (`yod`, `t_break`) → sequencial com anos na legenda;
  - NaN/NoData transparentes. Tudo sobrescrevível (`cmap=`, `vmin=`, `vmax=`, `kind=`).
- **Slider rápido** no tempo, com play/pause e passo ajustável; datas no título.
- **Legenda/colorbar** sempre, e inspeção de valor ao passar o mouse.
- **Basemap opcional** com imagem de satélite (Esri World Imagery) ou mapa (OSM/Carto),
  reprojetado para o CRS do raster, com transparência ajustável na camada de cima.
- **Clicar num pixel e ver a série dele** num painel ligado, com o ajuste do algoritmo
  quando houver (vértices do LandTrendr, modelo do CCDC, quebras do BFAST). É o uso mais
  comum na hora de calibrar parâmetros.
- **Jupyter e fora dele:** widget no notebook, janela própria em script (`window=True`,
  padrão fora do Jupyter), e figura estática (PNG/PDF) para relatórios.

### Tecnologias avaliadas

| Ferramenta | Pontos fortes | Limites para o zeit |
| :--- | :--- | :--- |
| **fastplotlib** (pygfx/WGPU) | Renderiza na GPU (Vulkan/DX12/Metal); `ImageWidget` com sliders de tempo para pilhas de imagens, aceita arrays "array-like"; roda no Jupyter (jupyter_rfb) e em janela Qt/glfw com o mesmo código | Sem noção de mapa (CRS, basemap); desempenho com arrays lazy (dask/zarr) no slider precisa ser medido; ainda em versão 0.x |
| **napari** | Visualizador n-D maduro; usa dask/zarr para carregar só o que aparece na tela; já usado com centenas de cenas Sentinel; pirâmides multiescala | Janela Qt (no notebook é limitado); dependência pesada; sem basemap |
| **HoloViz** (hvplot + datashader + Panel/GeoViews) | Basemaps prontos, CRS, `panel serve`/navegador fora do Jupyter; datashader agrega rasters enormes | Cada passo do slider faz ida e volta ao servidor e re-sombreamento (relatos de ~7 s por raster grande): lento para séries densas |
| **lonboard** (deck.gl) | GPU no navegador, excelente para vetores grandes | Focado em vetores; raster só como bitmap; só Jupyter |
| **leafmap / localtileserver** | Basemaps e COGs em mapa web | Um servidor de tiles por imagem; lento para passar por muitas datas |
| **matplotlib** | Universal, figuras estáticas de publicação | CPU; lento para animar séries grandes |

### Resultados do 7a (prova de conceito, RTX 3060, Edge, JupyterLab local)

Protótipo e scripts em `scratch/plot7a/` (fora do git). Três caminhos comparados:
**A** fastplotlib (renderiza na GPU do kernel e manda JPEG por quadro), **B** widget
próprio (anywidget + WebGL2: o quadro vai cru em `uint8` e a paleta é aplicada na GPU do
navegador) e **C** matplotlib (o que o ipympl faz).

| Medida | A. fastplotlib | B. anywidget + WebGL2 | C. matplotlib |
| :--- | :--- | :--- | :--- |
| Custo no servidor por quadro | 8–9 ms (render + JPEG) | 1,5–1,8 ms (quantizar) | 106–133 ms |
| Bytes por quadro | 70–230 KB (JPEG q80) | 300–450 KB cru (190–290 KB com zlib) | 265–490 KB (PNG) |
| Quadros/s no slider, notebook local | 60 (limite padrão do rendercanvas) | **144** (o teto do navegador) com quadros pré-carregados; ~200/s pedindo ao kernel | ~8 (limite do servidor) |
| Desenho no navegador | — | 0,1 ms por quadro | — |
| Pré-carga Rondônia (40 × 557×562) | — | 12,5 MB em 0,18 s | — |
| Pré-carga cubo denso (240 × 682×682) | — | 112 MB em 1,4 s | — |

Leitura dos dados (bomba de quadros): 0,1–7 ms por data na resolução da tela (TIF e Zarr);
o único custo alto é a primeira leitura de um TIF intercalado por pixel (~0,75 s para o
cubo inteiro, uma vez). Não é o gargalo.

A emulação de rede do Chrome não limita websockets de forma realista, então o caso de
notebook remoto (JupyterHub, Colab) ficou como estimativa: A precisa de uma ida e volta ao
kernel por quadro, com no máximo 2 quadros em trânsito (fps ≤ 2/latência: 100 ms →
~20 fps) e ~6–14 MB/s para 60 fps; B paga a pré-carga uma vez (12,5 MB: ~5 s a 20 Mbps,
~1 s a 100 Mbps) e depois não usa mais a rede.

### Decisão: caminho B como motor do `zeit.plot`

- **Depois de carregado, o slider não depende do kernel nem da rede:** roda no limite do
  navegador, e continua funcionando enquanto o kernel executa outra célula (com A, o
  viewer congela durante qualquer célula, porque cada quadro é renderizado em Python).
- **Não exige GPU no servidor:** quem desenha é a GPU do notebook do usuário, via WebGL.
  Kernels em JupyterHub/servidores raramente têm GPU; o fastplotlib precisa dela (ou cai
  em renderização por software) na máquina do kernel.
- **Dado exato, sem JPEG:** mapas categóricos ficam nítidos, e a cor e o valor sob o mouse
  vêm do próprio dado.
- **Uma implementação para todo lugar:** anywidget roda em JupyterLab, Notebook 7, VS Code
  e Colab; fora do notebook, o mesmo HTML/JS abre numa janela (pywebview) ou aba do
  navegador servida localmente. Leve: anywidget é a única dependência nova obrigatória.
- **Basemap direto no navegador:** os tiles XYZ (Esri World Imagery, OSM) são baixados
  pelo próprio navegador, como num Leaflet; com basemap, os quadros vão reprojetados para
  Web Mercator na resolução da tela (barato no servidor).
- **Custos e mitigação:** memória do navegador e pré-carga em conexões lentas. Pré-carregar
  progressivamente (o primeiro quadro na hora, os demais em segundo plano, primeiro em
  resolução reduzida), comprimir os quadros, e para cubos grandes manter no navegador
  uma janela de datas em torno do slider (LRU), pedindo o resto ao kernel (~5 ms por
  quadro, medido).
- **Precisão:** `uint8` dá 254 níveis para a cor (suficiente para ver); o valor exato sob o
  mouse vem do kernel sob demanda, ou os quadros vão em `uint16` quando pedido.
- fastplotlib fica de fora do núcleo; pode voltar como backend opcional de janela nativa
  se a medição da janela pywebview não for boa.
- matplotlib continua para figuras estáticas (`static=True`, PNG/PDF).

### Próximos passos (7b em diante)

1. 7b: estático com matplotlib + a inferência de cores e legendas (compartilhada com o
   viewer interativo). **Feito** (0.33.0): `zeit/_plot/` (`_data.Frames` vê qualquer
   entrada como quadros 2-D, toda dimensão além de `y`/`x` vira eixo de quadros com
   rótulos como os do `save_raster`; `_style.infer_style` escolhe contínuo, divergente,
   categórico, anos ou RGB a partir de uma amostra; `_static` desenha mapa, grade de datas
   ou série). `load_raster` passou a nomear o cubo pelo arquivo (a paleta de vegetação vem
   do NDVI/NBR no nome) e fecha os arquivos lazy na saída do Python.
2. 7c: viewer interativo (o protótipo B evoluído): pan/zoom, colorbar, valor sob o mouse,
   play/pause, pré-carga progressiva, seletor de variável para Datasets, RGB. **Feito**
   (0.34.0): `_session.Session` responde aos pedidos do navegador (`meta`, `frames`,
   `detail`, `select`) sem saber o transporte; `_widget` leva os pedidos pelas mensagens do
   anywidget; `viewer.js` (WebGL2, texturas inteiras + LUT) pré-carrega a partir do quadro
   atual para os dois lados (2 pedidos em voo, LRU de 400 MB, quadros comprimidos com
   deflate), busca o recorte em resolução maior ao dar zoom com o viewer parado, e mostra
   colorbar ou legenda, valor e coordenadas sob o mouse, play com fps ajustável e teclado
   (espaço, setas). `zeit.plot` abre o viewer no notebook e a figura fora dele.
3. 7d: clicar num pixel e ver a série com o ajuste do algoritmo. **Feito** (0.35.0):
   `zeit.plot(cube, fit=resultado)`; `_fit.overlays` traduz cada resultado em linhas,
   marcas e faixas (vértices do LandTrendr, modelos harmônicos e quebras do CCDC, evento do
   `extract_events`, quebras do BFAST, reta de Theil-Sen do Mann-Kendall), casando o
   resultado com o dado pelas coordenadas. No viewer, o clique marca o pixel e desenha o
   gráfico SVG com a data atual e clique no gráfico para pular de data; em estático,
   `zeit.plot(cube, fit=lt, pixel=(x, y), static=True)`.
4. 7e: basemap e vetores; avaliar MapLibre GL (camada WebGL própria sobre o mapa) contra
   pan/zoom e tiles feitos à mão. **Feito** (0.36.0), à mão (sem JS externo, funciona
   offline sem o basemap): o navegador baixa os tiles XYZ e os posiciona interpolando em
   duas grades de controle célula ↔ lon/lat calculadas no Python (qualquer CRS, sem
   reprojetar os quadros); `basemap="satellite"|"osm"|"light"|"dark"|"topo"`, nome do
   xyzservices ou URL; `vector=` (arquivo, GeoDataFrame, shapely) desenhado por cima com
   halo; slider de opacidade e atribuição. Estático: tiles baixados (cache em
   `~/.cache/zeit/tiles`), mosaico reprojetado para o CRS dos dados. Os tiles da CartoDB
   passaram a exigir chave: `light`/`dark` usam os fundos cinza da Esri.
5. 7f: janela fora do notebook (pywebview ou navegador), testes (Playwright com Edge/Chromium
   em modo headless, como no 7a) e docs. **Feito** (0.37.0): `_window.show_window` serve o
   mesmo viewer num servidor HTTP em 127.0.0.1 com token aleatório (pywebview num processo
   filho quando instalado, senão o navegador); num script bloqueia até a janela fechar,
   como `plt.show()`. `zeit.plot` escolhe sozinho: widget no notebook, janela num script ou
   terminal, figura com `save=` ou sem tela (CI, testes, servidor sem display,
   `ZEIT_PLOT=static`). Teste de ponta a ponta no Edge headless. Medido no cubo denso
   (240 × 2048², Zarr lazy): primeiro quadro em ~1 s, as 240 datas em 8,5 s, play a 60 fps
   exatos. A leitura dask fica em uma computação por vez (computações simultâneas se
   atrapalham: pré-carregar lotes em threads piorou de 9 para 15 s); a codificação e a
   compressão dos quadros rodam em paralelo. Docs: `docs/api/plot.md` e o tutorial
   `docs/tutorials/plotting.md`.

### Dependências

`anywidget` (obrigatória para o interativo, leve) e `matplotlib` (estático); `pywebview`
opcional para a janela nativa; `xyzservices` para nomear os basemaps. Tudo no extra
`pip install zeit-cdts[plot]`, sem pesar no `import zeit`.

## Fase 8: grade de referência, FTV e resultados que se salvam — **Feito** (0.38.0–0.40.0)

Os três itens que estavam em "Para depois". Vêm antes da Fase 9 porque o `like=` é o que
destrava combinar fontes (Landsat com Sentinel, um cubo com uma máscara de outra grade, cenas
de órbitas diferentes numa pasta).

### 8a: `load_raster(..., like=)` com o warp do landschaft em C++ — **Feito** (0.38.0)

`zeit.load_raster(source, *, like=None, resampling="auto", ...)` devolve o cubo **na grade
de `like`**: o mesmo CRS, transform, tamanho e as mesmas coordenadas `x`/`y` (bit a bit,
para o xarray alinhar sem reindexar). `time` e `band` não mudam. Uma função só também para
o que já está carregado: `zeit.load_raster(cube, like=ref)` faz o papel do `match_grid`.

- `like`: `DataArray`/`Dataset` ou caminho de raster; só a grade é lida (nada de dados).
  Com numpy continua como hoje (georreferencia, tamanho igual obrigatório): nos dois casos
  o resultado está na grade de `like`.
- `resampling="auto"`: nearest para inteiros e `bool` (classes, QA, índices escalados
  Int16: o valor é um que foi observado, e bits de QA nunca se misturam), bilinear para
  floats; bandas de QA pelo nome (`qa`, `fmask`, `scl`, `pixel_qa`...) sempre nearest. Ou
  um método do GDAL (`"nearest"`, `"bilinear"`, `"cubic"`, `"average"`, `"mode"`...).
- **Máscara por data, não a unificada do GDAL.** O kernel do landschaft segue o GDAL:
  uma célula é válida se *alguma* banda não for NoData. Numa série temporal cada data tem
  as próprias nuvens, então uma célula NaN numa data e válida nas outras entraria no
  bilinear/average dessa data e espalharia o NaN (o `average` de uma célula grossa com um
  pixel nublado daria NaN em vez da média dos limpos). Aqui cada plano `(time, band)` tem
  a própria máscara; o resultado de cada data é o do GDAL warpando aquela data sozinha
  (é com isso que os testes comparam).
- **Coordenadas uma vez só:** a grade de coordenadas de origem (nós exatos a cada 16
  células, refinados onde a interpolação erra mais que a tolerância) é calculada uma vez
  por par de grades e serve para todas as datas e bandas. Numa série de 240 datas o custo
  da transformação (PROJ) é pago uma vez, não 240.
- **Sem warp quando não precisa:** mesma grade → nada; mesmo CRS, mesma resolução e
  deslocamento inteiro de células → recorte/preenchimento por índices (o caso comum de
  "cortar na extensão da referência").
- **Lazy e por janela:** com `chunks=`, o resultado é dask em blocos fixos ancorados na
  origem de `like`, cada um lendo só a janela de origem que vê (mais o raio do filtro);
  `time` mantém os chunks. De um arquivo grande, só a região de `like` é lida. Em memória,
  lazy, de arquivo e com qualquer número de threads, os mesmos bits.
- **Pasta de cenas com grades diferentes:** `load_raster("cenas/", like="ref.tif")`
  warpa cada arquivo para a grade de `like` antes de empilhar (hoje um arquivo fora da
  grade do primeiro dá erro, `_load.py`).
- `clip=` junto com `like=`: o recorte é aplicado na grade de `like`.

Implementação: o kernel do landschaft 1.35.0 (o porte do `gdalwarpkernel.cpp` do GDAL
3.12, validado bit a bit contra o GDAL no landschaft) em `src/warp.cpp`/`.hpp`, com os
bindings em `src/warp_python.cpp` (`_core.warp.warp`, `coords`, `lattice_checks`); a parte
de rasters do `_reproject.py` (grade de coordenadas, escala dos kernels, tarefas lazy) em
`zeit/_warp.py`, e o transformador do GDAL do rasterio por ctypes (pyproj de reserva) em
`zeit/_gdaltransform.py`. Vetores e o método `sum` (que no landschaft roda o próprio GDAL)
ficaram de fora. Mudanças em relação ao landschaft:

- `WarpSpec::per_band`: cada banda com a própria fonte e destino de uma banda só (máscara e
  AvoidNoData próprios), usando a mesma linha de coordenadas; o zeit sempre usa.
- O número de threads vai em cada chamada (`n_threads`), e o OpenMP da thread que chama
  volta ao que era depois: o `omp_set_num_threads` não pode mudar o padrão
  (`omp_get_max_threads() - 1`) dos outros motores.
- Tarefas lazy divididas também nos grupos de datas (os chunks do cubo), não só em linhas e
  colunas: um cubo de 240 datas não vira uma tarefa de 240 × 1024².
- O recorte por índices (mesmo CRS e células, deslocamento inteiro) não passa pelo kernel;
  o resultado é o do GDAL, que também toma o nearest numa translação de células inteiras.
- QA pelo nome sempre nearest, separado das outras bandas e reordenado no fim.

Licença: o código é do mesmo autor e entra no zeit como GPL-2.0-or-later; o GDAL (MIT)
está no `THIRD_PARTY_NOTICES.md`.

Testes (`tests/test_warp.py`, 74): cada data igual ao GDAL warpando aquela data sozinha
(`tolerance=0`, os 13 métodos; uint8/uint16/int16/int32/float64, com e sem NoData; NaN em
lugares diferentes por data); um `average` cuja célula grossa tem um pixel nublado numa
data só; a tolerância padrão perto da exata; sem a DLL do GDAL (pyproj); memória = lazy =
arquivo = qualquer número de threads; o recorte igual ao GDAL; `load_raster` com `like=`
DataArray/caminho, `clip=`, `masked=`, pasta com uma cena em EPSG:4326 entre cenas em UTM;
bandas de QA.

Medido (cubo de 240 × 1000 × 1000 float32 com nuvens por data, UTM 33N → EPSG:3035,
i5-13600K, 19 threads), contra o `rioxarray.reproject_match`: nearest 0,46 s contra 2,4 s,
bilinear 0,70 s contra 22,6 s, average 0,90 s contra 6,4 s. O rioxarray deixa NaN a mais
em 0,13% das células no bilinear (as bordas das nuvens, pela máscara unificada).

### 8b: `zeit.landtrendr(..., ftv=[...])` — **Feito** (0.39.0)

Fitted-to-vertices como no LT-GEE: segmenta uma banda (`band=`) e ajusta as outras com os
mesmos anos de vértice. `ftv=["nbr", "tcw"]` (nomes de banda do cubo `(time, band, y, x)`
ou variáveis do `Dataset`) acrescenta ao resultado `ftv_<banda>` `(time, y, x)` com a série
ajustada e `vertex_value_<banda>` `(vertex, y, x)` com o valor ajustado em cada ano de
vértice.

- Porte do `apply_fitted_trajectory_v1.pro` + `ftv_v1.pro` (LLR-LandTrendr) em
  `src/landtrendr.cpp` (`fit_to_vertices`, `fit_to_vertices_batch`), reaproveitando o
  `desawtooth`, o `find_best_trace` (`fit_piecewise_sequential`) e o `fill_from_vertices`
  (`interpolate_fit`) do motor. A banda é desawtoothed com `spike_threshold` e não é
  invertida pelo `direction` (o original deixa o `modifier` comentado).
- Um ano de vértice sem observação na banda (nuvem só nela) vai para a última observação
  antes dele, ou para a primeira depois quando aquela já é vértice; primeiro/último anos
  ausentes ganham vértices planos, e o de menor ângulo sai se passar da contagem, como no
  `ftv_v1`.
- As bandas entram empilhadas com a segmentada no mesmo bloco: lazy, o LandTrendr e os
  FTV rodam juntos, uma vez por bloco.
- Paridade (`tests/test_landtrendr_ftv.py`): 10 casos rodados no IDL original sob o GDL
  (com o `regress.pro` sobre `LINFIT` do harness da paridade do LandTrendr): vértices
  observados, desawtooth, vértice nublado para trás e para frente, pontas ausentes, dois
  vértices, duas perturbações. Valores dos vértices iguais na precisão do float32 do
  original e a série truncada (o original guarda o `yfit` num array inteiro) igual em
  todos os anos.
- O `apply_vertices` (que só interpola os valores brutos) ficou, com a documentação
  apontando para o `ftv=`.

### 8c: resultados com `.save()` e `.plot()` — **Feito** (0.40.0)

`.zeit.save(caminho, **kwargs)` e `.zeit.plot(**kwargs)` em qualquer `DataArray` e, por um
accessor novo (`ZeitDatasetAccessor`), em qualquer `Dataset`, delegando para
`save_raster` e `zeit.plot`, como os `Saveable` do landschaft:
`zeit.landtrendr(ndvi).zeit.save("lt")`, `lt.zeit.plot()`, `loss.yod.zeit.plot()`. O
`sync_api` nomeia os métodos do accessor de Dataset como `Dataset.zeit.<método>`.

## Fase 9: o resto da API no padrão do cubo — **Feito** (0.41.0–0.42.0)

As Fases 3–5 levaram os algoritmos de mudança para "uma função, qualquer entrada, saída
georreferenciada". O que sobrou ainda recebe numpy cru, datas montadas à mão ou caminhos de
entrada e saída:

| Hoje | Depois |
| :--- | :--- |
| `run_twdtw`, `run_twdtw_batch`, `classify_twdtw(values, dates, patterns)` (fora do `__all__`) | `zeit.twdtw(cube, patterns)`: padrões como `dict` nome → série (`DataFrame`/`DataArray` com datas) ou amostras de pixels; saída `Dataset` com `label`, `distance` e a distância por padrão |
| `run_tmask_pixel`, `apply_tmask_stack(dates, green, swir)` | `zeit.tmask(cube)`: bandas `green`/`swir1` pelo nome no cubo `(time, band, y, x)`, datas de `time`; saída máscara `(time, y, x)` `bool` |
| `apply_savgol_filter`, `apply_whittaker_filter` (numpy, Whittaker fora do `__all__`) | `zeit.smooth(cube, method="whittaker"\|"savgol", ...)`: mantém `time`, georreferência e dtype; Whittaker com pesos pelo espaçamento das datas e NaN como peso 0 |
| `run_snic`, `snic_grid`, `snic_to_polygons` | `zeit.snic(cube, spacing=...)`: `Dataset` com `segment (y, x)` e médias; `.to_polygons()` já com CRS e transform |
| `train_ccdc_classifier`, `classify_ccdc_stack(clf, path, path)` | `zeit.classify(result_ou_cubo, model)`: aceita o `Dataset` do `zeit.ccdc` (coeficientes numa data, via `predict_synthetic_image`) ou qualquer cubo; treino a partir de pontos (`GeoDataFrame` com a coluna da classe) |
| `apply_mmu_filter(path, path)`, `apply_majority_filter`, `apply_bayesian_filter` (numpy) | mesmos nomes recebendo e devolvendo `DataArray` |
| `extract_water_mask(coefs, idx, idx)` | bandas pelo nome a partir do resultado do `zeit.ccdc` |
| `regularize_time_series(cube, freq)` | `load_raster(..., freq="16D")` ou `zeit.regularize(cube)` com o tempo do cubo; decidir na hora |

- Mesmo roteiro das fases anteriores: entrada lida pelo `load_raster`, motor antigo vira
  bloco interno, resultado idêntico ao motor, CLI/docs/exemplos migrados na mesma fase,
  linha nova na página "Upgrading to the one-function API".
- `zeit.plot(cube, fit=...)` passa a desenhar também o TWDTW (padrão alinhado) e a curva
  suavizada.
- Dividida em 9a (TWDTW e suavização, que são as que mais aparecem nos tutoriais) e 9b
  (TMASK, SNIC, classificação, filtros, máscara de água e regularização).

### 9a: `zeit.twdtw` e `zeit.smooth` — **Feito** (0.41.0)

- `zeit.twdtw(data, patterns, *, band, steepness=0.1, midpoint=50, cycle="year",
  max_elapsed, nodata, chunks, n_jobs)`: a distância do pacote R twdtw 1.0.1 (Maus), não
  mais a parametrização própria (`alpha`, `beta`, `gamma`): peso logístico
  `1 / (1 + exp(-steepness (Δt - midpoint)))`, Δt entre dias do ano dando a volta no ano
  (`cycle_length` novo no `TWDTWParams`, 0 mantém o antigo), o padrão casando em qualquer
  trecho da série (`subsequence_matching`), datas sem valor fora da série de cada pixel
  (o lote do C++ passou a descartar NaN, como o `complete.cases` do R; antes uma nuvem
  estragava a distância do pixel). Paridade com o R (`tests/test_twdtw_api.py`, fixtures
  geradas no R 4.4.2 em `tests/data`): 12 pares série × padrão iguais a 1e-12 (uma e duas
  bandas, datas nubladas, padrão dentro de quatro anos de série).
- Padrões como `pandas.Series`/`DataFrame` indexados por datas ou `DataArray`; cubo
  `(time[, band], y, x)` (bandas pelo nome), arquivo ou um pixel. Resultado: `label`
  (1.., 0 sem valor, nomes no `flag_meanings` que o `zeit.plot` agora lê para a legenda),
  `distance`, `distances (pattern, y, x)` e os próprios padrões (`pattern_value`,
  `pattern_time`), que o `save_raster` pula (variáveis sem `y`/`x`) e o
  `zeit.plot(cube, fit=resultado)` usa para desenhar o melhor padrão alinhado à série do
  pixel clicado (`_twdtw_api.match`).
- `zeit.smooth(data, *, method="whittaker"|"savgol", lmbda, weights, window, polyorder,
  nodata, chunks, n_jobs)`: Whittaker novo em C++ (`src/whittaker.cpp`, Cholesky de banda
  2) com diferenças divididas para datas irregulares, medidas em passos medianos (numa
  série regular é o clássico, igual ao `apply_whittaker_filter`), NaN com peso 0 (as
  lacunas são preenchidas) e `weights`; Savitzky-Golay do scipy com as lacunas
  interpoladas antes. Mantém dims, datas e georreferência; inteiros com NoData voltam ao
  tipo. `zeit.plot(cube, fit=suavizado)` desenha a curva.
- `zeit/twdtw.py` e `zeit/smooth.py` viraram `_twdtw.py` e `_smooth.py` (os nomes são as
  funções); `run_twdtw`, `run_twdtw_batch` e `apply_savgol_filter` saíram do `zeit`.
  Accessor: `cube.zeit.twdtw(...)`, `cube.zeit.smooth(...)`. Docs, tutorial (figura
  regerada), README, exemplo 15 e página de migração atualizados.

### 9b: TMASK, SNIC, classificação, filtros, água e regularização — **Feito** (0.42.0)

- `zeit.tmask(cube, green="green", swir="swir1", scale, nodata, chunks)`: bandas pelo nome,
  datas ordinais tiradas de `time`, `clear (time, y, x)` georreferenciado, lazy por blocos;
  o motor por pixel (`run_tmask_pixel`, sklearn Huber) é o mesmo. Mudança: observações sem
  dado agora são `False` (antes `True`). O sklearn passou a ser importado só no uso, e o
  `test_package` confere que `import zeit` não o carrega.
- `zeit.snic(cube, ...)`: `Dataset` com `labels (y, x)` georreferenciado, `means (segment,
  ...)` com as dims e coordenadas do cubo, `n_pixels` e centróides em coordenadas do mapa;
  `snic_to_polygons` aceita o `Dataset` (transform, CRS e colunas nomeadas pelas
  coordenadas, `2022-01-01_ndvi`). Accessor `run_snic` virou `snic`.
- `zeit.train_classifier(dado, amostras, label=, model=, date=)` e `zeit.classify(dado,
  modelo, date=, probability=)`: atributos = o que o pixel tem fora de `y`/`x` (bandas,
  datas × bandas, mapas de um `Dataset`) ou, num resultado do CCDC com `date`, o modelo do
  segmento daquela data (coeficientes com o intercepto movido para a data, `a0 + c1 t`, e o
  RMSE; o `segment_at` saiu do `predict_image` para ser compartilhado). Amostras como
  pontos (`GeoDataFrame` ou arquivo), reprojetadas; modelo padrão RandomForest de 100
  árvores; `zeit_features_` no modelo para conferir e reordenar na predição. Saída `label`
  com `flag_meanings` e `probability (class, y, x)`. `classify.py` virou `_classify.py`
  (`train_ccdc_classifier`/`classify_ccdc_stack` saíram do `zeit`).
- Filtros (`apply_mmu_filter`, `apply_majority_filter`, `apply_bayesian_filter`): mapas
  (`DataArray`, numpy ou caminho) entram e saem do mesmo tipo, com georreferência e NoData;
  o `apply_mmu_filter` deixou de ler e gravar arquivos (a CLI `mmu-filter` continua igual).
- `extract_water_mask(segmentos, green=, swir=, threshold=)` no resultado do CCDC, pelo
  nível do primeiro segmento no meio dele (`a0 + c1 t`: o `a0` sozinho é o intercepto no
  ano 0, não uma reflectância). No formato numpy antigo, corrigido o bug documentado: o
  intercepto da banda b está em `4 + 9 b`, não `4 + 7 b`.
- `regularize_time_series` aceita qualquer entrada do `load_raster`, faz o medoide de cubos
  de uma banda e mantém CRS e NoData; o nome ficou (já era do padrão do cubo).
- Testes em `tests/test_cube_tools_api.py`; docs, tutoriais (TMASK, SNIC, CCDC,
  LandTrendr), README, exemplos 12, 14, 17 e 18 e página de migração atualizados.

### Tmask em C++ — **Feito** (0.43.0)

- `src/tmask.cpp` (`_core.tmask`): o mesmo modelo (`a0 + c1 t + a1 cos wt + b1 sin wt`) e os
  mesmos limiares (0,04 de reflectância), mas o ajuste robusto passou do `HuberRegressor`
  do sklearn (L-BFGS-B, impossível de reproduzir bit a bit) para o `robustfit` do MATLAB
  (bisquare, como no artigo e no `autoTmask` do CCDC), reaproveitando o `robust_fit` já
  validado do `ccdc.cpp` por um invólucro público (`robustfit_bisquare`), sem mexer nele.
- Medido (140 datas): 58 µs por pixel numa thread contra 11 ms do sklearn (~190×), 9 µs
  com 19 threads. Num cubo sintético com 15% de nuvens e 10% de sombras: 99,98% de
  concordância com o motor antigo; o C++ acertou todas as contaminações, o sklearn deixou
  passar 12 de 400 × 140.
- `tests/test_tmask_engine.py`: o `robustfit` contra uma porta Python independente do
  algoritmo do MATLAB (1e-8), nuvens e sombras injetadas, séries curtas e as convenções do
  `apply_tmask_stack`. O sklearn não é mais usado pelo Tmask.

## Testes em todas as plataformas e `load_raster(..., crs=, res=)` — **Feito** (0.44.0)

Antes da Fase 10, duas coisas pequenas: o CI só testava o código no Linux, e o "Para
depois" tinha o `crs=`/`res=`.

- **Tests** (`tests.yml`): Linux, Windows e macOS no Python 3.12, mais o 3.10 e o 3.14 no
  Linux (antes, só Linux 3.12). No Linux o torch vem do índice de CPU do PyTorch, sem os
  gigabytes de CUDA. Bugs como o do 0.37.1 (as DLLs do Python carregadas depois do
  rasterio no Windows) aparecem aqui, antes do PyPI.
- **Wheels** (`build_wheels.yml`): cibuildwheel 2.17 → 4.2.1, que passa a construir o
  cp313 e o cp314 (o 2.17 parava no cp312; quem usava o 3.13 compilava do sdist), empacota
  as DLLs do Windows com o delvewheel (o `vcomp140.dll` do OpenMP vai junto) e confere os
  wheels com o abi3audit. Cada wheel cp312 (manylinux, Windows e macOS) é instalado num
  ambiente limpo e testado com a suíte inteira: arquivos que faltem no pacote (o
  `viewer.js`), bibliotecas não empacotadas e o import fora da árvore do código aparecem
  aqui. Fora: free-threaded (`cp314t`), o cp315 (até o numba e o torch terem wheels) e os
  testes do musllinux (o torch não tem wheels para ele).
- **`load_raster(..., crs=, res=)`**: a grade que o `gdalwarp -t_srs crs -tr res` faria
  (`_warp.suggested_grid`): a extensão que cobre o dado no CRS novo (a sugerida pelo
  GDAL), células de `res` a partir do canto superior esquerdo ou a resolução sugerida; uma
  pasta cobre todos os arquivos, na resolução mais fina deles, como o gdalwarp com várias
  entradas. Depois é o mesmo warp do 8a (cada data com a própria máscara, lazy, por
  janela). Sem `crs`, o CRS do dado; o próprio CRS sem `res` devolve a grade como está.
  `like=` junto com `crs=`/`res=` é erro, e numpy pede `like=` antes. Conferido contra o
  gdalwarp 3.12 (a grade em sete casos, um arquivo e pasta, e os valores do bilinear bit a
  bit com `tolerance=0`); os testes guardam as grades do gdalwarp para o cubo de teste.
  Sem `-tap`: para alinhar fontes diferentes, a primeira com `crs=`/`res=` e as outras
  com `like=` ela.

## Fase 10: fechar a convenção do cubo — **Feito** (0.45.0)

Depois da Fase 9 ainda sobram pontas fora do padrão "uma função, qualquer entrada, saída
georreferenciada": o SOM (o único algoritmo do README que ainda é uma classe sobre numpy
cru), a CLI (que parou nos algoritmos das Fases 3–5), os decodificadores de QA e dois
módulos que o `load_raster` e o `regularize_time_series` já cobrem.

| Hoje | Depois |
| :--- | :--- |
| `zeit.ai.SOM(x, y, input_len)` + `train`/`predict` sobre `(n_amostras, n_atributos)` montado à mão; importar pede o torch (o `zeit.ai/__init__` carrega os modelos) | `zeit.som(dado, x=, y=, ...)`: atributos e amostragem do `train_classifier`, saída `Dataset` georreferenciado; a classe continua como motor |
| `som.filter_noisy_samples(X, y)` | `zeit.clean_samples(dado, amostras, label=)`: os pontos de volta com uma coluna `keep` (o `sits_som_clean_samples`) |
| CLI: `landtrendr`, `ccdc`, `bfast-monitor`, `bfast-lite`, `bfast`, `mann-kendall`, `mmu-filter` | mais `phenology`, `smooth`, `tmask`, `twdtw`, `snic`, `classify` e `som` |
| `qc_modis_summary`, `qc_modis_state`, `qc_sentinel2_scl` só numpy; a docstring do módulo aponta para o `run_phenology`, que saiu na Fase 5 | recebem e devolvem `DataArray` (dims, `time`, georreferência), prontos para o `weights=` do `zeit.phenology` e do `zeit.smooth` |
| `build_local_cube(pasta, regex, date_format)` | sai: `load_raster(pasta, pattern=, date_format=, recursive=True, chunks="auto")` |
| `zeit.preprocessor.cbers_to_landtrendr`, `cbers_to_ccdc` (fora do `__all__`, com `print`; o segundo grava um CSV de datas que o `zeit.ccdc` não precisa mais) | saem: `regularize_time_series(cubo, freq="YS", method="medoid")` e `zeit.ccdc(cubo)` |

### `zeit.som` e `zeit.clean_samples`

`zeit.som(data, *, x=3, y=3, sample=50000, num_iters=None, algorithm="online", sigma=1.0,
learning_rate=0.5, decay="linear_decay_to_zero", neighborhood="gaussian",
topology="rectangular", init="random", seed=42, nodata="auto", chunks=None, n_jobs=-1)`
em `zeit/_som_api.py`:

- Atributos pela mesma função do `train_classifier` (`_classify_api.features`): tudo o que o
  pixel tem fora de `y`/`x`. NoData como no resto (`"auto"`; num `Dataset` de mapas, só o
  NoData gravado, porque o 0 é valor em mapas como `n_breaks`). Treino numa amostra de
  `sample` pixels válidos (reprodutível por `seed`), predição em todos, lazy por blocos.
- Resultado: `label (y, x)` (1..x·y, 0 sem valor, `flag_meanings` `i_j`), `distance (y, x)`,
  `prototypes (neuron, ...)` com as dims e coordenadas do cubo, `n_pixels (neuron)`, as
  coordenadas `i`/`j` e o `quantization_error`. O `save_raster` grava `label` e `distance`.
- `zeit.plot(cube, fit=som)` desenha o protótipo do neurônio do pixel clicado.
- Paridade: com `sample=None`, os protótipos são os do motor bit a bit e o `label` é o
  `SOM.predict` + 1 (testado com os três `init`, batch e online, dois `decay`).
- **Padrões diferentes dos planejados**, decididos por medição:
  - `algorithm="online"`, não `"batch"`: numa cena sintética de três trajetórias, o batch
    numa grade pequena deixa neurônios vazios (um neurônio que não ganha nenhuma amostra é
    puxado para a média do vizinho e fica igual a ele): acertou 13 de 20 sementes no 3×1 e
    16 de 20 no 2×2, contra 20 de 20 do online em tudo. Em 50 mil amostras × 60 datas os dois
    levam ~0,6–0,8 s.
  - `decay="linear_decay_to_zero"`, não o `asymptotic_decay` do MiniSom: o padrão do
    MiniSom termina com 1/3 da taxa de aprendizado, e os protótipos ficam puxados pelas
    últimas amostras. Em 30 mil trajetórias de Rondônia (2×2): distância média 4532 contra
    5005, protótipos a até 474 da média dos seus pixels contra 955. O batch fica com os
    protótipos mais perto da média (195), mas com distância média maior (4645).
  - `init="random"`, não `"pca"`: na grade de um neurônio de largura (3×1) o `"pca"` do
    MiniSom é degenerado (`linspace(-1, 1, 1)`) e junta classes.
- Rondônia inteira (40 anos × 1671 × 1686, 30 mil amostras): 2,1 s em memória, 4,4 s lazy,
  mesmo resultado.
- O motor saiu de `zeit/ai/som.py` para `zeit/_som.py` (sem torch); `zeit.ai.SOM` continua.
  Accessor `cube.zeit.som(...)`.
- `zeit.clean_samples(data, samples, *, label, x, y, ...)`: o SOM nos atributos das amostras
  (grade padrão de ~5√n neurônios, a regra de Vesanto) e as amostras de volta, no CRS
  delas, com `neuron`, `neuron_class`, `purity` e `keep`. No exemplo 16 marca exatamente as
  64 de 1600 amostras com rótulo trocado.

### CLI

- `phenology`, `smooth`, `tmask`, `twdtw`, `snic`, `classify` e `som`, todos por um corpo só
  (`_run_cube_cli`): `load_raster` lazy com `--chunk-size`, `--jobs`, e uma chamada de
  `save_raster` para `<output_dir>/<prefix>.tif`. `--weights` (fenologia e suavização) lê
  um raster de pesos; `twdtw --patterns` lê um CSV `pattern,date,<valor>` (ou uma coluna
  por banda; padrões a partir de pontos ficaram de fora, o `zeit.twdtw` não os aceita);
  `snic --polygons` grava o `.gpkg`; `classify --samples ... --model rf.joblib` treina e
  salva, `classify --model rf.joblib` reaproveita; `som` grava também
  `<prefix>_prototypes.csv`.
- `tests/test_cli.py`: cada subcomando contra a função Python (não havia testes de CLI).
  `docs/cli.md` com as seções 8–14.

### QC, `build_local_cube` e `preprocessor`

- `qc_*` aceitam numpy, `DataArray` ou caminho e devolvem o mesmo tipo (um `DataArray` com
  as dims, datas e georreferência, lazy se era); NoData e NaN da QA pesam 0. Docstring do
  módulo corrigida.
- `build_local_cube` e `zeit/local.py` saíram (exemplo 20, tutorial de STAC e referência
  passaram ao `load_raster`; o caso do `test_local_cube` já estava no `test_io`).
- `zeit/preprocessor.py` saiu. Ao conferir o `regularize_time_series(freq="YS",
  method="medoid")` contra o `cbers_to_landtrendr`, apareceu um bug no medoide: uma data
  sem valor tinha distância 0 (o `sum` do xarray pula NaN) e virava o medoide do pixel.
  Corrigido (`skipna=False`); fora os empates, o composto é o do `cbers_to_landtrendr`.
- Sem página de migração: a documentação atual foi atualizada no lugar, e a página
  "Upgrading to the one-function API" (das Fases 1–9) saiu do site e do README.

## Fase 11: `zeit.ai` de cubo a mapa — **Feito** (0.46.0)

Hoje o `zeit.ai` tem os modelos (U-TAE, L-TAE, TempCNN, Siamese, ViT) como `nn.Module`
soltos, mais o `STACCubeDataset` e três perdas. O caminho entre um cubo e um mapa fica com o
usuário: os tutoriais começam de `X: (n_amostras, n_bandas, n_datas)` já montado, escrevem o
laço de treino e, na inferência, a georreferência se perde. É o oposto do resto do zeit.
O `STACCubeDataset` também tem problemas próprios: descarta as bordas que não cabem num
patch inteiro, troca NaN por 0, usa o dia do ano como posição (séries de mais de um ano
colidem) e devolve uma tupla, embora a anotação diga `dict`.

A meta é o mesmo fluxo do `train_classifier`/`classify`, com os modelos profundos:

```python
samples = zeit.ai.samples(cubo, "pontos.gpkg", label="classe")          # pixels ou patches
model = zeit.ai.train(zeit.ai.TempCNN, samples, epochs=50)              # laço pronto
mapa = zeit.ai.predict(model, cubo)                                     # Dataset georreferenciado
mapa.zeit.save("classes")
```

### O que foi feito

Tudo em `zeit/ai/pipeline.py`, exportado por `zeit.ai`: `samples`, `SampleSet`, `train`,
`predict`, `save` e `load`. Um adaptador por modelo diz como construí-lo a partir das
amostras e como chamá-lo; outros modelos continuam com o `SampleSet` e um laço próprio.

- **`samples(data, amostras, *, label, patch=None, split=0.2, split_by="block",
  block_size=64, seed, nodata)`**: pontos e polígonos rasterizados na grade do cubo (pontos
  no mesmo pixel contam uma vez). Modo pixel: `(time, band)` por pixel rotulado. Modo patch:
  uma janela por ponto ou polígono pequeno e janelas que cobrem os polígonos maiores, com
  todos os pixels rotulados dentro e `-1` no resto. Validação por blocos espaciais inteiros.
  Normalização por banda pelos quantis 2–98 do treino; o dado bruto fica no `SampleSet`
  (NaN) e cada item sai normalizado, com ausentes como 0. Par de imagens `(antes, depois)`
  para o Siamese (posições 0 e 1).
- **`train(modelo, amostras, *, epochs=50, batch_size=None, lr, weight_decay, loss="ce"|
  "focal"|função, patience=10, device="auto", seed, verbose, **model_kwargs)`**: a classe é
  construída com bandas, datas e classes das amostras; Adam; melhor época restaurada;
  `zeit_meta_` e `zeit_history_` no modelo. `epochs=0` só registra os metadados, para um
  modelo treinado num laço próprio.
- **`predict(modelo, data, *, batch_size=256, overlap=0.25, probability, device, nodata,
  chunks)`**: bandas pelo nome, normalização do treino. Modo patch: grade global de janelas
  (a última encostada na borda), probabilidades médias ponderadas por uma pirâmide que cai
  para as bordas da janela; lazy, cada bloco lê o próprio núcleo mais uma janela em volta e
  usa as janelas da grade global, então o mapa lazy é o mapa em memória (testado). Saída como
  a do `zeit.classify`, que agora delega para o `predict` quando recebe um modelo do
  `zeit.ai`.
- **`save`/`load`**: pesos, meta, histórico e a classe com os argumentos; `torch.load` com
  `weights_only=True`. Um modelo treinado como instância pede `model=` no `load`.
- **`STACCubeDataset`** corrigido: as janelas cobrem as bordas (a última encostada), cubos
  menores que uma janela dão uma janela com NaN, posições em dias desde a primeira data,
  NaN mantido com uma máscara `valid`, itens como `dict` (com `row`/`col`).
- Testes em `tests/test_ai_pipeline.py` (11, ~20 s na CPU). Tutoriais (o geral e um por
  modelo), referência, README e os exemplos 03–06 (agora sintéticos, offline e de ponta a
  ponta) usam o caminho novo.

**Diferenças em relação ao plano, por medição ou por ler o código:**

- O `LightTAE` (porte do `sits_lighttae`) também fixa a linha do tempo na construção
  (`day_offsets` e um `LayerNorm((d_model, seq_len))`): como o TempCNN, pede o mesmo número
  de datas na predição. Só o U-TAE recebe as posições a cada chamada e classifica outras
  datas.
- `batch_size` padrão de 64 pixels, mas 8 patches: com poucos patches, lotes de 64 davam um
  passo por época. No cubo sintético com rótulos densos, o U-TAE pequeno passou de 33–64%
  para 90–95% em seis sementes só com isso.
- A parada antecipada só pode disparar depois de 50 passos do otimizador. A validação roda
  com as médias acumuladas do BatchNorm (momento 0,1), e até elas se acomodarem a perda de
  validação do U-TAE sobe enquanto o modelo aprende: com `patience=10` o treino parava na
  época 11 e restaurava a época 1 (37% no exemplo 04; 99,8% depois da correção). Junto, o
  último lote só é descartado quando tem uma amostra só (antes, `drop_last` jogava fora 4 de
  12 patches por época). Com as duas correções, o U-TAE com rótulos de ponto, que colapsava
  numa classe com a semente 2, ficou entre 98% e 100% nas quatro sementes testadas.
- Modelos de patch aprendem melhor com rótulos densos (polígonos): uma janela em volta de um
  ponto tem um único pixel rotulado. Está na documentação.
- `"tversky"` ficou de fora do `loss=` (o `TverskyLoss` é binário); `"focal"` é uma focal
  que ignora os pixels sem rótulo, e qualquer função `(logits, alvo)` serve.

## O medoide do `regularize_time_series` em C++ — **Feito** (0.47.0)

O `compute_medoid` (`_core.utils`), que ficou sem uso com a saída do `preprocessor`, passou a
fazer o `regularize_time_series(method="medoid")`: por `apply_ufunc` em cada período do
`resample` (lazy por blocos), com todos os pixels de um bloco numa linha só, para o OpenMP
dividir pixels. 60 datas × 4 bandas × 300²: 0,15 s contra 1,0 s do xarray (1 banda: 0,14 s
contra 0,31 s); lazy igual em memória.

- Comparado com a versão em xarray em 5,4 milhões de valores: as diferenças são só (1)
  empates, quando a mediana cai no meio de duas observações (número par de datas): o C++
  calcula em `float64` e fica com a primeira data; o xarray calculava em `float32` e o
  arredondamento decidia; e (2) pixels sem nenhuma observação com todas as bandas no
  período, que agora saem NaN (antes: a primeira data, com o buraco). Teste contra uma
  implementação direta em `float64`, valor a valor.
- Corrigido junto, nos dois métodos: o NoData de um cubo inteiro (Int16 com −9999) entrava
  nas medianas. Agora fica de fora, e o resultado volta ao tipo do cubo com NoData nos
  períodos vazios.

## Objetivo das próximas fases: carregar, rodar, exportar, **validar**

Com as Fases 1–11 o fluxo carregar → rodar → exportar está fechado para todos os
algoritmos. O que falta é o passo que todo artigo ou relatório de mudança pede: dizer
quanto o mapa acerta e quanta área mudou, com intervalo de confiança. Hoje só há o
`generate_landtrendr_accuracy_dashboard` (`zeit/validation.py`), que é um HTML só do
LandTrendr, sem desenho amostral nem estimativa de área. As Fases 12 e 13 fecham esse
ciclo; a 14 abre a frente de degradação florestal, que é o que o público do zeit (Amazônia,
Brazil Data Cube) mais monitora.

```python
lt = zeit.landtrendr(ndvi)
mapa = zeit.extract_events(lt)                                   # ou de zeit.ccdc, zeit.bfast...
pts = zeit.stratified_sample(mapa.yod, n=500)                    # desenho amostral
pts = zeit.plot(ndvi, fit=lt, samples=pts)                       # interpretar e rotular
acc = zeit.accuracy(mapa.yod, pts, reference="ref")              # Olofsson et al. (2014)
acc.area                                                         # área ajustada ± IC95
```

## Fase 12: um esquema só de eventos de mudança — **Feito** (0.48.0)

Cada algoritmo de mudança devolvia a mudança no próprio formato: vértices no LandTrendr
(`vertex_year`/`vertex_value`), `t_start`/`t_end`/`t_break` e coeficientes no CCDC,
`breakpoint`/`magnitude` no BFAST Monitor, índices de quebra no BFAST Lite e no BFAST.
Comparar algoritmos, validar ou combinar mapas pedia código diferente para cada um. O
`extract_events` já tinha o esquema certo, mas só lia o LandTrendr.

| Antes | Depois |
| :--- | :--- |
| `extract_events(lt)` só com o `Dataset` do LandTrendr (ou numpy de vértices) | `extract_events(resultado)` com o `Dataset` de `zeit.landtrendr`, `zeit.ccdc` (`band=`), `zeit.bfast_monitor`, `zeit.bfast_lite` e `zeit.bfast` |
| comparar dois algoritmos = código próprio | `zeit.agreement(mapa_a, mapa_b, ..., tolerance=1)` |
| combinar algoritmos = fora do zeit | receita na referência: os eventos empilhados como atributos do `train_classifier` (ensemble no estilo LCMS), testada |

### O que foi feito

- **As mesmas variáveis para todos:** `yod`, `date`, `magnitude`, `duration`, `pre_val`,
  `post_val`, `rate`, `dsnr`, com os mesmos tipos (o LandTrendr ganhou `date`). Convenção
  de tempo única, a do LandTrendr: `yod` é o ano da **última observação antes** da mudança
  e `date` a data da **primeira que a mostra**. Assim um desmatamento de fevereiro de 2005
  dá o mesmo `yod` num composto anual de julho (LandTrendr) e na série de 16 dias (CCDC,
  BFAST), e a comparação entre algoritmos não ganha um ano de viés.
- **Candidatos por pixel e uma escolha só** (`_select_event`, lazy por um peso one-hot como o
  `segment_at`): cada algoritmo vira uma pilha de eventos candidatos (mudança com sinal,
  valores antes e depois, ruído do ajuste, datas) e o `sort_by` escolhe igual para todos.
  - **CCDC:** cada quebra seguida de outro segmento; a mudança é o modelo do segmento
    seguinte menos o do anterior, os dois na data da quebra. Com `band=`, numa banda (com
    direção); sem, o comprimento do vetor de mudança nas `detection_bands` da rodada
    (`event_type="any"`), e o DSNR é o vetor padronizado banda a banda pelo RMSE. Uma
    quebra no fim da série, ainda sem modelo depois, não tem magnitude e fica de fora.
  - **BFAST Monitor:** a quebra, com a mediana dos resíduos do monitoramento e o `sigma`.
  - **BFAST Lite e BFAST:** cada quebra (de tendência, no clássico). Os motores em C++
    passaram a devolver a magnitude de cada quebra (no Lite, os dois modelos na primeira
    observação depois da quebra, para o sazonal se cancelar; no clássico, o salto da
    tendência de cada quebra, do qual o `magnitude` antigo é o maior) e o índice da
    primeira observação válida depois dela. Os resultados ganharam `magnitude_k`,
    `break_date_k` (Lite), `trend_magnitude_k`, `trend_break_date_k` (clássico) e
    `break_date` (Monitor), com as datas reais do cubo: o eixo `ts` do R (início + i/23)
    escorrega ~3 dias por ano das datas de 16 dias, e com nuvens a primeira observação
    depois da quebra não é `idx + 1`.
- `event_type="any"` novo para os algoritmos de quebra (padrão deles); `pre_val_threshold`
  só onde há valor antes (LandTrendr, CCDC com `band=`); `sort_by="dsnr"` no BFAST Lite
  pelo desvio padrão residual (`sqrt(rss / n_valid)`), indisponível no clássico.
- **`zeit.agreement`:** `yod` de consenso (o ano com mais mapas a até `tolerance` anos;
  empate: o ano que mais mapas dão exatamente, depois o mais antigo), `n_detected`,
  `n_agree`, `spread` e `agrees (map, y, x)`; mapas em grades diferentes dão erro apontando
  o `load_raster(like=)`.
- **Bug corrigido:** os `breakpoint_idx` do BFAST Lite e do BFAST sempre foram índices na
  série inteira (o C++ volta pelo `valid_rows`), mas a documentação dizia "entre as
  observações válidas" e o `zeit.plot` os tratava assim: com nuvens, a quebra era desenhada
  na data errada. Agora o `zeit.plot` usa o índice direto e desenha os eventos de quebra
  como uma linha na `date`.
- Testes em `tests/test_events.py` (13): quebras sintéticas conhecidas em cada algoritmo
  (data, ano, magnitude, direção), o vetor de mudança do CCDC, lazy igual a em memória,
  lacunas e `sort_by="newest"`, o mesmo esquema entre LandTrendr e BFAST, `agreement`
  (tolerância, desempates, grade), o ensemble pelo `train_classifier` e o viewer. O
  LandTrendr continua idêntico (só ganhou `date`).

## Fase 13: avaliação de acurácia e estimativa de área

As boas práticas de Olofsson et al. (2014) e o "Good Practices" do GFOI: amostra
estratificada pelo mapa, rótulos de referência interpretados, matriz de confusão
ponderada pela área de cada estrato, acurácias do usuário e do produtor e área ajustada,
todas com intervalo de confiança. No R isso é o `sits_sampling_design`,
`sits_stratified_sampling` e `sits_accuracy`; em Python não há nada no padrão do cubo.

### 13a: desenho amostral — **Feito** (0.49.0)

- `zeit.sampling_design(mapa, *, expected_ua=0.75, std_error=0.01, alloc="equal"|
  "proportional"|"neyman"|dict, min_per_stratum=50)`: tamanho total da amostra
  (Cochran, eq. 13 do Olofsson) e alocação por estrato, como `DataFrame` (estrato, área em
  pixels e em hectares a partir do transform, proporção, n). Estratos = valores do mapa;
  `yod` e outros mapas contínuos aceitam `bins=` (ex.: "mudança" × "sem mudança", ou um
  estrato por período).
- `zeit.stratified_sample(mapa, n=None, *, design=None, buffer=None, seed=42)`:
  `GeoDataFrame` de pontos nos centros dos pixels, no CRS do mapa, com `stratum` e o peso
  de inclusão. Lazy por blocos (um mapa nacional não precisa caber na memória: contagem
  por bloco, depois sorteio por bloco). `buffer=` evita pares de pontos vizinhos (fronteira
  de mudança).

### 13b: `zeit.interpret`, o modo de interpretação — **Feito** (0.50.0)

Uma função própria, não um parâmetro do `zeit.plot`: interpretar 500 pontos é outra tarefa
que explorar um cubo. É uma fila de pontos e não um pixel qualquer; devolve dados (os
rótulos são o produto); dura dias (salva sozinha e retoma); e mostrar o mapa e o ajuste do
algoritmo enviesa o intérprete (a interpretação de referência deve ser cega ao mapa). O
`zeit.plot` continua só mostrando; as duas funções dividem o motor do viewer (`Session`,
WebGL, série do pixel, basemap, janela fora do notebook) com layouts diferentes.

```python
pts = zeit.stratified_sample(mapa.yod, n=500)
s = zeit.interpret(cubo, pts, classes=["estável", "perda", "ganho"],
                   rgb=["red", "green", "blue"], save="referencia.gpkg")
acc = zeit.accuracy(mapa.yod, s)            # ou s.accuracy(mapa.yod)
```

`zeit.interpret(data, samples, *, classes, save=None, rgb=None, chips="year", fit=None,
map=None, blind=True, interpreter=None, basemap=None)`:

- **Tela por ponto:** à esquerda, a lista de pontos com o estado (feito, pulado, dúvida) e o
  progresso; no centro, a imagem centralizada no pixel com o slider no tempo e o basemap, e
  embaixo uma faixa de recortes em volta do ponto, um por ano (`chips=`; RGB com `rgb=`, o
  índice sem), como no TimeSync; à direita, a série do pixel (clique marca a data do
  evento) e o formulário: classe (teclas 1–9), confiança, comentário. Enter salva e vai
  para o próximo; setas navegam.
- **Cego por padrão:** com `blind=True`, o valor do mapa (`map=`) e o ajuste do algoritmo
  (`fit=`) ficam escondidos até o ponto ser rotulado.
- **Salva a cada rótulo** em `save=` (`.gpkg`, `.parquet` ou `.csv`); chamar de novo com o
  mesmo arquivo retoma de onde parou. Colunas `ref`, `ref_date`, `confidence`, `note`,
  `interpreter`, `labelled_at`, mais o que a amostra já tinha (`stratum`, peso).
- **Aba de revisão**, depois de rotular: matriz de confusão e acurácias (global, do usuário,
  do produtor, área ± IC95) ao vivo, pelos pesos do desenho amostral; clicar numa célula
  ("mapa: perda, referência: estável") filtra a lista para esses pontos, que podem ser
  reabertos e corrigidos.
- **Retorno:** um objeto `Interpretation` com `.samples` (o `GeoDataFrame`) e
  `.accuracy(mapa)`; no notebook o widget é exibido e o objeto atualiza ao vivo; num script
  abre a janela e bloqueia até fechar, como o `zeit.plot`.
- O `zeit.accuracy` aceita esse objeto ou qualquer `GeoDataFrame` com a coluna de
  referência (interpretado no QGIS, no Collect Earth, em campo).
- O `generate_landtrendr_accuracy_dashboard` sai, substituído por isso.
- Testes: a sessão sem navegador (pedidos `meta`/`sample`/`label` direto na `Session`,
  como os do viewer), salvar e retomar, o modo cego, e um teste de ponta a ponta no Edge
  headless rotulando três pontos pelo teclado.

### 13c: `zeit.accuracy` — **Feito** (0.49.0)

- `zeit.accuracy(mapa, amostras, *, reference="ref", strata=None, date_tolerance=None)`:
  o valor do mapa em cada ponto contra o rótulo; estratos do próprio mapa ou de `strata=`
  (outro mapa, quando a amostra foi estratificada por algo diferente do mapa avaliado).
- Resultado (um objeto com `__repr__` legível e `.to_dataframe()`):
  `confusion` (contagens e proporções de área), `overall`, `users`, `producers` (com erro
  padrão e IC95), `area` (área ajustada por classe em hectares, ± IC95) e a área mapeada
  ao lado, para mostrar o viés.
- `date_tolerance=` para mapas de data de mudança: acerto se a classe bate e a data do
  mapa está a até N anos (ou dias) da `ref_date`; acurácia da data à parte.
- Sem amostra estratificada (pontos de campo, por exemplo): matriz de confusão simples,
  com aviso de que a área não é estimável.
- Testes: os números do exemplo numérico do Olofsson et al. (2014, tabelas 8–10)
  reproduzidos; paridade com o `sits_accuracy` em fixtures geradas no R, como as do TWDTW;
  em mapa sintético com erro conhecido, a cobertura do IC95 perto de 95% em muitas
  sementes.

### 13a e 13c: o que foi feito (0.49.0)

Tudo em `zeit/_accuracy.py`: `sampling_design`, `stratified_sample`, `accuracy` e o objeto
`Accuracy`. Referência em `docs/api/validation.md` e tutorial "Accuracy & Area".

- **Estratos do mapa:** as classes (nomes do `flag_meanings`), intervalos (`bins=`) ou, num
  mapa de eventos (`extract_events`, `agreement`), "no change" e "change" (com `bins=`, um
  estrato por período do `yod`). Num mapa de eventos o 0 é o estrato "no change", não NoData.
- **Áreas pelo tamanho de cada pixel**, em hectares; num CRS geográfico, a área de cada
  linha na esfera autálica (a de um pixel de 30 m no Equador e a 30° S não é a mesma). As
  contagens por estrato e linha são uma passada lazy pelo mapa.
- **Sorteio independente dos chunks:** por estrato, `rng.choice` dos postos globais
  (linha a linha) e só as linhas sorteadas são lidas; mesma semente, mesmos pontos em
  memória ou lazy com qualquer chunk (testado).
- **Estimadores de Stehman (2014)** para amostra estratificada (médias de indicadores e
  razões, pesadas pela área de cada estrato), então os estratos não precisam ser as classes
  do mapa avaliado: `accuracy(mapa_b, pontos, strata=mapa_a)`. Com os estratos iguais às
  classes, são as eqs. 4–11 do Olofsson.
- **Paridade:** o exemplo numérico do Olofsson et al. (2014) (áreas 21.158 ± 6.158,
  11.686 ± 3.756, 285.770 ± 15.510 e 581.386 ± 16.282 ha; global 0,947 ± 0,018; usuário e
  produtor); o `sits:::.accuracy_area_assess` do sits 1.5.4 rodado no R 4.4.2 com as áreas
  no lugar do cubo (`tests/data/make_accuracy_sits.R`), igual a 1e-9 em dois casos; a
  variância do produtor contra a eq. 7 escrita à parte (o sits não dá os erros padrão das
  acurácias). Numa amostra estratificada por um mapa avaliando outro, os IC95 cobriram a
  acurácia verdadeira em ~95% de 120 sementes.

**Diferenças em relação ao plano:**

- Sem correção de população finita, como o Olofsson e o sits (com ela os números não batiam
  com os deles; num mapa de milhares de pixels ela é desprezível).
- `date_tolerance` é uma medida à parte (`acc.date`: a fração dos pontos que o mapa e a
  referência chamam de mudança com os anos a até N de distância, e a diferença média), e
  não muda a matriz: misturar data e classe na mesma matriz exigiria inventar uma classe.
- Sem `buffer=` no sorteio: rejeitar vizinhos quebraria a independência dos chunks e a
  probabilidade de inclusão igual dentro do estrato. Fica para depois, se fizer falta.
- `alloc` também aceita `"neyman"`; `min_per_stratum` tira os pontos dos estratos grandes
  para manter o `n` total (a recomendação do Olofsson de 50–100 para classes raras).
- Pontos sem `stratum` são pesados como amostra estratificada pelo mapa, com aviso (certo
  para amostra aleatória simples ou estratificada pelo mapa, não para pontos escolhidos).

### 13b: o que foi feito (0.50.0)

- `zeit/_plot/_interpret.py`: `InterpretSession` (uma `Session` do viewer com os pedidos
  `points`, `point`, `label` e `review`), `Interpretation` e `zeit.interpret`. No navegador,
  `interpret.js` monta o mesmo `Viewer` e acrescenta a fila, o formulário, a faixa de
  recortes e a revisão; como o anywidget carrega um módulo só, `viewer.js` e `interpret.js`
  viajam juntos (`interpret_bundle`), no widget e na janela (`show_window(kind="interpret")`,
  que serve também `interpret.js`/`interpret.css`). O `viewer.js` ganhou só o evento
  `chart` e a escala do gráfico, para o que se desenha por cima.
- **Recortes:** a mediana de cada ano (ou mês, ou cada data) numa janela de `chip_size`
  em volta do ponto, com as cores do mapa; completados com NoData na borda dos dados, para
  o ponto ficar sempre no meio. Um clique num recorte mostra aquela imagem.
- **Formulário:** classe (1–9), data (clique no gráfico ou `d` na imagem mostrada),
  confiança (`h`/`m`/`l`), nota; Enter salva e vai ao próximo sem rótulo, `s` pula,
  `n`/`p` navegam. Um clique num ponto do mapa vai até ele. O mapa fica centrado no ponto
  enquanto o usuário não o mover (também quando a janela muda de tamanho).
- **Salvar e retomar:** cada rótulo regrava o arquivo (`.gpkg`, `.geojson`, `.parquet`;
  por um temporário e `os.replace`); o mesmo `save=` retoma a sessão.
- **Cego:** o valor do mapa e o ajuste do algoritmo aparecem depois do rótulo; a revisão,
  depois de todos os pontos. A revisão é o `zeit.accuracy` com os pontos rotulados, mais a
  matriz em pontos, cujas células filtram a lista.
- `generate_landtrendr_accuracy_dashboard` e `zeit/validation.py` saíram; o exemplo 19
  virou o fluxo inteiro (mapa, amostra, rótulos, acurácia: o mapa dá 538 ha de perda, a
  amostra estima 680 ± 70 ha, a área verdadeira é 675 ha).
- Testes em `tests/test_interpret.py`: a sessão sem navegador (fila, ponto cego e
  revelado, recortes, rótulos inválidos), salvar e retomar, a revisão contra o
  `zeit.accuracy`, as opções (`rgb`, `band`, `series`, `chips`, `blind=False`), o widget, e
  no Edge headless o ponto centrado, três pontos rotulados pelo teclado, um recorte clicado
  e a revisão esperando os rótulos.

**Diferenças em relação ao plano:** `.csv` ficou de fora do `save=` (perderia o CRS); a
confiança é `high`/`medium`/`low` em vez de um número; `band=` e `zoom=` novos.

## Fase 14: degradação florestal (SMA, NDFI e CODED) — **Feito** (0.51.0)

O zeit detecta bem desmatamento (perda forte e permanente), mas degradação (corte
seletivo, fogo de sub-bosque) é sutil e curta, e quem a mede usa frações de mistura
espectral e o NDFI (Souza et al. 2005), não NDVI. O `compute_indices` hoje tem NDVI, EVI,
SAVI, kNDVI, NBR, NDMI, NDWI e MNDWI, e nada no zeit faz unmixing.

- **`zeit.unmix(cube, endmembers="souza2005"|DataFrame, *, bands=None, sum_to_one=True,
  nonneg=True, shade=True)`**: frações por pixel e data (`(time, endmember, y, x)`) e o
  `rmse` do ajuste. Motor em C++ (mínimos quadrados não negativos, Lawson–Hanson, com a
  restrição de soma 1 por linha aumentada), OpenMP por pixel, lazy por blocos. Endmembers
  padrão GV, NPV, solo e nuvem do Souza et al. (2005) para Landsat; Sentinel-2 pelas bandas
  equivalentes (a confirmar com a literatura antes de virar padrão).
- **NDFI** no `compute_indices`: `(GVs − (NPV + Soil)) / (GVs + NPV + Soil)`, com
  `GVs = GV / (1 − Shade)`; escala −1..1 (ou 0..200 como no Imazon, `scale=`). Lido pelo
  nome como os outros índices (`compute_indices(cube, ["NDFI"])`).
- **`zeit.coded(cube, *, monitor_start, ...)`**: o CODED (Bullock et al. 2020), CCDC sobre o
  NDFI com regras próprias de detecção (quedas curtas que se recuperam contam como
  degradação, quedas permanentes como desmatamento). Reaproveita o motor do `zeit.ccdc`;
  saída no esquema da Fase 12 com uma variável `class` (degradação, desmatamento, sem
  mudança), então a Fase 13 valida sem código novo.
- Paridade: o unmixing contra o `scipy.optimize.nnls` em pixels aleatórios; o CODED contra
  a implementação de referência do GEE (fixtures exportadas de pixels de Rondônia).
- Exemplo e tutorial em Rondônia: frações, NDFI, o CODED, e a área de degradação estimada
  com a Fase 13.

### O que foi feito (0.51.0)

- **`zeit.unmix`** (`zeit/_sma.py`, motor `src/sma.cpp`): mínimos quadrados totalmente
  restritos (Heinz & Chang 2001), o NNLS de Lawson–Hanson no sistema com a linha da soma 1
  (peso 1000), OpenMP por pixel e data, lazy. Endmembers `"souza2005"`: os que o CODED usa
  (o `cdd_params.yaml` do repositório bullocke/coded: GV, shade, NPV, soil do Souza et al.
  2005 e a nuvem do próprio CODED), para azul, verde, vermelho, NIR, SWIR1 e SWIR2; ou uma
  tabela própria. Escala automática (×10000 ou 0–1), máscara pela fração de nuvem.
- **NDFI**: variável do `unmix` e índice do `compute_indices` (que passou a achar uma banda
  pelo próprio nome do papel, `swir1`) e dos downloads do Earth Engine (pelo `unmix` do GEE).
- **`zeit.coded`** (`zeit/_coded.py`, motor `src/coded.cpp`): a versão do artigo (v0, o
  `cdd_simple.py`): modelo de treino (constante + harmônico anual, RMSE), monitoramento
  pelos resíduos normalizados (`consec` seguidos além de `thresh` RMSEs abaixo do modelo),
  um modelo novo nos `min_years` depois de cada mudança, e floresta ou não pelos modelos:
  Random Forest nos coeficientes (NDFI e frações) com pontos de treino, como no CODED, ou o
  NDFI médio do modelo contra `forest_ndfi`. Saída com `t_change`, `type` (degradação,
  desmatamento, perturbação sem classe), `ndfi_change` e o mapa `strata` (floresta, não
  floresta, degradação, desmatamento) com `flag_meanings`, pronto para o `sampling_design`.
  `extract_events` e `zeit.plot(..., fit=)` leem o resultado; CLI `zeit coded`.
- Exemplo 21 e tutorial "Forest Degradation (CODED)": o mapa dá 51 ha de degradação (os
  cortes leves escapam), a amostra estima 56 ± 6 ha, os verdadeiros são 54 ha.
- Testes em `tests/test_coded.py` (11): o NNLS contra o `scipy.optimize.nnls` (500 sistemas,
  1e-9), frações recuperadas de misturas conhecidas, o `unmix` (soma 1, máscara de nuvem, lazy,
  escalas, endmembers próprios), o NDFI, o motor contra uma implementação do algoritmo em
  Python (200 séries com quedas e lacunas, mais de 50 mudanças, 1e-9), o `coded` numa cena
  sintética (corte seletivo → degradação, corte raso → desmatamento, com e sem pontos de
  treino, lazy), a ligação com `extract_events`, `sampling_design`, `zeit.plot` e
  `compute_indices`, a CLI, e o ramo do Earth Engine com uma imagem falsa.
- Bug achado no caminho (da Fase 12): `save_raster` de um `Dataset` com datas num `.tif`
  único falhava (o `extract_events` agora tem `date`); as datas viram ano decimal também
  empilhadas.

**Diferenças em relação ao plano:**

- Sem paridade com o GEE: não há conta do Earth Engine neste ambiente, e o código das
  versões 1 e 2 do CODED (que usam o CCDC do GEE no monitoramento) não está no repositório,
  só no módulo do GEE. Implementei a versão do artigo, que o `cdd_simple.py` descreve por
  inteiro, e a conferi contra uma implementação independente em Python. As versões 1/2
  dão resultados "parecidos, mas não idênticos", segundo a própria documentação.
- `forest_ndfi=0.5` (sem pontos de treino) é escolha do zeit, não do CODED: o modelo depois
  de um corte seletivo cobre a queda e a recuperação, e o NDFI médio dele fica entre o de
  floresta fechada (~0,85) e o de pasto (< 0). Com pontos de treino, a Random Forest decide.
- Endmembers de Sentinel-2 ficaram de fora: não há um conjunto publicado que eu possa citar
  com segurança; uma tabela própria funciona (`endmembers=DataFrame`).

## Fase 15: embeddings de satélite (TESSERA e AlphaEarth) — **Feito** (0.52.0, 15a–15d)

Modelos de fundação já resumem um ano inteiro de imagens num vetor por pixel, e dois
desses produtos estão prontos, globais e abertos, com a mesma forma:

| | [TESSERA](https://geotessera.org/) (Feng et al. 2025) | AlphaEarth Foundations (Google Satellite Embedding) |
| :--- | :--- | :--- |
| Granularidade | pixel, 10 m | pixel, 10 m |
| Dimensões | 128 | 64 (vetores de comprimento 1) |
| Tempo | anual, 2017–2025 (v1.1) | anual, 2017–2025 (o registro da AWS ainda diz 2018–2024) |
| Sensores | Sentinel-1 e Sentinel-2 | Sentinel-1, Sentinel-2, Landsat e outros |
| Armazenamento | Zarr v3, int8 × escala por pixel, um grupo por zona UTM | COGs int8 por ano e zona UTM; Earth Engine |
| Acesso | biblioteca `geotessera` (MIT, Python 3.12+) | Source Cooperative/AWS sem conta (`aef-loader`); GEE |
| Licença | CC0 (a confirmar) | CC-BY 4.0 (atribuição a Google e Google DeepMind) |

Os dois já são a convenção de cubo do zeit: um cubo anual georreferenciado de 64 ou 128
bandas. Com ele carregado, quase tudo o que existe roda sem código novo: classificação
few-shot (o uso principal dos dois artigos) pelo `train_classifier`/`classify`, validação
pela Fase 13, agrupamento e limpeza de amostras pelo `som`/`clean_samples`, segmentação pelo
`snic`, modelos neurais pelo `zeit.ai`. O que falta é a entrada, um jeito de olhar dezenas
de bandas, a busca por similaridade e a mudança entre anos. E, com dois produtos e os
algoritmos clássicos, comparar quem vê o quê.

```python
emb = zeit.load_embeddings("aoi.gpkg", source="tessera", years=range(2018, 2025))
aef = zeit.load_embeddings("aoi.gpkg", source="alphaearth", years=range(2018, 2025))
zeit.plot(emb)                                                  # PCA em RGB, igual em todos os anos
modelo = zeit.train_classifier(emb.sel(time="2024"), pontos)    # few-shot: 128 features por pixel
mapa = zeit.classify(emb.sel(time="2024"), modelo)
parecido = zeit.similarity(emb, pontos_de_mineracao)            # "ache mais lugares como este"
mud = zeit.extract_events(zeit.embedding_change(emb))           # esquema da Fase 12
zeit.agreement(mud, zeit.extract_events(zeit.embedding_change(aef)),
               zeit.extract_events(zeit.landtrendr(ndvi)), tolerance=1)
```

**Princípios:**

- **Uma função, um adaptador por fonte.** `zeit.load_embeddings(..., source=)` devolve o
  mesmo cubo para qualquer produto; o que muda de um para outro (desquantização, máscaras,
  layout, autenticação) fica no adaptador. Tudo o que vem depois (15b, 15c) não sabe de
  qual produto o cubo veio.
- **Envolver os leitores oficiais, não reimplementar o acesso.** O layout do store do
  TESSERA já mudou duas vezes em 2026 (tiles NPY, depois um store por ano, depois o store
  único com `time`), e a API de tiles/NPY está marcada para sair. O `geotessera` resolve
  store, versão, variante e zona; o zeit monta o cubo lazy no próprio padrão. Dependência
  opcional: `pip install zeit-cdts[tessera]`, com uma mensagem clara quando falta
  (explicando o Python 3.12+). O AlphaEarth são COGs com VRTs, que o rasterio já lê.
- **Seguir a convenção geo-embeddings do Zarr** (os atributos `geoemb:`, que o TESSERA já
  adota): um store que a siga entra sem adaptador próprio (15e).

### 15a: `zeit.load_embeddings` e o adaptador do TESSERA

`zeit.load_embeddings(region=None, *, source, years, like=None, crs=None,
resolution=None, resampling="nearest", version=None, variant=None, depth=None,
dequantize=True, water=False, chunks="auto", cache_dir=None)` devolve um `DataArray`
`(time, band, y, x)` georreferenciado, `band` = `"A00"`..`"A127"` (os nomes que o Earth
Engine usa para o AlphaEarth, `"A00"`..`"A63"`), `time` = 1º de janeiro de cada ano (como
os compostos anuais).

Comum a todas as fontes:

- **Região** como no resto do zeit: bounds, geometrias, arquivo vetorial (`clip=` do
  `load_raster`), ou `like=` um cubo/raster, cuja grade é usada.
- **Lazy sempre.** Um tile MGRS (110×110 km) do TESSERA dá ~62 GB por ano em float32 (~15 GB
  em int8). A desquantização fica dentro do grafo dask, chunk a chunk, alinhada aos chunks
  do armazenamento. `dequantize=False` devolve o int8 (e as escalas, quando houver), para
  guardar barato; o `load_raster` desquantiza quando o lê de volta.
- **Várias zonas UTM:** uma região dentro de uma zona sai no CRS dela. Uma região que cruza
  zonas pede `crs=` (ou `like=`), e cada zona é reprojetada e o mosaico montado; senão, erro
  apontando as zonas. `resampling="nearest"` por padrão: interpolar embeddings cria
  vetores que o modelo nunca produziu. `like=cubo_landsat` (30 m) usa `"average"`, porque a
  média dos embeddings de 3×3 pixels é uma agregação razoável, a mesma que os trabalhos
  com embeddings fazem para mudar de escala.
- **A fonte viaja com o dado.** Os dois artigos avisam que embeddings de produtos ou
  versões diferentes não se misturam (treinar num e prever noutro). Isso vira regra do
  zeit: `attrs` `embedding_source`, `embedding_version`, `embedding_variant` (e a licença,
  para a atribuição do CC-BY); o `save_raster` grava nos metadados (`ZEIT_EMBEDDING`) e o
  `load_raster` lê de volta; o `train_classifier` e o `zeit.ai.train` guardam no modelo; o
  `classify`/`predict` recusam outra fonte ou versão com uma mensagem dizendo qual era qual.
- **Algoritmos que não fazem sentido recusam.** Embeddings já são resumos anuais, sem sinal
  intra-anual nem unidade física: `ccdc`, `bfast*`, `phenology`, `landtrendr`, `twdtw`,
  `regularize_time_series`, `compute_indices` e `unmix` dão erro claro quando o cubo tem
  `embedding_source` (em vez de rodar e devolver lixo).
- **Pontos de treino:** medir se o `train_classifier` num cubo lazy lê só os chunks dos
  pontos. Se não ler, um caminho por pontos no adaptador (no TESSERA, o `sample_points` do
  `geotessera`, uma leitura em bloco por zona), sem montar a região.

Do TESSERA (`source="tessera"`):

- Store, versão e variante pelo `geotessera` (`version="v1.1"` por padrão, como ele
  recomenda); int8 × escala por pixel; escala `NaN` (água) ou `+inf` (terra ainda sem dado)
  viram NaN.
- **`water=True`** acrescenta a máscara de água (escala `NaN`) como coordenada `(y, x)`;
  vem de graça e serve de máscara para o resto do zeit.
- **Matryoshka (v2):** `depth=k` pega só as primeiras k dimensões (os stores v2 são
  treinados para isso). Na v1.1, `depth=` dá erro.

Testes: sem rede, contra um store Zarr falso com o mesmo layout (int8, escalas com `NaN` e
`+inf`, duas zonas UTM, anos com zeros), criado numa fixture: desquantização, máscaras,
mosaico entre zonas, `like=` com média, `depth=`, lazy igual a em memória, ida e volta pelo
`save_raster`/`load_raster`, a recusa de fonte ou versão trocada no `classify` e a recusa
dos algoritmos. Um teste com rede, opcional (marcado), lendo uma região pequena de verdade.

### 15b: olhar e buscar: PCA no `zeit.plot` e `zeit.similarity`

- **`zeit.plot(emb)`** mostra por padrão os três primeiros componentes de uma PCA em RGB
  (como o `visualize` do `geotessera`). A PCA é ajustada **uma vez**, numa amostra de pixels
  de todos os anos, e aplicada a todos: assim a mesma cor quer dizer o mesmo embedding em
  2018 e em 2024, e o slider no tempo mostra mudança de verdade. Estiramento por percentis
  fixos pelo mesmo motivo. O clique num pixel mostra, em vez de dezenas de séries, a
  distância (cosseno) de cada ano ao anterior, a mesma série da 15c. `rgb=["A00", "A05",
  "A10"]` continua mostrando bandas escolhidas.
- **`zeit.similarity(emb, ref, *, by=None, metric="cosine")`**: `ref` são pontos ou
  polígonos (ou um vetor); a referência é a média dos embeddings deles, e o resultado é a
  similaridade de cada pixel a ela, `(time, y, x)` ou `(y, x)`. Com `by=` uma coluna, um mapa
  por valor dela (`(class, y, x)`). Com uma referência de um ano e o cubo de vários, mostra
  onde aquilo aparece em cada ano (mineração, pivôs, desmatamento novo). Lazy, numba por
  bloco. No AlphaEarth os vetores já têm comprimento 1, e o cosseno é o produto escalar.
- Testes: a PCA igual em todos os anos (mesmo vetor, mesma cor), o widget com 128 bandas, a
  similaridade contra uma conta em numpy, `by=`, lazy.

### 15c: mudança pelos embeddings, no esquema da Fase 12

- **`zeit.embedding_change(emb, *, metric="cosine", baseline="previous")`**: `Dataset` com a
  distância de cada ano ao anterior (`baseline="previous"`), ao primeiro (`"first"`) ou a um
  ano dado, `(time, y, x)`, mais o ruído de cada pixel (a mediana das distâncias entre anos,
  robusta a uma mudança só).
- **`extract_events`** lê esse resultado: um candidato por par de anos, `magnitude` = a
  distância, `dsnr` = distância / ruído, `event_type="any"`, `pre_val`/`post_val` vazios (não
  há um valor físico). Convenção de tempo da Fase 12: o embedding do ano Y é o primeiro que
  mostra a mudança, então `yod` = Y − 1 e `date` = Y-01-01. Uma mudança no meio do ano
  aparece em parte em Y e em parte em Y + 1, então a comparação com outros algoritmos pede
  `tolerance=1` (documentar).
- Com isso, `agreement` compara o TESSERA, o AlphaEarth, o LandTrendr, o CCDC, o BFAST e o
  CODED, e a Fase 13 valida sem código novo.
- Testes: mudanças sintéticas conhecidas (um pixel que troca de vetor num ano dá aquele
  `yod`, os estáveis não), o `dsnr`, o esquema igual ao dos outros algoritmos, o `agreement`
  com o LandTrendr e o `sampling_design` sobre o resultado.

### 15d: o adaptador do AlphaEarth (Google Satellite Embedding)

- **`source="alphaearth"`** lê os COGs abertos do Source Cooperative/AWS (`tge-labs/aef`,
  sem conta nem custo), por ano e zona UTM. Os COGs estão gravados de baixo para cima: a
  leitura passa pelos VRTs que eles publicam, como a própria documentação pede. O
  `aef-loader` é a referência do layout; como são só COGs e VRTs, o rasterio que o zeit já
  usa dá conta, sem dependência nova.
- **`backend="gee"`**: o mesmo produto pela coleção do Earth Engine
  (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), pelo `zeit.gee`, para quem já usa o GEE ou
  precisa do ano mais recente antes de ele chegar ao espelho aberto.
- int8 com uma desquantização fixa (sem escala por pixel) e um valor de NoData; vetores de
  comprimento 1. `depth=` e `water=` dão erro (não existem nesse produto).
- `attrs` com a licença e a atribuição pedida pelo CC-BY ("Google e Google DeepMind"), que o
  `save_raster` grava.
- Exemplo e tutorial em Rondônia, no período em que os dados se sobrepõem (2017 em diante):
  os dois produtos lado a lado na PCA; classificação few-shot com os mesmos pontos em cada
  um, e as acurácias comparadas pela Fase 13; a mudança pelo TESSERA, pelo AlphaEarth, pelo
  LandTrendr e pelo CODED no `agreement`, e a área de mudança estimada.
- Testes: COGs falsos de baixo para cima com um VRT na fixture (a orientação certa no
  cubo), a desquantização contra a fórmula, NoData, duas zonas, o ramo do GEE com uma
  imagem falsa (como no CODED), e um teste com rede opcional.

### 15e: qualquer store na convenção geo-embeddings (fica para depois)

- `source=` um caminho ou URL `.zarr`: o adaptador lê os atributos `geoemb:` (dimensões,
  quantização, `scale_array`, `nodata`, `spatial_layout`, fontes, versão) e monta o cubo sem
  código próprio do produto. O adaptador do TESSERA passa a ser um caso disso mais o
  `geotessera` para achar o store.
- Só quando a convenção estiver ratificada e houver um segundo produto que a siga; até lá,
  fica o desenho.

### A verificar antes de começar: o que se achou

- TESSERA: o `open_zone` do `geotessera` (0.11) devolve um `Dataset` lazy, mas com chunks do
  dask do tamanho de um shard inteiro (`(1, 128, 4096, 4096)`, 2 GB em int8) e baixando a
  coordenada `y` da zona inteira (1,8 milhão de valores, 15 MB): 12 s para abrir. O zeit usa o
  `geotessera` só para achar o store (`zarr_store_url`, `dataset_for_location`) e abri-lo
  (`zarr_store`, com as novas tentativas e a leitura direta do bucket), e lê os arrays com o
  `zarr`, em blocos de 512 × 512 alinhados aos shards, com as coordenadas pelo
  `spatial:transform`: 5 s para abrir.
- TESSERA: o padrão é a v1.1 `dclimate`, um store Zarr no Source Cooperative
  (`tessera/tessera/zarr/v1.1-dclimate`, 2017–2025, com o `geoemb:`); as escalas são
  `(time, y, x)`. Na v1.1, `NaN` na escala é "sem dado na origem"
  (`geotessera:mask_source: source_nodata`, `geoemb:landmask: false`), não água: água e
  lacuna não se distinguem. Os grupos `utmNN` guardam o hemisfério sul no CRS norte, com
  northings negativos. Há também `s1_*`/`s2_*` com as contagens de observações por pixel e
  ano (não lidas por enquanto).
- TESSERA: licença CC0 para embeddings e pesos (README do projeto, que pede a citação); o
  CC-BY-SA do registro da AWS é do espelho de lá.
- AlphaEarth: 2017–2025 no índice do espelho aberto (302.466 arquivos); a desquantização
  `sign(v) · (v / 127,5)²` e o NoData −128 estão no README do espelho e conferem (vetores de
  comprimento 1,000 ± 0,01); os VRTs cobrem cada arquivo (um `VRTWarpedDataset` que só inverte
  as linhas); a coleção no GEE é `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`; os arquivos de uma
  zona estão numa grade comum de 10 m.

### O que foi feito (0.52.0)

- **`zeit.load_embeddings`** (`zeit/_embeddings.py`): região, anos, grade (`like=`,
  `crs=`/`res=`), reamostragem `"auto"` (vizinho mais próximo; média a partir de células 1,5×
  maiores), corte pelos polígonos, identidade nos `attrs`. Um adaptador por produto devolve
  um cubo lazy por zona UTM; uma zona só fica como está, várias vão para uma grade pelo
  `to_grid` (`load_raster(like=)`) e se juntam.
- **TESSERA** (`zeit/_tessera.py`): o store pelo `geotessera` (extra `[tessera]`) ou um
  `store=` local sem ele; blocos lidos pelo `zarr` e desquantizados na leitura; `depth=` pelos
  arrays Matryoshka do `geoemb:depths`; região toda ao sul do equador sai no CRS UTM sul
  (as mesmas células), como Landsat e AlphaEarth; versão e variante do `geotessera` (ou do
  próprio store, para um espelho).
- **AlphaEarth** (`zeit/_alphaearth.py`): o índice (78 MB) baixado uma vez para
  `~/.cache/zeit/embeddings` e renovado a cada 30 dias; os COGs lidos pelo endpoint do bucket
  (não pelo gateway, que derruba pedidos sob carga), cada bloco lido e invertido (as mesmas
  células do VRT, conferido numa janela real); blocos de 1024 × 1024 alinhados aos dos
  arquivos e de 16 dimensões, que rodam em paralelo (cada banda é um pedido). `backend="gee"`
  baixa cada ano da coleção uma vez pelo `download_gee_image`.
- **Identidade:** `save_raster` grava `ZEIT_EMBEDDING`, `load_raster` lê de volta;
  `train_classifier` guarda em `zeit_embedding_`, `zeit.ai.samples` em `meta["embedding"]`;
  `classify` e `zeit.ai.predict` recusam outra fonte, versão ou variante. LandTrendr, CCDC,
  BFAST (os três), Mann-Kendall, fenologia, TWDTW, `smooth`, `tmask`, `unmix`, `coded`,
  `compute_indices` e `regularize_time_series` recusam um cubo de embeddings.
- **`zeit.similarity`** e **`zeit.embedding_change`** (`zeit/_embedding_tools.py`);
  `extract_events` lê o `embedding_change` (com `baseline="previous"`); a PCA do `zeit.plot`
  (`pca_rgb`) é ajustada uma vez em amostras de até 6 anos.
- **CLI** `zeit embeddings` (não estava no plano): baixa uma região para um GeoTIFF.
- Em Rondônia (5 × 5 km, 2017–2025, de um notebook na Europa): os dois produtos na mesma
  grade de 554 × 549 células em EPSG:32720; nove anos em 43 s (TESSERA) e 70 s (AlphaEarth).
  A distância mediana de um ano ao outro é 0,05–0,1 no TESSERA e 0,02 no AlphaEarth (cada
  produto tem sua escala). Com limiares de 0,3 e 0,2, os dois veem mudança em 6.459 dos
  mesmos pixels e concordam no ano (±1) em 6.326. Os dois mostram mais mudança de 2017 para
  2018 que nos outros anos.
- Docs: referência "Embeddings", tutorial "Embeddings (TESSERA, AlphaEarth)", CLI, README,
  instalação; exemplo 22 com os dados reais.
- Testes em `tests/test_embeddings.py` (18), sem rede: um store Zarr com o layout publicado
  (esparso, duas zonas, hemisfério sul no CRS norte, escalas `NaN` e `+inf`, um store v2 com
  `geoemb:depths`), uma cópia dos COGs de baixo para cima com o índice, o Earth Engine e o
  `geotessera` substituídos; a desquantização, as máscaras, lazy igual a em memória, o mosaico
  entre zonas, a média sobre células de 30 m, `depth=`, o índice baixado uma vez, a ida e volta
  pelo `save_raster`, as recusas, a similaridade, a mudança e seus eventos, a PCA e a CLI.

**Diferenças em relação ao plano:**

- `water=` saiu: na v1.1 a escala `NaN` não separa água de lacuna.
- `dequantize=False` saiu: guardar em int8 economiza disco, mas pede um formato próprio de
  ida e volta; o float32 comprimido do `save_raster` basta por enquanto.
- `resolution=` virou `res=`, como no `load_raster`; `store=` (cópias locais, e os testes)
  e `backend=` são novos.
- O clique num pixel do `zeit.plot` mostra a série dos três componentes, não a distância de
  ano a ano (seria um `fit=` novo).
- Pontos de treino: um cubo lazy lê os blocos inteiros sob os pontos (uma leitura de pontos
  pelo `sample_points` precisaria de outro caminho no `train_classifier`). Documentado: o
  mesmo cubo é classificado em seguida, então `.persist()` (ou `chunks=None`) lê uma vez.
- A recusa vale também para Mann-Kendall, `smooth`, `tmask` e `coded`.

## Para depois

- `zeit.plot`: medir de verdade o caso de notebook remoto (JupyterHub, Colab), que ficou
  como estimativa no 7a.
- Harmonizar Landsat e Sentinel-2 (ajuste de bandpass do HLS, Claverie et al. 2018) para
  séries densas que misturam os dois sensores: `zeit.harmonize(cube, to="landsat8")`, com
  o sensor de cada data lido do STAC ou de uma coordenada `platform`. Hoje um cubo misto
  tem um degrau entre sensores que vira quebra falsa no CCDC e no BFAST.
