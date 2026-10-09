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

## Fase 8: grade de referência, FTV e resultados que se salvam

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

### 8b: `zeit.landtrendr(..., ftv=[...])`

Fitted-to-vertices como no LT-GEE: segmenta uma banda (`band=`) e ajusta as outras com os
mesmos anos de vértice. `ftv=["nbr", "tcw"]` (nomes de banda do cubo `(time, band, y, x)`
ou do `Dataset`) acrescenta ao resultado `ftv_<banda>` `(time, y, x)` com a série ajustada
e `vertex_value_<banda>` `(vertex, y, x)`. O ajuste é linear por partes e contínuo, com os
anos dos vértices fixos (como o `ftv_` do LT-GEE); o `apply_vertices` atual só interpola os
valores brutos nos anos dos vértices, o que não é a mesma coisa: conferir contra uma
exportação do LT-GEE antes de decidir se ele sai ou passa a chamar o novo. Em C++, junto
do motor do LandTrendr (um pixel, todas as bandas).

### 8c: resultados com `.save()` e `.plot()`

Os `Dataset`s de resultado ganham `.zeit.save(caminho)` e `.zeit.plot(...)` pelo accessor
que já existe (`xarray_api.py`), delegando para `save_raster` e `zeit.plot`, como os
`Saveable` do landschaft: `zeit.landtrendr(ndvi).zeit.save("lt")`. Também nos
`DataArray`s (`ndvi.zeit.plot()`).

## Fase 9: o resto da API no padrão do cubo

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
- Pode ser dividida (9a TWDTW e suavização, que são as que mais aparecem nos tutoriais;
  9b TMASK, SNIC e classificação).

## Para depois

- `load_raster(..., crs=, res=)`: reprojetar para um CRS ou resolução sem um raster de
  referência (o mesmo warp do 8a; `like=` cobre o caso mais comum).
- `zeit.plot`: medir de verdade o caso de notebook remoto (JupyterHub, Colab), que ficou
  como estimativa no 7a.
