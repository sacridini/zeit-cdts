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

## Fase 7: `zeit.plot` (7a feito: prova de conceito)

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
3. 7d: clicar num pixel e ver a série com o ajuste do algoritmo.
4. 7e: basemap e vetores; avaliar MapLibre GL (camada WebGL própria sobre o mapa) contra
   pan/zoom e tiles feitos à mão.
5. 7f: janela fora do notebook (pywebview ou navegador), testes (Playwright com Edge/Chromium
   em modo headless, como no 7a) e docs.

### Dependências

`anywidget` (obrigatória para o interativo, leve) e `matplotlib` (estático); `pywebview`
opcional para a janela nativa; `xyzservices` para nomear os basemaps. Tudo no extra
`pip install zeit-cdts[plot]`, sem pesar no `import zeit`.

## Para depois

- `zeit.landtrendr(..., ftv=[...])`: fitted-to-vertices de outras bandas com os vértices
  da banda de segmentação, como no LT-GEE.
- `load_raster(..., like=)`: reprojetar/reamostrar para a grade de outro raster.
- Resultados com `.save(pasta)` e `.plot()`, como os `Saveable` do landschaft.
