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

## Fase 1: `load_raster` (0.27.0) — **Feito** (0.27.0)

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

## Fase 2: `save_raster` simétrico (0.28.0)

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
- Sem `print`; erros de georreferência viram exceção em vez de passar em silêncio.

## Fase 3: `zeit.landtrendr` (0.29.0)

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

## Fase 4: `zeit.ccdc` (0.30.0)

Mesmo padrão para o CCDC/COLD: `zeit.ccdc(data, *, qa=None, dates=None, ...)` aceita o
cubo `(time, band, y, x)` (e `qa` como cubo, banda do próprio cubo ou nome), tira as
datas ordinais de `time` e devolve um `Dataset` georreferenciado por segmento.
`predict_synthetic_image` passa a aceitar esse resultado. `run_ccdc_image`,
`run_ccdc_array` e o accessor viram internos/delegam. O módulo `zeit/ccdc.py` vira
`zeit/_ccdc.py`.

## Fase 5: BFAST, Mann-Kendall e fenologia (0.31.0)

`zeit.bfast`, `zeit.bfast_lite`, `zeit.bfast_monitor`, `zeit.mann_kendall` e
`zeit.phenology` no mesmo padrão: qualquer entrada, tempo tirado do cubo (`start_time` e
`frequency` inferidos de `time` quando regulares), saída `Dataset` georreferenciado com
as métricas nomeadas. Os `run_*_image` e métodos do accessor viram atalhos para as
funções novas. Módulos `bfast.py` e `phenology.py` viram `_bfast.py` e `_phenology.py`.

## Fase 6: arrumação final (0.32.0)

- README, quickstart, conceitos e tutoriais reescritos no fluxo carregar, rodar,
  exportar.
- `examples/` atualizados; CLI usando as funções novas.
- Lista de mudanças de API (o que saiu, o que substitui) na documentação.

## Para depois

- `zeit.landtrendr(..., ftv=[...])`: fitted-to-vertices de outras bandas com os vértices
  da banda de segmentação, como no LT-GEE.
- `load_raster(..., like=)`: reprojetar/reamostrar para a grade de outro raster.
- Resultados com `.save(pasta)` e `.plot()`, como os `Saveable` do landschaft.
