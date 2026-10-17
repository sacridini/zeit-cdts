# Choosing an Algorithm

<p class="lead">Zeit implements more than a dozen methods. Most projects need one or two. Start from the question you are trying to answer, and let it pick the method.</p>

## Start from your question

| Your question | Use | Needs |
| :--- | :--- | :--- |
| *When* was this area disturbed (cleared, burned) and how badly? | [LandTrendr](../tutorials/landtrendr.md) | One image per year |
| Did anything change, at any time of year? What did it change into? | [CCDC](../tutorials/ccdc.md) | Every clear image (dense, multi-band) |
| Is something changing *right now*, compared with a stable past? | [BFAST Monitor](../tutorials/bfast_monitor.md) | Regular series (e.g. 16-day composites) |
| How many changes happened, and when, in a regular series? | [BFAST Lite](../tutorials/bfast_lite.md) | Regular series |
| Did the trend change, the seasonal cycle, or both? | [BFAST](../tutorials/bfast.md) | Regular series |
| Is this area getting greener or browner over the decades? | [Mann-Kendall](../tutorials/mann_kendall.md) | One value per year (or `method="seasonal"`) |
| When does the growing season start, peak and end? Is this year late? | [Phenology](../tutorials/phenology.md) | Dense series, several per month |
| Which crop or land-cover type is this pixel, given a few examples? | [TWDTW](../tutorials/twdtw.md) | Reference time series per class |
| Which land-cover types exist here, with no labels at all? | [SOM](../tutorials/som.md) | Any cube |
| Can I work with fields and patches instead of pixels? | [SNIC](../tutorials/snic.md) | Any cube |
| I have many labelled samples and want the best classifier. | [TempCNN / LightTAE](../tutorials/ai.md) | Labelled time series |
| I have few labelled samples, gaps or several sensors. | [Presto](../tutorials/presto.md) (pretrained) | Monthly composites, up to 24 |
| I have two dates and labelled change masks. | [Siamese Change Detector](../tutorials/siamese.md) | Image pairs + masks |

## A quick decision guide

```mermaid
flowchart TD
    Q{"What do you want to know?"}
    Q -->|"When / where did it change?"| C{"How often is it observed?"}
    Q -->|"Is there a long-term trend?"| MK["Mann-Kendall"]
    Q -->|"What is it?"| L{"Do you have labels?"}
    Q -->|"When is the growing season?"| PH["Phenology"]

    C -->|"Once a year"| LT["LandTrendr"]
    C -->|"Every clear image"| CC["CCDC"]
    C -->|"Regular composites"| B{"Past or present?"}
    B -->|"Whole history"| BL["BFAST Lite / BFAST"]
    B -->|"New observations only"| BM["BFAST Monitor"]

    L -->|"A few examples per class"| TW["TWDTW / Presto"]
    L -->|"Many labelled samples"| DL["TempCNN / LightTAE / U-TAE"]
    L -->|"None"| SO["SOM clustering"]
```

## Change detection methods compared

The four change-detection families answer related but different questions. The figures below all come from their tutorials.

<div class="gallery" markdown>

<a class="tile" href="../../tutorials/landtrendr/">
  <img src="../../assets/figures/thumbs/landtrendr_pixels.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Annual · retrospective</span><span class="tile-title">LandTrendr</span><span class="tile-text">Straight segments through yearly values. Best for forest disturbance and recovery over decades.</span></span>
</a>

<a class="tile" href="../../tutorials/ccdc/">
  <img src="../../assets/figures/thumbs/ccdc.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Dense · retrospective</span><span class="tile-title">CCDC</span><span class="tile-text">Harmonic models of the seasonal cycle, multi-band. Dates changes within the year and describes the new state.</span></span>
</a>

<a class="tile" href="../../tutorials/bfast_lite/">
  <img src="../../assets/figures/thumbs/bfast_lite.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Regular · retrospective</span><span class="tile-title">BFAST Lite / BFAST</span><span class="tile-text">Statistically optimal number of breaks in trend and season. Single index.</span></span>
</a>

<a class="tile" href="../../tutorials/bfast_monitor/">
  <img src="../../assets/figures/thumbs/bfast_monitor.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Regular · near real time</span><span class="tile-title">BFAST Monitor</span><span class="tile-text">Is the newest data consistent with the stable past? The basis of many alert systems.</span></span>
</a>

</div>

| | LandTrendr | CCDC | BFAST Lite / BFAST | BFAST Monitor |
| :--- | :--- | :--- | :--- | :--- |
| Input | 1 index, 1 value per year | Up to 7 bands, every clear observation | 1 index, regular series | 1 index, regular series |
| Models the season? | No (annual values) | Yes (harmonics) | Yes (harmonics) | Yes (harmonics) |
| Detects | Abrupt and gradual change, recovery | Abrupt change, any time of year | Breaks in trend (and season, for BFAST) | The first break after a date |
| Typical use | Forest disturbance history | Land-cover change mapping and classification | Retrospective break analysis | Alerts |
| Speed per pixel | Very fast | Moderate | Moderate | Very fast |

!!! tip "Combine methods"
    The methods complement each other. A common pattern is Mann-Kendall to find *where* a region is trending, then LandTrendr or CCDC to find *when* and *how abruptly* it changed there. Another is Tmask to clean the data before CCDC. See the [User Guide](../tutorials/index.md) for every workflow.
