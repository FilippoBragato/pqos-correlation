# PQoS using Correlation
The aim of this project is studying the correlation betweeen sensors data in order to perform predictive quality of service.

## Roadmap
1. Studio del dataset SELMA [deadline 10/03] 
2. Individuare possibili tecniche per analizzare la correlazione [deadline 24/03].
 In ordine crescente di difficoltà, complessità computazionale ed accuratezza:
    - "Voxelizzazione" della pointcloud
    - "Clusterizzazione" della pointcloud
    - "Segmentazione" della pointcloud
    - …
4. Implementare un sottoinsieme delle tecniche di correlazione individuate al punto precedente [deadline 21/04]
5. Studio della correlazione, ottenendo risultati numerici che possano essere poi usati per il framework PQoS [deadline 05/05]

## SELMA dataset study

The dataset provides 216 settings whose parameters are:
- 3 daytime (Night, Noon, Sunset)
- 9 weather conditions (Clear, Cloudy, HardFog, HardRain, MidFog, MidRain, SoftRain, Wet, WetCloudy)
- 8 towns

Each acquisition is made by:
- 7 rgb cameras ( $90\deg$ horizontal FoV, res $1280 \times 640$)
- 7 depth cameras ( $90\deg$ horizontal FoV, res $1280 \times 640$)
- 7 semantic cameras ( $90\deg$ horizontal FoV, res $1280 \times 640$)
- 3 semantic LiDARs ( $64\deg$ verical channels, $100 000$ points per second, range $100\mathsf{m}$)

The dataset provides the ground truth for $36$ distinct classes 

## Dependencies
    anytree                   2.8.0                    pypi_0    pypi
    asttokens                 2.2.1              pyhd8ed1ab_0    conda-forge      
    attrs                     22.2.0                   pypi_0    pypi
    autopep8                  1.6.0              pyhd3eb1b0_1
    backcall                  0.2.0              pyh9f0ad1d_0    conda-forge      
    backports                 1.0                pyhd8ed1ab_3    conda-forge      
    backports.functools_lru_cache 1.6.4              pyhd8ed1ab_0    conda-forge
    ca-certificates           2023.01.10           haa95532_0
    certifi                   2022.12.7        py38haa95532_0
    click                     8.1.3                    pypi_0    pypi
    colorama                  0.4.6              pyhd8ed1ab_0    conda-forge
    configargparse            1.5.3                    pypi_0    pypi
    contourpy                 1.0.7                    pypi_0    pypi
    cycler                    0.11.0                   pypi_0    pypi
    dash                      2.8.1                    pypi_0    pypi
    dash-core-components      2.0.0                    pypi_0    pypi
    dash-html-components      2.0.0                    pypi_0    pypi
    dash-table                5.0.0                    pypi_0    pypi
    debugpy                   1.5.1            py38hd77b12b_0
    decorator                 5.1.1              pyhd8ed1ab_0    conda-forge
    executing                 1.2.0              pyhd8ed1ab_0    conda-forge
    fastjsonschema            2.16.3                   pypi_0    pypi
    flask                     2.2.3                    pypi_0    pypi
    fonttools                 4.38.0                   pypi_0    pypi
    importlib-metadata        6.0.0              pyha770c72_0    conda-forge
    importlib-resources       5.12.0                   pypi_0    pypi
    importlib_metadata        6.0.0                hd8ed1ab_0    conda-forge
    ipykernel                 6.15.0             pyh025b116_0    conda-forge
    ipython                   8.10.0             pyh08f2357_0    conda-forge
    ipywidgets                8.0.4                    pypi_0    pypi
    itsdangerous              2.1.2                    pypi_0    pypi
    jedi                      0.18.2             pyhd8ed1ab_0    conda-forge
    jinja2                    3.1.2                    pypi_0    pypi
    jsonschema                4.17.3                   pypi_0    pypi
    jupyter_client            8.0.3              pyhd8ed1ab_0    conda-forge
    jupyter_core              5.2.0            py38haa244fe_0    conda-forge
    jupyterlab-widgets        3.0.5                    pypi_0    pypi
    kiwisolver                1.4.4                    pypi_0    pypi
    libffi                    3.4.2                hd77b12b_6
    libsodium                 1.0.18               h8d14728_1    conda-forge
    markupsafe                2.1.2                    pypi_0    pypi
    matplotlib                3.7.0                    pypi_0    pypi
    matplotlib-inline         0.1.6              pyhd8ed1ab_0    conda-forge
    nbformat                  5.5.0                    pypi_0    pypi
    nest-asyncio              1.5.6              pyhd8ed1ab_0    conda-forge
    numpy                     1.24.2                   pypi_0    pypi
    open3d                    0.16.0                   pypi_0    pypi
    openssl                   1.1.1t               h2bbff1b_0
    packaging                 23.0               pyhd8ed1ab_0    conda-forge
    pandas                    1.5.3                    pypi_0    pypi
    parso                     0.8.3              pyhd8ed1ab_0    conda-forge
    pickleshare               0.7.5                   py_1003    conda-forge
    pillow                    9.4.0                    pypi_0    pypi
    pip                       22.3.1           py38haa95532_0
    pkgutil-resolve-name      1.3.10                   pypi_0    pypi
    platformdirs              3.0.0              pyhd8ed1ab_0    conda-forge
    plotly                    5.13.1                   pypi_0    pypi
    plyfile                   0.7.4                    pypi_0    pypi
    prompt-toolkit            3.0.36             pyha770c72_0    conda-forge
    psutil                    5.9.0            py38h2bbff1b_0
    pure_eval                 0.2.2              pyhd8ed1ab_0    conda-forge
    pycodestyle               2.10.0           py38haa95532_0
    pygments                  2.14.0             pyhd8ed1ab_0    conda-forge
    pyparsing                 3.0.9                    pypi_0    pypi
    pyrsistent                0.19.3                   pypi_0    pypi
    python                    3.8.16               h6244533_2
    python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
    python_abi                3.8                      2_cp38    conda-forge
    pytz                      2022.7.1                 pypi_0    pypi
    pywin32                   227              py38h294d835_1    conda-forge
    pyzmq                     23.2.0           py38hd77b12b_0
    seaborn                   0.12.2                   pypi_0    pypi
    setuptools                65.6.3           py38haa95532_0
    six                       1.16.0             pyh6c4a22f_0    conda-forge
    sqlite                    3.40.1               h2bbff1b_0
    stack_data                0.6.2              pyhd8ed1ab_0    conda-forge
    tenacity                  8.2.2                    pypi_0    pypi
    toml                      0.10.2             pyhd3eb1b0_0
    tornado                   6.2              py38h294d835_0    conda-forge
    traitlets                 5.9.0              pyhd8ed1ab_0    conda-forge
    typing-extensions         4.4.0                hd8ed1ab_0    conda-forge
    typing_extensions         4.4.0              pyha770c72_0    conda-forge
    vc                        14.2                 h21ff451_1
    vs2015_runtime            14.27.29016          h5e58377_2
    wcwidth                   0.2.6              pyhd8ed1ab_0    conda-forge
    werkzeug                  2.2.3                    pypi_0    pypi
    wheel                     0.38.4           py38haa95532_0
    widgetsnbextension        4.0.5                    pypi_0    pypi
    wincertstore              0.2              py38haa95532_2
    zeromq                    4.3.4                h0e60522_1    conda-forge
    zipp                      3.15.0             pyhd8ed1ab_0    conda-forge