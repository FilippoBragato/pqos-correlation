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

# Observations
- when increasing the size of voxels the correlation increases
- cumulative voxels do not lead to sensible results in the correlation domain

# Dependencies
    # Name                    Version                   Build  Channel
    _libgcc_mutex             0.1                 conda_forge    conda-forge
    _openmp_mutex             4.5                       2_gnu    conda-forge
    _tflow_select             2.3.0                       mkl  
    abseil-cpp                20211102.0           h27087fc_1    conda-forge
    absl-py                   1.4.0              pyhd8ed1ab_0    conda-forge
    addict                    2.4.0                    pypi_0    pypi
    aiohttp                   3.8.4           py310h1fa729e_0    conda-forge
    aiosignal                 1.3.1              pyhd8ed1ab_0    conda-forge
    anyio                     3.6.2              pyhd8ed1ab_0    conda-forge
    argon2-cffi               21.3.0             pyhd8ed1ab_0    conda-forge
    argon2-cffi-bindings      21.2.0          py310h5764c6d_2    conda-forge
    asttokens                 2.2.1              pyhd8ed1ab_0    conda-forge
    astunparse                1.6.3              pyhd8ed1ab_0    conda-forge
    async-timeout             4.0.2              pyhd8ed1ab_0    conda-forge
    attrs                     22.2.0             pyh71513ae_0    conda-forge
    autopep8                  1.6.0              pyhd3eb1b0_1  
    backcall                  0.2.0              pyh9f0ad1d_0    conda-forge
    backports                 1.0                pyhd8ed1ab_3    conda-forge
    backports.functools_lru_cache 1.6.4              pyhd8ed1ab_0    conda-forge
    beautifulsoup4            4.11.2             pyha770c72_0    conda-forge
    binutils_impl_linux-64    2.40                 hf600244_0    conda-forge
    blas                      1.1                    openblas    conda-forge
    bleach                    6.0.0              pyhd8ed1ab_0    conda-forge
    blinker                   1.5                pyhd8ed1ab_0    conda-forge
    bottleneck                1.3.5           py310ha9d4c09_0  
    brotli                    1.0.9                h166bdaf_7    conda-forge
    brotli-bin                1.0.9                h166bdaf_7    conda-forge
    brotlipy                  0.7.0           py310h5764c6d_1005    conda-forge
    bzip2                     1.0.8                h7b6447c_0  
    c-ares                    1.18.1               h7f98852_0    conda-forge
    ca-certificates           2023.01.10           h06a4308_0  
    cachetools                5.3.0              pyhd8ed1ab_0    conda-forge
    certifi                   2022.12.7       py310h06a4308_0  
    cffi                      1.15.0          py310h0fdd8cc_0    conda-forge
    charset-normalizer        2.1.1              pyhd8ed1ab_0    conda-forge
    click                     8.1.3           unix_pyhd8ed1ab_2    conda-forge
    configargparse            1.5.3                    pypi_0    pypi
    contourpy                 1.0.5           py310hdb19cb5_0  
    cryptography              38.0.4          py310h597c629_0    conda-forge
    cycler                    0.11.0             pyhd8ed1ab_0    conda-forge
    dash                      2.8.1                    pypi_0    pypi
    dash-core-components      2.0.0                    pypi_0    pypi
    dash-html-components      2.0.0                    pypi_0    pypi
    dash-table                5.0.0                    pypi_0    pypi
    dbus                      1.13.18              hb2f20db_0  
    debugpy                   1.5.1           py310h295c915_0  
    decorator                 5.1.1              pyhd8ed1ab_0    conda-forge
    defusedxml                0.7.1              pyhd8ed1ab_0    conda-forge
    entrypoints               0.4                pyhd8ed1ab_0    conda-forge
    executing                 1.2.0              pyhd8ed1ab_0    conda-forge
    expat                     2.2.10               h9c3ff4c_0    conda-forge
    flask                     2.2.3                    pypi_0    pypi
    flatbuffers               22.12.06             hcb278e6_2    conda-forge
    flit-core                 3.8.0              pyhd8ed1ab_0    conda-forge
    fontconfig                2.14.1               hef1e5e3_0  
    fonttools                 4.25.0             pyhd3eb1b0_0  
    freetype                  2.10.4               h0708190_1    conda-forge
    frozenlist                1.3.3           py310h5764c6d_0    conda-forge
    gast                      0.4.0              pyh9f0ad1d_0    conda-forge
    gcc                       12.2.0              h26027b1_11    conda-forge
    gcc_impl_linux-64         12.2.0              hcc96c02_19    conda-forge
    giflib                    5.2.1                h36c2ea0_2    conda-forge
    glib                      2.69.1               he621ea3_2  
    google-auth               2.16.2             pyh1a96a4e_0    conda-forge
    google-auth-oauthlib      0.4.6              pyhd8ed1ab_0    conda-forge
    google-pasta              0.2.0              pyh8c360ce_0    conda-forge
    grpc-cpp                  1.46.1               h33aed49_1  
    grpcio                    1.42.0          py310hce63b2e_0  
    gst-plugins-base          1.14.1               h6a678d5_1  
    gstreamer                 1.14.1               h5eee18b_1  
    h5py                      3.7.0           py310he06866b_0  
    hdf5                      1.10.6               h3ffc7dd_1  
    icu                       58.2              hf484d3e_1000    conda-forge
    idna                      3.4                pyhd8ed1ab_0    conda-forge
    importlib-metadata        6.0.0              pyha770c72_0    conda-forge
    importlib_resources       5.12.0             pyhd8ed1ab_0    conda-forge
    ipykernel                 6.15.0             pyh210e3f2_0    conda-forge
    ipython                   8.11.0             pyh41d4057_0    conda-forge
    ipython_genutils          0.2.0                      py_1    conda-forge
    ipywidgets                8.0.4              pyhd8ed1ab_0    conda-forge
    itsdangerous              2.1.2                    pypi_0    pypi
    jedi                      0.18.2             pyhd8ed1ab_0    conda-forge
    jinja2                    3.1.2              pyhd8ed1ab_1    conda-forge
    joblib                    1.2.0                    pypi_0    pypi
    jpeg                      9e                   h166bdaf_1    conda-forge
    jsonschema                4.17.3             pyhd8ed1ab_0    conda-forge
    jupyter                   1.0.0           py310hff52083_8    conda-forge
    jupyter_client            7.3.4              pyhd8ed1ab_0    conda-forge
    jupyter_console           6.6.3              pyhd8ed1ab_0    conda-forge
    jupyter_core              5.2.0           py310hff52083_0    conda-forge
    jupyter_server            1.23.6             pyhd8ed1ab_0    conda-forge
    jupyterlab_pygments       0.2.2              pyhd8ed1ab_0    conda-forge
    jupyterlab_widgets        3.0.5              pyhd8ed1ab_0    conda-forge
    keras                     2.10.0          py310h06a4308_0  
    keras-preprocessing       1.1.2              pyhd8ed1ab_0    conda-forge
    kernel-headers_linux-64   2.6.32              he073ed8_15    conda-forge
    keyutils                  1.6.1                h166bdaf_0    conda-forge
    kiwisolver                1.4.4           py310h6a678d5_0  
    krb5                      1.19.3               h3790be6_0    conda-forge
    lcms2                     2.12                 h3be6417_0  
    ld_impl_linux-64          2.40                 h41732ed_0    conda-forge
    lerc                      3.0                  h295c915_0  
    libblas                   3.9.0           15_linux64_openblas    conda-forge
    libbrotlicommon           1.0.9                h166bdaf_7    conda-forge
    libbrotlidec              1.0.9                h166bdaf_7    conda-forge
    libbrotlienc              1.0.9                h166bdaf_7    conda-forge
    libcblas                  3.9.0           15_linux64_openblas    conda-forge
    libclang                  10.0.1          default_hb85057a_2  
    libcurl                   7.87.0               h91b91d3_0  
    libdeflate                1.17                 h5eee18b_0  
    libedit                   3.1.20191231         he28a2e2_2    conda-forge
    libev                     4.33                 h516909a_1    conda-forge
    libevent                  2.1.12               h8f2d780_0  
    libffi                    3.4.2                h6a678d5_6  
    libgcc-devel_linux-64     12.2.0              h3b97bd3_19    conda-forge
    libgcc-ng                 12.2.0              h65d4601_19    conda-forge
    libgfortran-ng            12.2.0              h69a702a_19    conda-forge
    libgfortran5              12.2.0              h337968e_19    conda-forge
    libgomp                   12.2.0              h65d4601_19    conda-forge
    liblapack                 3.9.0           15_linux64_openblas    conda-forge
    libllvm10                 10.0.1               he513fc3_3    conda-forge
    libllvm11                 11.1.0               hf817b99_2    conda-forge
    libnghttp2                1.46.0               hce63b2e_0  
    libopenblas               0.3.20          pthreads_h78a6416_0    conda-forge
    libpng                    1.6.39               h5eee18b_0  
    libpq                     12.9                 h16c4e8d_3  
    libprotobuf               3.20.3               he621ea3_0  
    libsanitizer              12.2.0              h46fd767_19    conda-forge
    libsodium                 1.0.18               h36c2ea0_1    conda-forge
    libssh2                   1.10.0               ha56f1ee_2    conda-forge
    libstdcxx-ng              12.2.0              h46fd767_19    conda-forge
    libtiff                   4.5.0                h6a678d5_2  
    libuuid                   1.41.5               h5eee18b_0  
    libwebp                   1.2.4                h11a3e52_1  
    libwebp-base              1.2.4                h5eee18b_1  
    libxcb                    1.15                 h7f8727e_0  
    libxkbcommon              1.0.1                hfa300c1_0  
    libxml2                   2.9.14               h74e7548_0  
    libxslt                   1.1.35               h4e12654_0  
    llvmlite                  0.39.1          py310he621ea3_0  
    lz4-c                     1.9.3                h9c3ff4c_1    conda-forge
    markdown                  3.4.1              pyhd8ed1ab_0    conda-forge
    markupsafe                2.1.1           py310h7f8727e_0  
    matplotlib                3.7.0           py310h06a4308_0  
    matplotlib-base           3.7.0           py310h1128e8f_0  
    matplotlib-inline         0.1.6              pyhd8ed1ab_0    conda-forge
    mistune                   2.0.5              pyhd8ed1ab_0    conda-forge
    multidict                 6.0.4           py310h1fa729e_0    conda-forge
    munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
    nbclassic                 0.5.3              pyhb4ecaf3_3    conda-forge
    nbclient                  0.7.2              pyhd8ed1ab_0    conda-forge
    nbconvert                 7.2.9              pyhd8ed1ab_0    conda-forge
    nbconvert-core            7.2.9              pyhd8ed1ab_0    conda-forge
    nbconvert-pandoc          7.2.9              pyhd8ed1ab_0    conda-forge
    nbformat                  5.5.0                    pypi_0    pypi
    ncurses                   6.4                  h6a678d5_0  
    nest-asyncio              1.5.6              pyhd8ed1ab_0    conda-forge
    notebook                  6.5.3              pyha770c72_0    conda-forge
    notebook-shim             0.2.2              pyhd8ed1ab_0    conda-forge
    nspr                      4.33                 h295c915_0  
    nss                       3.74                 h0370c37_0  
    numba                     0.56.4          py310ha5257ce_0    conda-forge
    numexpr                   2.8.4           py310h757a811_0  
    numpy                     1.22.3          py310h4ef5377_2    conda-forge
    oauthlib                  3.2.2              pyhd8ed1ab_0    conda-forge
    open3d                    0.16.0                   pypi_0    pypi
    openblas                  0.3.20          pthreads_h320a7e8_0    conda-forge
    openssl                   1.1.1t               h7f8727e_0  
    opt_einsum                3.3.0              pyhd8ed1ab_1    conda-forge
    packaging                 23.0               pyhd8ed1ab_0    conda-forge
    pandas                    1.5.3                    pypi_0    pypi
    pandoc                    2.19.2               ha770c72_0    conda-forge
    pandocfilters             1.5.0              pyhd8ed1ab_0    conda-forge
    parso                     0.8.3              pyhd8ed1ab_0    conda-forge
    patsy                     0.5.3              pyhd8ed1ab_0    conda-forge
    pcre                      8.45                 h9c3ff4c_0    conda-forge
    pexpect                   4.8.0              pyh1a96a4e_2    conda-forge
    pickleshare               0.7.5                   py_1003    conda-forge
    pillow                    9.4.0           py310h6a678d5_0  
    pip                       23.0.1          py310h06a4308_0  
    pkgutil-resolve-name      1.3.10             pyhd8ed1ab_0    conda-forge
    platformdirs              3.1.0              pyhd8ed1ab_0    conda-forge
    plotly                    5.13.1                   pypi_0    pypi
    ply                       3.11                       py_1    conda-forge
    plyfile                   0.7.4              pyhd8ed1ab_0    conda-forge
    prometheus_client         0.16.0             pyhd8ed1ab_0    conda-forge
    prompt-toolkit            3.0.38             pyha770c72_0    conda-forge
    prompt_toolkit            3.0.38               hd8ed1ab_0    conda-forge
    protobuf                  3.20.3          py310h6a678d5_0  
    psutil                    5.9.0           py310h5eee18b_0  
    ptyprocess                0.7.0              pyhd3deb0d_0    conda-forge
    pure_eval                 0.2.2              pyhd8ed1ab_0    conda-forge
    pyasn1                    0.4.8                      py_0    conda-forge
    pyasn1-modules            0.2.7                      py_0    conda-forge
    pycodestyle               2.10.0          py310h06a4308_0  
    pycparser                 2.21               pyhd8ed1ab_0    conda-forge
    pygments                  2.14.0             pyhd8ed1ab_0    conda-forge
    pyjwt                     2.6.0              pyhd8ed1ab_0    conda-forge
    pyopenssl                 23.0.0             pyhd8ed1ab_0    conda-forge
    pyparsing                 3.0.9              pyhd8ed1ab_0    conda-forge
    pyqt                      5.15.7          py310h6a678d5_1  
    pyqt5-sip                 12.11.0                  pypi_0    pypi
    pyquaternion              0.9.9                    pypi_0    pypi
    pyrsistent                0.18.0          py310h7f8727e_0  
    pysocks                   1.7.1              pyha2e5f31_6    conda-forge
    python                    3.10.9               h7a1cb2a_2  
    python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
    python-fastjsonschema     2.16.3             pyhd8ed1ab_0    conda-forge
    python-flatbuffers        23.1.21            pyhd8ed1ab_0    conda-forge
    python_abi                3.10                    2_cp310    conda-forge
    pytz                      2022.7.1           pyhd8ed1ab_0    conda-forge
    pyu2f                     0.1.5              pyhd8ed1ab_0    conda-forge
    pyyaml                    6.0                      pypi_0    pypi
    pyzmq                     23.2.0          py310h6a678d5_0  
    qt-main                   5.15.2               h327a75a_7  
    qt-webengine              5.15.9               hd2b0992_4  
    qtconsole                 5.4.0              pyhd8ed1ab_0    conda-forge
    qtconsole-base            5.4.0              pyha770c72_0    conda-forge
    qtpy                      2.3.0              pyhd8ed1ab_0    conda-forge
    qtwebkit                  5.212                h4eab89a_4  
    re2                       2022.04.01           h27087fc_0    conda-forge
    readline                  8.2                  h5eee18b_0  
    requests                  2.28.2             pyhd8ed1ab_0    conda-forge
    requests-oauthlib         1.3.1              pyhd8ed1ab_0    conda-forge
    rsa                       4.9                pyhd8ed1ab_0    conda-forge
    scikit-learn              1.2.2                    pypi_0    pypi
    scipy                     1.8.1           py310h7612f91_0    conda-forge
    seaborn                   0.12.2               hd8ed1ab_0    conda-forge
    seaborn-base              0.12.2             pyhd8ed1ab_0    conda-forge
    send2trash                1.8.0              pyhd8ed1ab_0    conda-forge
    setuptools                65.6.3          py310h06a4308_0  
    sip                       6.6.2           py310h6a678d5_0  
    six                       1.16.0             pyh6c4a22f_0    conda-forge
    snappy                    1.1.10               h9fff704_0    conda-forge
    sniffio                   1.3.0              pyhd8ed1ab_0    conda-forge
    soupsieve                 2.3.2.post1        pyhd8ed1ab_0    conda-forge
    sparse                    0.14.0             pyhd8ed1ab_0    conda-forge
    sqlite                    3.40.1               h5082296_0  
    stack_data                0.6.2              pyhd8ed1ab_0    conda-forge
    statsmodels               0.13.5          py310ha9d4c09_1  
    sysroot_linux-64          2.12                he073ed8_15    conda-forge
    tenacity                  8.2.2                    pypi_0    pypi
    tensorboard               2.10.0          py310h06a4308_0  
    tensorboard-data-server   0.6.1           py310h52d8a92_0  
    tensorboard-plugin-wit    1.8.1              pyhd8ed1ab_0    conda-forge
    tensorflow                2.10.0          mkl_py310h24f4fea_0  
    tensorflow-base           2.10.0          mkl_py310hb9daa73_0  
    tensorflow-estimator      2.10.0          py310h06a4308_0  
    termcolor                 2.2.0              pyhd8ed1ab_0    conda-forge
    terminado                 0.17.1             pyh41d4057_0    conda-forge
    threadpoolctl             3.1.0                    pypi_0    pypi
    tinycss2                  1.2.1              pyhd8ed1ab_0    conda-forge
    tk                        8.6.12               h1ccaba5_0  
    toml                      0.10.2             pyhd8ed1ab_0    conda-forge
    tornado                   6.1             py310h5764c6d_3    conda-forge
    tqdm                      4.65.0                   pypi_0    pypi
    traitlets                 5.9.0              pyhd8ed1ab_0    conda-forge
    typing-extensions         4.4.0                hd8ed1ab_0    conda-forge
    typing_extensions         4.4.0              pyha770c72_0    conda-forge
    tzdata                    2022g                h04d1e81_0  
    urllib3                   1.26.15            pyhd8ed1ab_0    conda-forge
    wcwidth                   0.2.6              pyhd8ed1ab_0    conda-forge
    webencodings              0.5.1                      py_1    conda-forge
    websocket-client          1.5.1              pyhd8ed1ab_0    conda-forge
    werkzeug                  2.2.3              pyhd8ed1ab_0    conda-forge
    wheel                     0.38.4          py310h06a4308_0  
    widgetsnbextension        4.0.5              pyhd8ed1ab_0    conda-forge
    wrapt                     1.15.0          py310h1fa729e_0    conda-forge
    xz                        5.2.10               h5eee18b_1  
    yarl                      1.8.2           py310h5764c6d_0    conda-forge
    zeromq                    4.3.4                h9c3ff4c_1    conda-forge
    zipp                      3.15.0             pyhd8ed1ab_0    conda-forge
    zlib                      1.2.13               h5eee18b_0  
    zstd                      1.5.2                ha4553b6_0  
