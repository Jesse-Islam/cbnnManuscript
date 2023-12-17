# cbnnManuscript

This repository contains the code required to replicate the analyses in the CBNN manuscript and the tex to generate the manuscript itself.

analyses/ contains the following pipelines:
1. optimFLchain.Rmd
2. optimizeComplex.Rmd
3. optimMyeloma.Rmd
4. optimProstate.Rmd
5. plotter.Rmd
Where pipelines 1-4 may be run in parallel should you have the system resources, or in sequence. pipeline 5 uses the stored results in figures similar to those generated in the CBNN manuscript.

manuscript/ contains the .tex to generate the manuscript.



This package is based on keras. As its R implementation has python-based dependencies, its important to follow https://tensorflow.rstudio.com/install/ and correctly install keras. Then, install the following packages:

If you are interested in *Using* CBNN, We recommend using the dedicated R package found at https://github.com/Jesse-Islam/cbnn

The vignette in the package provides a deeper explanation of what is happening at each step, dedicated to CBNN. Here, DeepHit and DeepSurv are appropriately described as well, should the user want to replicate analyses in the study. Specifically, I recommend going through the vignette (https://github.com/Jesse-Islam/cbnn/tree/main/doc/Time-varying-interactions-and-flexible-baseline-hazard.html) next, as it uses the complex simulation from this manuscript (analyses/optimComplex.Rmd), an example to better understand the implementation of CBNN on tabular data.


```

install.packages(c("casebase","data.table","mltools","survival","survminer","ggplot2","colorspace","DT","splines","reticulate","simsurv","flexsurv","survivalmodels","visreg","riskRegression",'pseudo',"ggrepel","dplyr","cowplot","matrixStats","tableone","gridExtra","grid","gtable","flextable"))
library(reticulate)
py_install(c("pycox", "torch", "torchtuples","scikit-learn", "matplotlib", "numpy","scikit-survival"))

```

After this installation, the user should be able to run any of the .Rmds. each optim*.Rmd may be run in parallel, with plotter.Rmd producing the figures in the paper with your results. Note that though we set the seed, its possible results may vary, as we're using both R and python to run these methods in unison. 

If you have any questions, feel free to open an issue in this repository and I'd be happy to help.

Note that we modified deepsurvivalmachines to permit early stopping and dropout (analyses/dsm), therefore, we keep a copy with said modifications in our repository, with all the licensing rules applying separately for said section of our repo https://github.com/autonlab/DeepSurvivalMachines.


