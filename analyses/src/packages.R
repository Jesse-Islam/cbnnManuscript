#code to install missing packages


getPckg <- function(pckgName) {install.packages(paste(pckgName), repos = "http://cran.r-project.org")}
getPckgBM <- function(pckgName) {BiocManager::install(paste(pckgName))}

pckgNamesCran = c("casebase",
                  "DT",
                  "data.table",
                  "flexsurv",
                  "ggplot2",
                  "keras",
                  "matrixStats",
                  "mltools",
                  "RColorBrewer",
                  "riskRegression",
                  "pseudo",
                  "simsurv",
                  "splines",
                  "survival",
                  "survminer",
                  "survivalmodels",
                  "tensorflow",
                  "visreg")

for(pckgName in pckgNamesCran){
  pckg = try(require(pckgName,character.only = TRUE))
  if(!pckg) {
    cat(paste("Installing '",pckgName, "' from CRAN\n",sep=""))
    getPckg(pckgName)
    require(pckgName,character.only = TRUE)
  }
}



#biomaRt from bioconductor can give us gene name to symbol transformations.
pckgNamesBioC<-c()#"TCGAbiolinks"

for(pckgName in pckgNamesBioC){
  pckg = try(require(pckgName, character.only = TRUE))
  if(!pckg) {
    cat(paste("Installing '",pckgName, "' from CRAN\n", sep=""))
    getPckgBM(pckgName)
    require(pckgName, character.only = TRUE)
  }
}






