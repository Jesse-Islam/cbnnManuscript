#!/bin/bash

cd ~/pmnnProjects/pmnnPlayground/reviewedAnalysisPMNN

epo=2000
patience=10
iteration=100
layer1=50
layer2=50
layer3=25
layer4=25
drpt=0.5


#explaination of structure (overview)
#domain knowledge is too specific to be understood at a glance

params='list('"epo=$epo,patience=$patience,iteration=$iteration,layer1=$layer1,layer2=$layer2,layer3=$layer3,layer4=$layer4,drpt=$drpt"')'
metabric='library(rmarkdown); rmarkdown::render("metabricStudy.Rmd",'"params=$params)"
support='library(rmarkdown); rmarkdown::render("supportStudy.Rmd",'"params=$params)"
simple='library(rmarkdown); rmarkdown::render("simpleSim.Rmd",'"params=$params)"
complex='library(rmarkdown); rmarkdown::render("complexSim.Rmd",'"params=$params)"
oldmort='library(rmarkdown); rmarkdown::render("oldmortality.Rmd",'"params=$params)"
(trap 'kill 0' SIGINT; \
sleep 1 && eval `echo Rscript -e "'"{$simple}"'"` && bash ~/phoneNotification.sh simpleTest && sleep 9 && eval `echo Rscript -e "'"{$complex}"'"` && bash ~/phoneNotification.sh complexTest \ &
sleep 6 && eval `echo Rscript -e "'"{$support}"'"` && bash ~/phoneNotification.sh supportTest \ &
sleep 3 && eval `echo Rscript -e "'"{$metabric}"'"` && bash ~/phoneNotification.sh metabricTest \ &
sleep 6 && eval `echo Rscript -e "'"{$oldmort}"'"` && bash ~/phoneNotification.sh oldmortTest 
) 




Rscript -e '{library(rmarkdown); rmarkdown::render("oldmortality.Rmd",params=list(epo=2000,patience=10,iteration=2,layer1=200,layer2=20,layer3=25,layer4=25,drpt=0.1))}'