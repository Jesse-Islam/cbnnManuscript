
statCalcSim<-function(metricSims){
  toRemove <- which(apply(metricSims, 1, function(x){any(is.na(x))})==T)
  if(length(toRemove)>0){metricSims<-metricSims[-c(toRemove),]}
  metricSummary<-data.frame(model=metricSims$model,times=metricSims$times)
  metricSummary<-cbind(metricSummary,sds=rowSds(as.matrix(metricSims[,-c(1,2)])),
                       means=rowMeans(as.matrix(metricSims[,-c(1,2)])))
  ci<-apply(as.matrix(metricSims[,-c(1,2)]), 1, function(x){mean(x)+c(-1.96,1.96)*sd(x)/sqrt(length(x))})
  rownames(ci)<-c('cilower95','ciupper95')
  pi<-apply(as.matrix(metricSims[,-c(1,2)]), 1, function(x){quantile(x,c(.025, .975))})
  rownames(pi)<-c('pilower95','piupper95')
  metricSummary<-cbind(metricSummary,t(ci),t(pi))
  return(metricSummary)
}

plotPerformance<-function(metricSummary,xlab="Follow-up-time",method='Metric',iterations='NA'){
  mainTitle<-paste(method,' : 95% C.I. over ', iterations, ' iterations',sep='')
  
  a<-ggplot(data=metricSummary, aes(x=times,y=means,col=model))+
    geom_line() +
    geom_ribbon(aes(ymin=cilower95,ymax=ciupper95,fill=model),linetype=0,alpha=0.3) +
    ggtitle(mainTitle) + 
    xlab(xlab) +
    ylab(method)
  if(method=="IPA"){
    a<-a+coord_cartesian()#ylim(-1,1))#
    return(a)
  }
  return(a+coord_cartesian())
}


pyProcess<-function(predictions,times=times,apr="constant"){
  interpolated <- apply(predictions, 1, function(x) approx(as.numeric(colnames(predictions)), y=x, method = apr, xout=times)$y)
  class(interpolated)<-c("tunnel",class(interpolated))
  interpolated<-1-interpolated
  interpolated[is.na(interpolated) & is.na(interpolated)>50] <- 0
  interpolated[is.na(interpolated) & is.na(interpolated)<50] <- 1
  return(t(interpolated))
}




calcCidx<-function(et_train,et_test,risk,times){
  #et_train <- et_train[order(et_train[,2]),]
  #et_test <- et_test[order(et_test[,2]),]
  #tims<-times
  tims<-head(times, -1)
  tims<-tail(tims, -1)
  cindexes<-matrix(NA,nrow=length(tims),ncol=2)
  cindexes[,1]<-tims
  
  for( i in 1:(length(tims))){#metabric needs to skip first few
    

    cindexes[i,2]<-cidx(et_train,et_test,risk[,i,drop=F])
  }
  return(cindexes)
}


cIndexSummary<-function(et_train, et_test,riskList,cScore){
  
  for (i in 1:length(riskList)){
    #print(i)
    cScore[,i+1]<-calcCidx(et_train=et_train,et_test=et_test,risk=riskList[[i]],
                           times = as.numeric(colnames(riskList[[1]])))[,2]
  }
  cScore<-cScore
  return(na.omit(cScore))
}




plotAbRisk<- function(annPreds,glmAbsRisk){
  
  
  resultPlots<-rbind(as.data.frame(melt(setDT(annPreds), id.vars = c("model","Times"), variable.name = "sex")),
                     as.data.frame(melt(setDT(glmAbsRisk), id.vars = c("model","Times"), variable.name = "sex"))
                     # ,kmPreds
  )
  
  myCols<-brewer.pal(6,"Dark2")
  myCols<-myCols[c(1,3,2,4,6,5)]
  resultPlots$groups <- paste(resultPlots$model,resultPlots$sex)
  ggplot(data = resultPlots, aes(x = Times, y = value*100,group=groups)) +
    geom_line(aes(#linetype=groups,
      color=groups,lty=groups),size=2)+
    xlab("Follow-up Time")+
    ylab("Probability of death (%)")+
    ggtitle("GLM vs. NN comparing screening to control")+
    scale_color_manual(values = myCols)+ 
    guides(fill=guide_legend(title="Model + group"))
  
}



plotHazard<- function(annHaz,glmHazard){
  resultPlots<-rbind(as.data.frame(melt(setDT(annHaz), id.vars = c("model","Times"), variable.name = "sex")),
                     as.data.frame(melt(setDT(glmHazard), id.vars = c("model","Times"), variable.name = "sex"))
  )
  myCols<-brewer.pal(6,"Dark2")
  myCols<-myCols[c(1,3,2,4,6,5)]
  resultPlots$groups <- paste(resultPlots$model,resultPlots$sex)
  ggplot(data = resultPlots, aes(x = Times, y = value,group=groups)) +
    geom_line(aes(color=groups),size=2)+
    xlab("Follow-up Time")+
    ylab("Probability of death (%)")+
    ggtitle("GLM vs. NN comparing screening to control")+
    scale_color_manual(values = myCols)+ 
    guides(fill=guide_legend(title="Model + group"))
}


















predictRisk.testing <- function(object, newdata, times, cause, ...) {
  if (!is.null(object$matrix.fit)) {
    #get all covariates excluding intercept and time
    coVars=colnames(object$originalData$x)
    #coVars is used in lines 44 and 50
    newdata=data.matrix(drop(subset(newdata, select=coVars)))
  }
  #browser()
  # if (missing(cause)) stop("Argument cause should be the event type for which we predict the absolute risk.")
  # the output of absoluteRisk is an array with dimension dependening on the length of the requested times:
  # case 1: the number of time points is 1
  #         dim(array) =  (length(time), NROW(newdata), number of causes in the data)
  if (length(times) == 1) {
    a <- casebase::absoluteRisk(object, newdata = newdata, time = times)
    p <- matrix(a, ncol = 1)
  } else {
    # case 2 a) zero is included in the number of time points
    if (0 %in% times) {
      # dim(array) =  (length(time)+1, NROW(newdata)+1, number of causes in the data)
      a <- casebase::absoluteRisk(object, newdata = newdata, time = times)
      p <- t(a)
    } else {
      # case 2 b) zero is not included in the number of time points (but the absoluteRisk function adds it)
      a <- casebase::absoluteRisk(object, newdata = newdata, time = times)
      ### we need to invert the plot because, by default, we get cumulative incidence
      #a[, -c(1)] <- 1 - a[, -c(1)]
      ### we remove time 0 for everyone, and remove the time column
      a <- a[-c(1), -c(1)] ### a[-c(1), ] to keep times column, but remove time 0 probabilities
      # now we transpose the matrix because in riskRegression we work with number of
      # observations in rows and time points in columns
      p <- t(a)
    }
  }
  p<-p[-1,,drop=F]
  if (NROW(p) != NROW(newdata) || NCOL(p) != length(times)) {
    stop(paste("\nPrediction matrix has wrong dimensions:\nRequested newdata x times: ", 
               NROW(newdata), " x ", length(times), "\nProvided prediction matrix: ", 
               NROW(p), " x ", NCOL(p), "\n\n", sep = ""))
  }
  p
}


predictRisk.tunnel <- function(object, newdata, times, cause, ...){
  class(object)<-class(object)[-1]
  return(object)
}






##########################################################
###CBNN
#########################################################


cbnnModel<-function(features, feature_input, feature_output, originalData=originalData,offset, timeVar,eventVar, ratio=100, compRisk=FALSE,censored.indicator=0,optimizer=optimizer_adam(learning_rate = 0.001,decay=10^-7)){
  if(class(originalData)[1]=="cbData"){
    data<-originalData
  }else{
    print("hi")
    data<-sampleCaseBase(originalData,time=timeVar,event=eventVar,ratio,compRisk)
    offset<-data[,ncol(data),drop=F]
    data<-data[,-ncol(data),drop=F]
  }
  
  
  offsetInput<- keras::layer_input(shape = c(1), name = 'offsetInput')
  mainOutput <- keras::layer_add(c(feature_output,offsetInput)) %>%
    keras::layer_activation(activation="sigmoid")
  
  model<-keras::keras_model(
    inputs = c(feature_input, offsetInput),
    outputs = c(mainOutput)
  )
  
  model%>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer, #optimizer_rmsprop(lr = 0.001)
    metrics = c('binary_accuracy')
  )
  features<-features[which(!(features %in% eventVar))]
  return(list(model=model,
              casebaseData=data,
              offset=offset,
              originalData=originalData,
              timeVar=timeVar,
              eventVar=eventVar,
              features=features
  ))
  
}



fitSmoothHaz<-function(cbnn,epochs=2000,batch_size=500,verbose=0,monitor="val_loss",val_split=0.2,min_delta=10^-7,patience=10,val){
  #cbnn$casebaseData
  
  offset<-as.matrix(cbnn$offset)
  #cbnn$casebaseData<-as.matrix(cbnn$casebaseData[,cbnn$features])
  #x_train<-as.matrix(cbnn$casebaseData)
  x_train<-as.matrix(cbnn$casebaseData[,cbnn$features])
  
  y_train<-as.matrix(cbnn$casebaseData[,c(cbnn$eventVar)])
  xTensor<-list(x_train, offset)
  resultOfFit<-cbnn$model %>% fit(
    x = xTensor,
    y = y_train,
    epochs = epochs,#30000,
    batch_size = batch_size,#54540,
    #validation_split = val_split,
    validation_data=val,
    shuffle=T,
    verbose=verbose,
    #class_weight=list("0"=1,"1"=1),
    callbacks=list(callback_early_stopping(
      monitor = monitor,
      min_delta = min_delta,
      patience = patience,
      verbose = verbose,
      mode = c("min"),
      baseline =NULL,#lossCutOff,
      restore_best_weights = F#,
      # validation_split = 0.2
    )))
  
  cbnn[[length(cbnn)+1]]<-resultOfFit
  names(cbnn)[length(cbnn)]<-"resultOfFit"
  cbnn[[length(cbnn)+1]]<-x_train
  names(cbnn)[length(cbnn)]<-"x_train"
  cbnn[[length(cbnn)+1]]<-y_train
  names(cbnn)[length(cbnn)]<-"y_train"
  
  return(cbnn)
  
}



fitSmoothHazSpline<-function(cbnn,epochs=2000,batch_size=500,verbose=0,monitor="val_loss",val_split=0.2,min_delta=10^-7,patience=10,val){
  #cbnn$casebaseData
  
  offset<-as.matrix(cbnn$offset)
  #cbnn$casebaseData<-as.matrix(cbnn$casebaseData[,cbnn$features])
  #x_train<-as.matrix(cbnn$casebaseData)
  #x_train<-cbnn$casebaseData[,-c(which(cbnn$features %in% cbnn$eventVar))]
  
  x_train<-cbnn$casebaseData[,c(cbnn$features)]
  x_train<-as.matrix(cbind(x_train,bs(x_train$time)))
  y_train<-as.matrix(cbnn$casebaseData[,cbnn$eventVar])
  xTensor<-list(x_train, offset)

  val[[1]][[1]]<-as.matrix(cbind(val[[1]][[1]][,c(cbnn$features)],bs(val[[1]][[1]][,ncol(val[[1]][[1]])])))
  
  resultOfFit<-cbnn$model %>% fit(
    x = xTensor,
    y = y_train,
    epochs = epochs,#30000,
    batch_size = batch_size,#54540,
    #validation_split = val_split,
    validation_data=val,
    shuffle=T,
    verbose=verbose,callbacks=list(callback_early_stopping(
      monitor = monitor,
      min_delta =min_delta,
      patience = patience,
      verbose = verbose,
      mode = c("auto"),
      baseline =NULL,#lossCutOff,
      restore_best_weights = F#,
      # validation_split = 0.2
    )))
  
  cbnn[[length(cbnn)+1]]<-resultOfFit
  names(cbnn)[length(cbnn)]<-"resultOfFit"
  cbnn[[length(cbnn)+1]]<-x_train
  names(cbnn)[length(cbnn)]<-"x_train"
  cbnn[[length(cbnn)+1]]<-y_train
  names(cbnn)[length(cbnn)]<-"y_train"
  
  return(cbnn)
  
}






aarSplines<-function(cbnn, times=times,x_test=x_test){
  #offset<-exp(offset)
  # browser()
  # which(colnames(cbnn$casebaseData) %in% cbnn$timeVar))
  x_test<-x_test[,c(cbnn$features),drop=F]
  
  results<- matrix(data=NA,nrow=length(times),ncol= nrow(x_test)+1)
  results[,1]<-times
  #seq along  seqlength to produce the sequence before hand, and set j to be logical
  for(i in 2:ncol(results)){
    j=i-1
    # this is a waste of time make this funciton more efficent
    #    tempX<-as.matrix(data.frame( do.call("rbind",
    #                                        replicate(length(times),
    #                                                 x_test[j,-c(which(colnames(x_test) %in% cbnn$timeVar))],
    #                                                simplify = FALSE)),ftime=times))
    tempX<-as.matrix(data.frame( do.call("rbind",
                                         replicate(length(times),
                                                   x_test[j,],
                                                   simplify = FALSE))))
    tempX[,c(which(colnames(x_test) == cbnn$timeVar))]<-times
    tempX<-cbind(tempX,bs(times))
    #tempa<-cbind(tempX[,1:3],tempX[,ncol(tempX)], tempX[,4:(ncol(tempX)-1)])
    #tempX<-tempa
    tempOffset<-as.matrix(data.frame(offset=rep(0,length(times))))
    #score = model %>% evaluate(tempX,tempY , batch_size=128)
    aTemp=cbnn$model%>% predict(list(tempX,tempOffset))
    #results[,i]<-(aTemp/(1-aTemp))*(1/offset)
    #results[,i]<-exp(log(aTemp/(1-aTemp))+offset) #hazard?
    #results[,i]<-1-exp(-cumsum(exp(log(aTemp/(1-aTemp))+offset))) #external offset
    results[,i]<-1-exp(-cumsum((aTemp/(1-aTemp))*(diff(times)[1]))) 
    #results[,i]<-1-exp(-cumsum(exp(aTemp)*(1/length(times))))
    #odds<-((aTemp)*(1/length(times)))
    #results[,i]<-(aTemp/(1-aTemp))
    #results[,i]<-1-exp(-cumsum( odds * offset  )*(1/length(times)) )
    #results[,i]<-1-exp(-cumsum((aTemp/(1-aTemp))*(1/offset)))
  }
  results<-as.data.frame(results)
  #results$model<-"NN"
  return(results)
}















hazardcbnn<-function(cbnn, times=times,x_test=x_test){
  #offset<-exp(offset)
  # which(colnames(cbnn$casebaseData) %in% cbnn$timeVar))
  
  x_test<-x_test[,c(cbnn$features),drop=F]
  
  results<- matrix(data=NA,nrow=length(times),ncol= nrow(x_test)+1)
  results[,1]<-times
  
  for(i in 2:ncol(results)){
    j=i-1
    
    tempX<-as.matrix(data.frame( do.call("rbind",
                                         replicate(length(times),
                                                   x_test[j,],
                                                   simplify = FALSE))))
    tempX[,c(which(colnames(x_test) == cbnn$timeVar))]<-times
    
    tempOffset<-as.matrix(data.frame(offset=rep(0,length(times))))
    #score = model %>% evaluate(tempX,tempY , batch_size=128)
    #results[,i]<-(aTemp/(1-aTemp))*(1/offset)
    #results[,i]<-exp(log(aTemp/(1-aTemp))+offset) #hazard?
    #results[,i]<-1-exp(-cumsum(exp(log(aTemp/(1-aTemp))+offset))) #external offset
    results[,i]<-cbnn$model%>% predict(list(tempX,tempOffset))
    #results[,i]<-1-exp(-cumsum(exp(aTemp)*(1/length(times))))
    #odds<-((aTemp)*(1/length(times)))
    #results[,i]<-(aTemp/(1-aTemp))
    #results[,i]<-1-exp(-cumsum( odds * offset  )*(1/length(times)) )
    #results[,i]<-1-exp(-cumsum((aTemp/(1-aTemp))*(1/offset)))
  }
  results<-as.data.frame(results)
  #results$model<-"NN"
  return(results)
}






aar<-function(fit, times=times,x_test=x_test){
  tempOffset<-as.matrix(data.frame(offset=rep(0,nrow(x_test))))
  x_test<-as.matrix(x_test[,c(fit$features),drop=F])
  timeColumn<-which(fit$timeVar== colnames(x_test))
  results<- matrix(data=NA,nrow=length(times),ncol= nrow(x_test)+1)
  results[,1]<-times
  for(i in 1:length(times)){
    x_test[,timeColumn]<-rep(times[i],nrow(x_test))
    results[i,-1]=fit$model%>% predict(list(x_test,tempOffset))
  }
  for (i in 2:ncol(results)){
    results[,i]=1-exp(-1*cumsum((results[,i]/(1-results[,i]))*(diff(times)[1])))
  }
  return(results)
}



##################
#Depreciated
##################

aarOld<-function(cbnn, times=times,x_test=x_test){

  x_test<-x_test[,c(cbnn$features),drop=F]
  
  results<- matrix(data=NA,nrow=length(times),ncol= nrow(x_test)+1)
  results[,1]<-times
  
  for(i in 2:ncol(results)){
    j=i-1
    tempX<-as.matrix(data.frame( do.call("rbind",
                                         replicate(length(times),
                                                   x_test[j,],
                                                   simplify = FALSE))))
    tempX[,c(which(colnames(x_test) == cbnn$timeVar))]<-times
    
    tempOffset<-as.matrix(data.frame(offset=rep(0,length(times))))
    
    aTemp=cbnn$model%>% predict(list(tempX,tempOffset))
    rm(tempX)
    
    results[,i]<-1-exp(-cumsum((aTemp/(1-aTemp))*(diff(times)[1]))) 
    
  }
  results<-as.data.frame(results)
  return(results)
}



normalizer<-function(data,means,sds,maxTime){
  normalized<-as.data.frame(data)
  for (i in 1:ncol(data)){
    if(length(unique(data[,i]))>1){
    normalized[,i]<-(data[,i]-means[i])/sds[i]
    }
  }
  normalized$status<-data$status

  normalized$time<-data$time/maxTime
  return(normalized)
}









###########################
## hyperparameter optimization routines
###########################


optimDeepSurv<-function(klist,lr,drpt,layer1,layer2,nBatch,actv){
  
  paramTrials<-as.data.frame(tidyr::crossing(lr=lr,
                                             drpt=drpt,
                                             layer1=layer1,
                                             layer2=layer2,
                                             nBatch=nBatch,
                                             actv=actv,
                                             valLoss=NA,
                                             runtime=NA))
  

  
  
  for(i in 1:nrow(paramTrials)){
    tempRes<-c()
    tempVal<-c()
    runTime<-c()
    
    for(data in klist){
      currentTime<-Sys.time() 
      train<-data$train
      val<-data$val
      maxTime<-max(train$time)
      sds<-sapply(train, function(x) (sd(x)))
      means<-sapply(train, function(x) (mean(x)))
      val<-normalizer(val,means,sds,maxTime)
      train<-normalizer(train,means,sds,maxTime)
      times<-seq(from=min(val$time),
                 to=max(val$time),
                 length.out = 22
      )
      
      
      times<-head(times, -1)
      times<-tail(times, -1)
      
      lr=paramTrials[i,]$lr
      drpt=paramTrials[i,]$drpt
      layer1=paramTrials[i,]$layer1
      layer2=paramTrials[i,]$layer2
      bsize=ceiling(nrow(train)/paramTrials[i,]$nBatch)
      actv=paramTrials[i,]$actv

      source_python('src/dsurv.py')
      
      ctrain<-train

      
      
      coxnnsurv=fitDeepSurv(ctrain,val,bsize,epochs=epo,valida=val,patience=patience,min_delta=min_delta,drpt=drpt,lay1=layer1,lay2=layer2,lr=lr,actv=actv)
      
      
      
      coxnnsurv<-t(coxnnsurv)
      cnnCleaned<-pyProcess(coxnnsurv,times=times)
      colnames(cnnCleaned)<-times
      py_run_string("del fitDeepSurv")
      rm(fitDeepSurv)
      
      miniee <- Score(list('mod'=cnnCleaned),
                      data =val, 
                      formula = Hist(time, status != 0) ~ 1, summary = c("risks","IPA","ibs"), 
                      se.fit = FALSE, metrics = "brier", contrasts = FALSE, times = times)$Brier$score
      
      
      tempVal<-append(tempVal,miniee$IBS[which(miniee$model=="mod" & miniee$times==max(miniee$times))])
      runTime<-append(runTime, Sys.time()-currentTime)
      currentTime<-Sys.time() 
      
    }
    
    
    paramTrials$valLoss[i]<-mean(tempVal)
    paramTrials$runtime[i]<-mean(runTime)
    
    print(paste("trial",i))
    print(sum(runTime))
    
    gc()
  }
  bestPerformance<-paramTrials[which.min(paramTrials$valLoss),]
  return(bestPerformance)
}





optimDeepHit<-function(klist,lr,drpt,layer1,layer2,nBatch,actv,alp){
  
  paramTrials<-as.data.frame(tidyr::crossing(lr=lr,
                                             drpt=drpt,
                                             layer1=layer1,
                                             layer2=layer2,
                                             nBatch=nBatch,
                                             actv=actv,
                                             alp=alp,
                                             valLoss=NA,
                                             runtime=NA))
  
  
  for(i in 1:nrow(paramTrials)){
    
    tempRes<-c()
    tempVal<-c()
    runTime<-c()
    
    for(data in klist){
      currentTime<-Sys.time() 
      train<-data$train
      val<-data$val
      maxTime<-max(train$time)
      sds<-sapply(train, function(x) (sd(x)))
      means<-sapply(train, function(x) (mean(x)))
      val<-normalizer(val,means,sds,maxTime)
      train<-normalizer(train,means,sds,maxTime)
      times<-seq(from=min(val$time),
                 to=max(val$time),
                 length.out = 22
      )
      
      times<-head(times, -1)
      times<-tail(times, -1)
      
      lr=paramTrials[i,]$lr
      drpt=paramTrials[i,]$drpt
      layer1=paramTrials[i,]$layer1
      layer2=paramTrials[i,]$layer2
      bsize=ceiling(nrow(train)/paramTrials[i,]$nBatch)
      actv=paramTrials[i,]$actv
      alp=paramTrials[i,]$alp
      
      source_python('src/deephitter.py')
      hitnnSurv=fitDeephit(train,val,bsize,epochs=epo,valida=val,patience=patience,min_delta=min_delta,drpt=drpt,lay1=layer1,lay2=layer2,lr=lr,actv=actv,alp=alp)
      hitnnSurv<-t(hitnnSurv)
      deephitCleaned<-pyProcess(hitnnSurv,times=times)
      colnames(deephitCleaned)<-times
      py_run_string("del fitDeephit")
      rm(fitDeephit)
      
      
      miniee <- Score(list('mod'=deephitCleaned),
                      data =val, 
                      formula = Hist(time, status != 0) ~ 1, summary = c("risks","IPA","ibs"), 
                      se.fit = FALSE, metrics = "brier", contrasts = FALSE, times = times)$Brier$score
      
      
      tempVal<-append(tempVal,miniee$IBS[which(miniee$model=="mod" & miniee$times==max(miniee$times))])
      runTime<-append(runTime, Sys.time()-currentTime)
      currentTime<-Sys.time() 
    }
    
    
    paramTrials$valLoss[i]<-mean(tempVal)
    paramTrials$runtime[i]<-mean(runTime)
    
    print(paste("trial",i))
    print(sum(runTime))
    gc()
  }
  bestPerformance<-paramTrials[which.min(paramTrials$valLoss),]
  return(bestPerformance)
}



optimCBNN<-function(klist,lr,drpt,layer1,layer2,nBatch,actv){
  
  
  
  
  
  paramTrials<-as.data.frame(tidyr::crossing(lr=lr,
                                             drpt=drpt,
                                             layer1=layer1,
                                             layer2=layer2,
                                             nBatch=nBatch,
                                             actv=actv,
                                             valLoss=NA,
                                             runtime=NA))
  for(i in 1:nrow(paramTrials)){
    tempRes<-c()
    tempVal<-c()
    runTime<-c()
    
    for(data in klist){
      
      
      
      train<-data$train
      val<-data$val
      maxTime<-max(train$time)
      sds<-sapply(train, function(x) (sd(x)))
      means<-sapply(train, function(x) (mean(x)))
      train<-normalizer(train,means,sds,maxTime)
      val<-normalizer(val,means,sds,maxTime)
      valTest<-val
      times<-seq(from=min(val$time),
                 to=max(val$time),
                 length.out = 22
      )
      
      times<-head(times, -1)
      times<-tail(times, -1)
      
      train<-sampleCaseBase(data = as.data.frame(train),time="time",event="status",ratio=100)
      valcb<-sampleCaseBase(data = as.data.frame(val),time="time",event="status",ratio=100)
      val<-list(list(as.matrix(valcb[,-c(ncol(valcb)-1,ncol(valcb))]),
                     as.matrix(valcb[,ncol(valcb),drop=F])
      ),
      as.matrix(valcb[,ncol(valcb)-1,drop=F])
      )
      val[[1]][[2]]<-rep(train$offset[1],nrow(val[[1]][[1]]))
      
      lr=paramTrials[i,]$lr
      drpt=paramTrials[i,]$drpt
      layer1=paramTrials[i,]$layer1
      layer2=paramTrials[i,]$layer2
      bsize=c(floor(nrow(train)/paramTrials[i,]$nBatch))
      actv=paramTrials[i,]$actv
      
      
      
      currentTime<-Sys.time() 
      
      covars_input<-layer_input(shape=c(length(colnames(train))-2),
                                name = 'main_input')
      
      covars_output<-covars_input%>% 
        layer_dense(units=layer1,use_bias = T,activation = actv)%>%
        layer_dropout(drpt)%>%
        layer_dense(units=layer2,use_bias = T,activation = actv)%>%
        layer_dropout(drpt)%>%
        layer_dense(units=1,use_bias = T)
      
      cbnn<-cbnnModel(features=colnames(train)[-ncol(train)],
                      feature_input = covars_input,
                      feature_output = covars_output,
                      originalData = train,
                      offset=train$offset,
                      timeVar = "time",
                      eventVar= "status",optimizer=optimizer_adam(learning_rate = lr)
      )
      
      fit<-fitSmoothHaz(cbnn,
                        epochs=epo,
                        batch_size=bsize,
                        verbose=0,
                        monitor="val_loss",
                        #val_split=0.2,
                        min_delta=min_delta,
                        patience=patience,val=val)
      annPreds<-aar(fit,
                    times=times,
                    x_test=valTest[,-c(ncol(valTest))]
      )
      
      rownames(annPreds)<-annPreds[,1]
      annProperpoly<- t(annPreds[,-1])
      class(annProperpoly)<-c("tunnel",class(annProperpoly)) 
      
      
      miniee <- Score(list('mod'=annProperpoly),
                      data =valTest, 
                      formula = Hist(time, status != 0) ~ 1, summary = c("risks","IPA","ibs"), 
                      se.fit = FALSE, metrics = "brier", contrasts = FALSE, times = times)$Brier$score
      
      
      tempVal<-append(tempVal,miniee$IBS[which(miniee$model=="mod" & miniee$times==max(miniee$times))])
      
      runTime<-append(runTime, Sys.time()-currentTime)
      currentTime<-Sys.time() 
      
      rm(cbnn,fit,covars_input,covars_output)
    }
    
    
    
    
    paramTrials$valLoss[i]<-mean(tempVal)
    paramTrials$runtime[i]<-mean(runTime)
    
    print(paste("trial",i,"out of ",nrow(paramTrials)))
    print(sum(runTime))
    gc()
  }
  bestPerformance<-paramTrials[which.min(paramTrials$valLoss),]
  return(bestPerformance)
}

