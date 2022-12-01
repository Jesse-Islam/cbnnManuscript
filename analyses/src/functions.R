
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
  mainTitle<-paste(method,' : 95% P.I. over ', iterations, ' iterations',sep='')
  
  a<-ggplot(data=metricSummary, aes(x=times,y=means,col=model))+
    geom_line() +
    geom_ribbon(aes(ymin=pilower95,ymax=piupper95,fill=model),linetype=0,alpha=0.3) +
    ggtitle(mainTitle) + 
    xlab(xlab) +
    ylab(method)
  if(method=="IPA"){
    a<-a+coord_cartesian()#ylim(-1,1))#
    return(a)
  }
  return(a+coord_cartesian())
}


pyProcess<-function(predictions,times=times){
  interpolated <- apply(predictions, 1, function(x) approx(as.numeric(colnames(predictions)), y=x, method = "linear", xout=times)$y)
  class(interpolated)<-c("tunnel",class(interpolated))
  interpolated<-1-interpolated
  interpolated[is.na(interpolated) & is.na(interpolated)>50] <- 0
  interpolated[is.na(interpolated) & is.na(interpolated)<50] <- 1
  return(t(interpolated))
}




calcCidx<-function(et_train,et_test,risk,times){
  #et_train <- et_train[order(et_train[,2]),]
  #et_test <- et_test[order(et_test[,2]),]
  
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
    cScore[,i+1]<-calcCidx(et_train=et_train,et_test=et_test,risk=riskList[[i]],
                           times = as.numeric(colnames(riskList[[i]])))[,2]
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
###PMNN
#########################################################


pmnnModel<-function(features, feature_input, feature_output, originalData=originalData,offset, timeVar,eventVar, ratio=100, compRisk=FALSE,censored.indicator=0,optimizer=optimizer_adam(learning_rate = 0.001,decay=10^-7)){
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



fitSmoothHaz<-function(pmnn,epochs=20000,batch_size=500,verbose=0,monitor="val_loss",val_split=0.2,min_delta=10^-7,patience=10){
  #pmnn$casebaseData
  
  offset<-as.matrix(pmnn$offset)
  #pmnn$casebaseData<-as.matrix(pmnn$casebaseData[,pmnn$features])
  #x_train<-as.matrix(pmnn$casebaseData)
  x_train<-as.matrix(pmnn$casebaseData[,pmnn$features])
  
  y_train<-as.matrix(pmnn$casebaseData[,c(pmnn$eventVar)])
  xTensor<-list(x_train, offset)
  resultOfFit<-pmnn$model %>% fit(
    x = xTensor,
    y = y_train,
    epochs = epochs,#30000,
    batch_size = batch_size,#54540,
    validation_split = val_split,
    shuffle=T,
    verbose=verbose,callbacks=list(callback_early_stopping(
      monitor = monitor,
      min_delta = min_delta,
      patience = patience,
      verbose = verbose,
      mode = c("auto"),
      baseline =NULL,#lossCutOff,
      restore_best_weights = T#,
      # validation_split = 0.2
    )))
  
  pmnn[[length(pmnn)+1]]<-resultOfFit
  names(pmnn)[length(pmnn)]<-"resultOfFit"
  pmnn[[length(pmnn)+1]]<-x_train
  names(pmnn)[length(pmnn)]<-"x_train"
  pmnn[[length(pmnn)+1]]<-y_train
  names(pmnn)[length(pmnn)]<-"y_train"
  
  return(pmnn)
  
}



fitSmoothHazSpline<-function(pmnn,epochs=20000,batch_size=500,verbose=0,monitor="val_loss",val_split=0.2,min_delta=10^-7,patience=10){
  #pmnn$casebaseData
  
  offset<-as.matrix(pmnn$offset)
  #pmnn$casebaseData<-as.matrix(pmnn$casebaseData[,pmnn$features])
  #x_train<-as.matrix(pmnn$casebaseData)
  #x_train<-pmnn$casebaseData[,-c(which(pmnn$features %in% pmnn$eventVar))]
  
  x_train<-pmnn$casebaseData[,c(pmnn$features)]
  x_train<-as.matrix(cbind(x_train,bs(x_train$time)))
  y_train<-as.matrix(pmnn$casebaseData[,pmnn$eventVar])
  xTensor<-list(x_train, offset)
  
  resultOfFit<-pmnn$model %>% fit(
    x = xTensor,
    y = y_train,
    epochs = epochs,#30000,
    batch_size = batch_size,#54540,
    validation_split = 0.2,
    shuffle=T,
    verbose=verbose,callbacks=list(callback_early_stopping(
      monitor = monitor,
      min_delta =min_delta,
      patience = patience,
      verbose = verbose,
      mode = c("auto"),
      baseline =NULL,#lossCutOff,
      restore_best_weights = T#,
      # validation_split = 0.2
    )))
  
  pmnn[[length(pmnn)+1]]<-resultOfFit
  names(pmnn)[length(pmnn)]<-"resultOfFit"
  pmnn[[length(pmnn)+1]]<-x_train
  names(pmnn)[length(pmnn)]<-"x_train"
  pmnn[[length(pmnn)+1]]<-y_train
  names(pmnn)[length(pmnn)]<-"y_train"
  
  return(pmnn)
  
}






aarSplines<-function(pmnn, times=times,x_test=x_test){
  #offset<-exp(offset)
  # browser()
  # which(colnames(pmnn$casebaseData) %in% pmnn$timeVar))
  x_test<-x_test[,c(pmnn$features),drop=F]
  
  results<- matrix(data=NA,nrow=length(times),ncol= nrow(x_test)+1)
  results[,1]<-times
  #seq along  seqlength to produce the sequence before hand, and set j to be logical
  for(i in 2:ncol(results)){
    j=i-1
    # this is a waste of time make this funciton more efficent
    #    tempX<-as.matrix(data.frame( do.call("rbind",
    #                                        replicate(length(times),
    #                                                 x_test[j,-c(which(colnames(x_test) %in% pmnn$timeVar))],
    #                                                simplify = FALSE)),ftime=times))
    tempX<-as.matrix(data.frame( do.call("rbind",
                                         replicate(length(times),
                                                   x_test[j,],
                                                   simplify = FALSE))))
    tempX[,c(which(colnames(x_test) == pmnn$timeVar))]<-times
    tempX<-cbind(tempX,bs(times))
    #tempa<-cbind(tempX[,1:3],tempX[,ncol(tempX)], tempX[,4:(ncol(tempX)-1)])
    #tempX<-tempa
    tempOffset<-as.matrix(data.frame(offset=rep(0,length(times))))
    #score = model %>% evaluate(tempX,tempY , batch_size=128)
    aTemp=pmnn$model%>% predict(list(tempX,tempOffset))
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















hazardpmnn<-function(pmnn, times=times,x_test=x_test){
  #offset<-exp(offset)
  # which(colnames(pmnn$casebaseData) %in% pmnn$timeVar))
  
  x_test<-x_test[,c(pmnn$features),drop=F]
  
  results<- matrix(data=NA,nrow=length(times),ncol= nrow(x_test)+1)
  results[,1]<-times
  
  for(i in 2:ncol(results)){
    j=i-1
    
    tempX<-as.matrix(data.frame( do.call("rbind",
                                         replicate(length(times),
                                                   x_test[j,],
                                                   simplify = FALSE))))
    tempX[,c(which(colnames(x_test) == pmnn$timeVar))]<-times
    
    tempOffset<-as.matrix(data.frame(offset=rep(0,length(times))))
    #score = model %>% evaluate(tempX,tempY , batch_size=128)
    #results[,i]<-(aTemp/(1-aTemp))*(1/offset)
    #results[,i]<-exp(log(aTemp/(1-aTemp))+offset) #hazard?
    #results[,i]<-1-exp(-cumsum(exp(log(aTemp/(1-aTemp))+offset))) #external offset
    results[,i]<-pmnn$model%>% predict(list(tempX,tempOffset))
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

aarOld<-function(pmnn, times=times,x_test=x_test){

  x_test<-x_test[,c(pmnn$features),drop=F]
  
  results<- matrix(data=NA,nrow=length(times),ncol= nrow(x_test)+1)
  results[,1]<-times
  
  for(i in 2:ncol(results)){
    j=i-1
    tempX<-as.matrix(data.frame( do.call("rbind",
                                         replicate(length(times),
                                                   x_test[j,],
                                                   simplify = FALSE))))
    tempX[,c(which(colnames(x_test) == pmnn$timeVar))]<-times
    
    tempOffset<-as.matrix(data.frame(offset=rep(0,length(times))))
    
    aTemp=pmnn$model%>% predict(list(tempX,tempOffset))
    rm(tempX)
    
    results[,i]<-1-exp(-cumsum((aTemp/(1-aTemp))*(diff(times)[1]))) 
    
  }
  results<-as.data.frame(results)
  return(results)
}