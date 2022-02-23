%custom Classification Ensemble function, using method Bag and learner
%Tree (Random Forest). Uses custom templateTree with hyperparameters
%Inputs: X,Y,MaxNumSplits,MinLeafSize,NumVariablesToSample,
%PredictorSelection, NumLearningCycles, viewtree (True or false)
%Outputs: F1 Score, KFoldAccuracy, view of a random tree if True
function [RF_F1_score, RF_KFoldAcc] = CustomRF(X,Y,maxnsplits,minlfsize,numvarsam,predsel,numlcycl,viewtree)
    rng(0);
    %partition using non Stratified K-Fold (K=10) to deal with class imbalance
    cv = cvpartition(Y,'KFold',10,'Stratify',false);
    %predictor names for the tree
    pnames = ["fa","va","ca","rs","ch","fsd","tsd","d","pH","slp","alc"];
    %create template tree to pass to ensemble
    t = templateTree("MaxNumSplits",maxnsplits,"MinLeafSize",minlfsize,"NumVariablesToSample",numvarsam,"PredictorSelection",predsel, ...
        "Prune","on","PruneCriterion","impurity","SplitCriterion","gdi");
    %create Random forest model (Method=Bag)
    Mdl = fitcensemble(X,Y,"CVPartition",cv,"Method","Bag","NumLearningCycles",numlcycl,"Learners",t,"PredictorNames",pnames);
    
    %Errors
    %check KFold loss
    KFoldError = kfoldLoss(Mdl);
    %calculace accuracy
    RF_KFoldAcc = 100 * (1-KFoldError);
    %generate confusion matrix
    pred = kfoldPredict(Mdl);
    cm = confusionmat(Y,pred);
    
    %calculate precision, recall and f1 score
    %inspired by https://youtu.be/5mVv2VocH2o
    cmt = cm'; %transpose the matrix
    diagonal = diag(cmt); %get diagonal of matrix
    
    sum_of_rows = sum(cmt,2); %get sum of rows
    sum_of_columns = sum(cmt,1); %get sum of columns
    
    precision = diagonal ./ sum_of_rows; %get precision
    overall_precision = mean(precision); %get overall precision
    
    recall = diagonal ./ sum_of_columns'; %get recall
    overall_recall = mean(recall); %get overall recall
    
    RF_F1_score = 2 * ((overall_precision * overall_recall) / (overall_precision + overall_recall)); %get f1 score
    
    if viewtree
        %store tree random tree
        tree = Mdl.Trained{randi(10),1}.Trained{randi(100)};
        %view tree
        view(tree,"Mode","graph")
    end
end

