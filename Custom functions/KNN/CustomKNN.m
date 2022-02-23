%custom KNN function 
%Inputs: X,Y,number of neightbors, distance function
%Outputs: F1 Score, KFoldAccuracy
function [KNN_F1_score, KNN_KFoldAcc] = CustomKNN(X,Y,neighnum,dist)
    rng(0);
    %K-Nearest Neighbour Algorithm
    %partition using Stratified K-Fold (K=10) to deal with class imbalance
    cv = cvpartition(Y,'KFold',10);
    %K-Nearest Neighboor with K passed as a parameter, with standardization
    Mdl = fitcknn(X,Y,'CVPartition',cv,'NumNeighbors',neighnum,'Standardize',1,'Distance',dist);
    
    %Calculate Errors
    %check KFold loss
    KFoldError = kfoldLoss(Mdl);
    %calculace accuracy
    KNN_KFoldAcc = 100 * (1-KFoldError);
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
    
    KNN_F1_score = 2 * ((overall_precision * overall_recall) / (overall_precision + overall_recall)); %get f1 score
end