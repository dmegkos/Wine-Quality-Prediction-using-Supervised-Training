%K-Nearest Neighbour classifier to be used for comparison 
function [KNN_F1_score, KNN_TestAcc, Mdl2] = FinalCKNN(X,Y,Xval,Yval,neighnum,dist)
    rng(0); %for reproducability
    
    %fit and train model
    Mdl2 = fitcknn(X,Y,"NumNeighbors",neighnum,"Standardize",1,"Distance",dist);
    
    Testloss = loss(Mdl2,Xval,Yval); %get loss
    pred = predict(Mdl2,Xval); %get prediction
    cm = confusionmat(Yval,pred); %create confusion matrix
    
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
    KNN_TestAcc = 100*(1-Testloss); %get test accuracy
    %show confusion matrix
    cm = confusionchart(Yval,pred);
    %set title
    cm.Title = 'White Wine Quality Classification Using KNN';
    %Add column and row summaries
    cm.RowSummary = 'row-normalized';
    %cm.ColumnSummary = 'column-normalized';
end

