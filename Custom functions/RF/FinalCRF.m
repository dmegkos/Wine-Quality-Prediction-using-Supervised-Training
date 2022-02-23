%Random Forest to be used for comparison
function [RF_F1_score, RF_TestAcc, Mdl1] = FinalCRF(X,Y,Xval,Yval,maxnsplits,minlfsize,numvarsam,predsel,numlcycl,viewtree)
    rng(0);%for reproducability
    %predictor names for the tree
    pnames = ["fa","va","ca","rs","ch","fsd","tsd","d","pH","slp","alc"];
    %create template tree to pass to ensemble
    t = templateTree("MaxNumSplits",maxnsplits,"MinLeafSize",minlfsize,"NumVariablesToSample",numvarsam,"PredictorSelection",predsel, ...
        "Prune","on","PruneCriterion","impurity","SplitCriterion","gdi");
    %create Random forest model (Method=Bag)
    Mdl1 = fitcensemble(X,Y,"Method","Bag","NumLearningCycles",numlcycl,"Learners",t,"PredictorNames",pnames);
    
    %Errors
    Testloss = loss(Mdl1,Xval,Yval); %get loss
    pred = predict(Mdl1,Xval); %get prediction
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
    
    RF_F1_score = 2 * ((overall_precision * overall_recall) / (overall_precision + overall_recall)); %get f1 score
    RF_TestAcc = 100*(1-Testloss); %get test accuracy
    %show random tree
    if viewtree
        %store random tree
        tree = Mdl1.Trained{randi(100)};
        %view tree
        view(tree,"Mode","graph")
    end
    %show confusion matrix
    cm = confusionchart(Yval,pred);
    %set title
    cm.Title = 'White Wine Quality Classification Using RF';
    %Add column and row summaries
    cm.RowSummary = 'row-normalized';
    %cm.ColumnSummary = 'column-normalized';
end

