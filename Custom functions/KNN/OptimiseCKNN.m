%Optimise KNN. 
%Inputs: X,Y,neighnum,dist
%Outputs: F1 Score, KFold Accuracy, best combination of K and Distance
%Create a for loop for different K Neighbors and distance functions and 
%call a custom KNN function that takes as input X,Y,number of
%neightbors and distance function to be used, performs 10 fold cross validation
%and returns F1 score and Accuracy
function [KNN_TopF1_score, KNN_TopKFoldAcc, BestKN, BestDist] = OptimiseCKNN(X,Y,neighnum,dist)
    rng(0);
    KNN_TopF1_score = 0; %best F1 Score
    KNN_TopKFoldAcc = 0; %best KFold Accuracy
    BestKN = 0; %best K-Neighbour
    BestDist = 0; %best Distance Function
    %check all given K values in array
    for i=neighnum
        %check all given distance functions in array
        for j=dist
            %call CustomKNN with i and j
            [f1_score, KFoldAcc] = CustomKNN(X,Y,i,j);
            if f1_score > KNN_TopF1_score
                KNN_TopF1_score = f1_score;
                KNN_TopKFoldAcc = KFoldAcc;
                BestDist = j;
                BestKN = i;
            end
        end
    end
end

