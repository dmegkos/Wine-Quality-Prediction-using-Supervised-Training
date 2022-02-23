%Optimise RF.
%Inputs: X,Y,MaxNumSplits,MinLeafSize,NumVariablesToSample,PredictorSelection,NumLearningCycles
%Outputs: F1 Score, KFold Accuracy, best combination of MaxNumSplits,
%MinLeafSize,NumVariablesToSample,PredictorSelection,NumLearningCycles
%Create for loops for different hyperparameters
%call a custom RF function that takes as input X,Y and hyperparameters
%to be used, performs 10 fold cross validation
function [RF_TopF1_score, RF_TopKFoldAcc, BestMaxNSpl, BestMinLfSz, BestNumVSam, BestPrSl, BestNumLCycl] = OptimiseCRF( ...
        X,Y,maxnsplits,minlfsize,numvarsam,predsel,numlcycl)
    
    rng(0); %for reproducability
    %create variables for storing
    RF_TopF1_score = 0; %best F1 Score
    RF_TopKFoldAcc = 0; %best KFold Accuracy
    BestMaxNSpl = 0; %best MaxNumSplits
    BestMinLfSz = 0; %best MinLeafSize
    BestNumVSam = 0; %best NumVariablesToSample
    BestPrSl = 0; % best PredictorSelection
    BestNumLCycl = 0; %best NumLearningCycles
    %Begin checking
    for prdsel = predsel %check all predictor selection values
        for nlc = numlcycl %check all NumLearningCycles
            for nvs = numvarsam %check all NumVariablesToSample
                for mls = minlfsize %check all MinLeafSize
                    for mns = maxnsplits %check all MaxNumSplits
                        [f1_score, KFoldAcc] = CustomRF(X,Y,mns,mls,nvs,prdsel,nlc,false); %call our custom RF
                        %compare F1 Scores and store scores and best hyperparameters
                        if f1_score > RF_TopF1_score
                            RF_TopF1_score = f1_score;
                            RF_TopKFoldAcc = KFoldAcc;
                            BestMaxNSpl = mns;
                            BestMinLfSz = mls;
                            BestNumVSam = nvs;
                            BestPrSl = prdsel;
                            BestNumLCycl = nlc;
                        end
                    end
                end
            end
        end
    end
end
