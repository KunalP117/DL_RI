clear all;
clc()

tic %Execution time quoted in the paper obtained using the tic-toc construct

load('JOINT_PROB_NN/DATA_LeNet.mat'); % CNN data trained in phase 3 - 30 epochs
JOINT_PROB  = joint_prob_ls_0.JOINT_PROB;
COND_PROB = joint_prob_ls_0.COND_PROB;
PROB_ACT = joint_prob_ls_0.PROB_ACT;
load('sparse_findings.mat');
start = sparse_NIN_ls_1; %pre-computed sparsest utility.
%load('dl_data_30.mat'); % CNN data trained in phase 3 - 30 epochs
%load('dl_data_60.mat'); % CNN data trained in phase 2 - 60 epochs
%load('dl_data_125.mat'); % CNN data trained in phase 1 - 125 epochs




% Function Information:
% 1. To check for feasibility, use   : fmincon_feasibility.m
% 2. To find sparsest solution, use  : fmincon_sparse.m
% 3. To find most robust solution,use: fmincon_robust.m


%Parameters for BRP test
K = 15; % for testing only one batch of trained CNNs
X=10;
% REPAIR COND_PROB
COND_PROB_NEW = [];
for k=1:K
    CB = COND_PROB((k-1)*X*X + 1 : k*X*X );
    CB = reshape(CB,[10,10]);  
    CB = CB';
    CB = reshape(CB,[1,100]);
    COND_PROB_NEW = [COND_PROB_NEW CB];
end
COND_PROB = COND_PROB_NEW;


%K = 15; % number of CNNs whose data is being tested via the BRP test of Theorem 1

%%
% example function call for testing CNNs trained on a single phase, say
% phase 2 - 30 epochs.

% Below, 'r' contains 
% 1. The sparsity enhanced utility values of 5 CNNs (index 1:500)
% 2. Information acquisition costs of the 5 CNNs (index 501:505)
%[r,x] = fmincon_sparse_less_var(K,COND_PROB,PROB_ACT,JOINT_PROB);
%[r,x] = fmincon_sparse_less_var_specify_util(K,COND_PROB,PROB_ACT,JOINT_PROB,start);

%% SPARSITY:
%[r,x] = fmincon_feasibility_general(K,X,COND_PROB(1:X*X*K),PROB_ACT(1:X*K),JOINT_PROB(1:X*X*K));

load('sparse_LeNet_CBRP_0.mat');
start = x;
[r,x] = fmincon_sparse_CBRP(K,X,COND_PROB(1:X*X*K),PROB_ACT(1:X*K),JOINT_PROB(1:X*X*K),start);
utility_sparse_CBRP = reshape(r(1:X*X),[X,X]);
utility_sparse_CBRP = utility_sparse_CBRP/(1e-4); %lb of each value
utility_sparse_CBRP = diag(utility_sparse_CBRP);
%% MARGINS:
% [r,x] = fmincon_feasibility_general(K,X,COND_PROB(1:X*X*K),PROB_ACT(1:X*K),JOINT_PROB(1:X*X*K));
%[r,x] = fmincon_feasibility_singleU_multiple_lambda(K,X,COND_PROB(1:X*X*K),PROB_ACT(1:X*K),JOINT_PROB(1:X*X*K));


% [r,x] = fmincon_feasibility(K,horzcat(cond_prob_full_30,cond_prob_fuller_30,cond_prob_short_30,cond_prob_shorter_30,cond_prob_shortest_30),...
%    prob_act_30, horzcat(joint_prob_vec_full_30, joint_prob_vec_fuller_30, joint_prob_vec_short_30, joint_prob_vec_shorter_30, joint_prob_vec_shortest_30));


%%
% function call for testing CNNs trained on all 3 single phases combined

% Below, 'r' contains 
% 1. The sparsity enhanced utility values of 15 CNNs (index 1:1500)
% 2. Information acquisition costs of the 15 CNNs (index 1501:1515)
%combined datasets
% [r,x] = fmincon_feasibility( K,horzcat(cond_prob_full,cond_prob_fuller,cond_prob_short,cond_prob_shorter,cond_prob_shortest,...
%         cond_prob_full_30,cond_prob_fuller_30,cond_prob_short_30,cond_prob_shorter_30,cond_prob_shortest_30,...    
%         cond_prob_full_60,cond_prob_fuller_60,cond_prob_short_60,cond_prob_shorter_60,cond_prob_shortest_60),...   
%         horzcat(prob_act,prob_act_30,prob_act_60),...
%         horzcat(joint_prob_vec_full, joint_prob_vec_fuller, joint_prob_vec_short, joint_prob_vec_shorter, joint_prob_vec_shortest,...
%         joint_prob_vec_full_30, joint_prob_vec_fuller_30, joint_prob_vec_short_30, joint_prob_vec_shorter_30, joint_prob_vec_shortest_30,...
%         joint_prob_vec_full_60, joint_prob_vec_fuller_60, joint_prob_vec_short_60, joint_prob_vec_shorter_60, joint_prob_vec_shortest_60) );

toc

