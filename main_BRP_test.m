clear all;
clc()

tic %Execution time quoted in the paper obtained using the tic-toc construct

load('dl_data_30.mat'); % CNN data trained in phase 3 - 30 epochs
load('dl_data_60.mat'); % CNN data trained in phase 2 - 60 epochs
load('dl_data_125.mat'); % CNN data trained in phase 1 - 125 epochs


% Function Information:
% 1. To check for feasibility, use   : fmincon_feasibility.m
% 2. To find sparsest solution, use  : fmincon_sparse.m
% 3. To find most robust solution,use: fmincon_robust.m


%Parameters for BRP test
K = 5; % for testing only one batch of trained CNNs
%K = 15; % number of CNNs whose data is being tested via the BRP test of Theorem 1

%%
% example function call for testing CNNs trained on a single phase, say
% phase 2 - 30 epochs.

% Below, 'r' contains 
% 1. The sparsity enhanced utility values of 5 CNNs (index 1:500)
% 2. Information acquisition costs of the 5 CNNs (index 501:505)
[r,x] = fmincon_feasibility(K,horzcat(cond_prob_full_30,cond_prob_fuller_30,cond_prob_short_30,cond_prob_shorter_30,cond_prob_shortest_30),...
   prob_act_30, horzcat(joint_prob_vec_full_30, joint_prob_vec_fuller_30, joint_prob_vec_short_30, joint_prob_vec_shorter_30, joint_prob_vec_shortest_30),r);


%%
% function call for testing CNNs trained on all 3 single phases combined

% Below, 'r' contains 
% 1. The sparsity enhanced utility values of 15 CNNs (index 1:1500)
% 2. Information acquisition costs of the 15 CNNs (index 1501:1515)
%combined datasets
[r,x] = fmincon_feasibility(K,horzcat(cond_prob_full,cond_prob_fuller,cond_prob_short,cond_prob_shorter,cond_prob_shortest,...
        cond_prob_full_30,cond_prob_fuller_30,cond_prob_short_30,cond_prob_shorter_30,cond_prob_shortest_30,...    
        cond_prob_full_60,cond_prob_fuller_60,cond_prob_short_60,cond_prob_shorter_60,cond_prob_shortest_60),...   
        horzcat(prob_act,prob_act_30,prob_act_60),...
        horzcat(joint_prob_vec_full, joint_prob_vec_fuller, joint_prob_vec_short, joint_prob_vec_shorter, joint_prob_vec_shortest,...
        joint_prob_vec_full_30, joint_prob_vec_fuller_30, joint_prob_vec_short_30, joint_prob_vec_shorter_30, joint_prob_vec_shortest_30,...
        joint_prob_vec_full_60, joint_prob_vec_fuller_60, joint_prob_vec_short_60, joint_prob_vec_shorter_60, joint_prob_vec_shortest_60));


toc

