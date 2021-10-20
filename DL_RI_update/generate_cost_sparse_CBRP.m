%clear all;
clc;

MAX_PT = 1000; %number of points in plot
K=10; %num_agents

load('JOINT_PROB_NN/DATA_VGG16.mat');
JOINT_PROB  = joint_prob_ls_0.JOINT_PROB;
low_jp = JOINT_PROB(1:100);
high_jp = JOINT_PROB((K-1)*100+1:K*100);


low_jp_two = 0.1*ones(1,10);
high_jp_two = 0.9*ones(1,10);
%%
load('sparse_CBRP.mat');
util_costs = sparse_CBRP_VGG16_0; %variable
load('JOINT_PROB_NN/DATA_VGG16.mat'); %variable
%load('JOINT_PROB_NN/DATA_LeNet.mat');
%load('JOINT_PROB_NN/DATA_NiN.mat');
%load('JOINT_PROB_NN/DATA_ResNet.mat'); %change K to 10 for this
%load('JOINT_PROB_NN/DATA_VGG16.mat');


util_mat = diag(reshape(util_costs(1:100),[10,10]));
JOINT_PROB  = joint_prob_ls_0.JOINT_PROB;
%%




grid = linspace(0,1,MAX_PT);
cost = zeros(1,MAX_PT);
cost_two = zeros(1,MAX_PT);

for i=1:MAX_PT
    jp_i = (1-sqrt(grid(i)^2) )*low_jp + sqrt(grid(i)^2)*high_jp;
    jp_i_two = (1-sqrt(grid(i)^2) )*low_jp_two + sqrt(grid(i)^2)*high_jp_two;
    cost_cand = zeros(1,K);
    cost_cand_two = zeros(1,K);
    for j=1:K
        jp_mat = diag(reshape(JOINT_PROB((j-1)*100+1:j*100),[10,10]));
        cost_cand(j) = util_costs(100+j) + util_costs(100+K+j)*sum((jp_i - JOINT_PROB((j-1)*100+1:j*100) ).*util_costs(1:100)');
        cost_cand_two(j) = util_costs(100+j) + util_costs(100+K+j)*sum((jp_i_two - jp_mat' ).*util_mat');
    end
    %reshape(jp_i,[10,10])
    %cost_cand
    cost(i) = max(cost_cand) ;
    cost_two(i) = max(cost_cand_two);
end
cost = cost - cost(1);
cost_two = cost_two - cost_two(1);
%stem(util_costs(101:115)-util_costs(101));