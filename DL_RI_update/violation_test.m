clear;
clc();

num_btsp = 1000;
% ref_img_dataset = zeros(1,10000);
% 
% eps_true_val = 0.02;% although true val = 0.0098
% num_denom=10;
% eps_test_vec = linspace(0.03,0.027,num_denom);
% test_results = zeros(num_btsp,num_denom);

X=3; %num dp
K=5;
prior = [0.2 0.3 0.5];

%load('dl_data_30.mat'); % CNN data trained in phase 3 - 30 epochs
tic
for dataset = 1:num_btsp
    dataset
    %Prep bootstrapped dataset
    act_sel_1 = rand(X);
    act_sel_2 = rand(X);
    act_sel_3 = rand(X);
    act_sel_4 = rand(X);
    act_sel_5 = rand(X);
    for state=1:X
        act_sel_1(state,:) = act_sel_1(state,:)/sum(act_sel_1(state,:));
        act_sel_2(state,:) = act_sel_2(state,:)/sum(act_sel_2(state,:));
        act_sel_3(state,:) = act_sel_3(state,:)/sum(act_sel_3(state,:));
        act_sel_4(state,:) = act_sel_4(state,:)/sum(act_sel_4(state,:));
        act_sel_5(state,:) = act_sel_5(state,:)/sum(act_sel_5(state,:));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    prior_mat = [prior(1);prior(2);prior(3)];
    
    joint_prob_1 = act_sel_1.*prior_mat;
    joint_prob_2 = act_sel_2.*prior_mat;
    joint_prob_3 = act_sel_3.*prior_mat;
    joint_prob_4 = act_sel_4.*prior_mat;
    joint_prob_5 = act_sel_5.*prior_mat;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    prob_act_1 = zeros(1,X);
    prob_act_2 = zeros(1,X);
    prob_act_3 = zeros(1,X);
    prob_act_4 = zeros(1,X);
    prob_act_5 = zeros(1,X);
    for act=1:X
        prob_act_1(act) = sum(joint_prob_1(:,act));
        prob_act_2(act) = sum(joint_prob_2(:,act));
        prob_act_3(act) = sum(joint_prob_3(:,act));
        prob_act_4(act) = sum(joint_prob_4(:,act));
        prob_act_5(act) = sum(joint_prob_5(:,act));
    end
    
    prob_act = horzcat(prob_act_1,prob_act_2,prob_act_3,prob_act_4,prob_act_5);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    cond_prob_1=joint_prob_1;
    cond_prob_2=joint_prob_2;
    cond_prob_3=joint_prob_3;
    cond_prob_4=joint_prob_5;
    cond_prob_5=joint_prob_5;
    for act=1:X
        cond_prob_1(:,act) = cond_prob_1(:,act)/prob_act_1(act);
        cond_prob_2(:,act) = cond_prob_2(:,act)/prob_act_2(act);
        cond_prob_3(:,act) = cond_prob_3(:,act)/prob_act_3(act);
        cond_prob_4(:,act) = cond_prob_4(:,act)/prob_act_4(act);
        cond_prob_5(:,act) = cond_prob_5(:,act)/prob_act_5(act);
    end
    
    % Vec operations
    cond_prob_1 = reshape(cond_prob_1,[1,X*X]);
    cond_prob_2 = reshape(cond_prob_2,[1,X*X]);
    cond_prob_3 = reshape(cond_prob_3,[1,X*X]);
    cond_prob_4 = reshape(cond_prob_4,[1,X*X]);
    cond_prob_5 = reshape(cond_prob_5,[1,X*X]);
    
    joint_prob_1 = reshape(joint_prob_1,[1,X*X]);
    joint_prob_2 = reshape(joint_prob_2,[1,X*X]);
    joint_prob_3 = reshape(joint_prob_3,[1,X*X]);
    joint_prob_4 = reshape(joint_prob_4,[1,X*X]);
    joint_prob_5 = reshape(joint_prob_5,[1,X*X]);
    
    
    % Test
    [r,x] = fmincon_feasibility_general(K,X,horzcat(cond_prob_1,cond_prob_2,cond_prob_3,cond_prob_4,cond_prob_5),prob_act,...
    horzcat(joint_prob_1,joint_prob_2,joint_prob_3,joint_prob_4,joint_prob_5));
        %[r,x] = feasibility_with_eps(5,eps_test_vec(denom),horzcat(cond_prob_full_30,cond_prob_fuller_30,cond_prob_short_30,cond_prob_shorter_30,cond_prob_shortest_30),...
        %prob_act_30, horzcat(joint_prob_vec_full_30, joint_prob_vec_fuller_30, joint_prob_vec_short_30, joint_prob_vec_shorter_30, joint_prob_vec_shortest_30));

    if x~=1
        break
    end
end

toc