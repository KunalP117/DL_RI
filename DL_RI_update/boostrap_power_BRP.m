% hist_vec_pmf=zeros(1,10);
% for i=1:140
%     for j=1:10
%         if test_results_cut(i,j)==1
%             hist_vec_pmf(j)=hist_vec_pmf(j)+1;
%             break
%         end
%     end
% end
% hist_vec_pmf = hist_vec_pmf/sum(hist_vec_pmf)
% 
% %sum=0;
% hist_vec_pmf = flip(hist_vec_pmf);
% for i=1:10
%     hist_vec_cdf(i) = sum( hist_vec_pmf(1:i) );
% end

clear;
clc();

num_btsp = 1000;
ref_img_dataset = zeros(1,10000);

eps_true_val = 0.02;% although true val = 0.0098
num_denom=10;
eps_test_vec = linspace(0.03,0.027,num_denom);
test_results = zeros(num_btsp,num_denom);
load('dl_data_30.mat'); % CNN data trained in phase 3 - 30 epochs

tic
for dataset = 1:num_btsp
    dataset
    %Prep bootstrapped dataset
    act_sel_full = zeros(10,10);
    for state=1:10
        act_sel_full(state,:) = btsp_gen_emp_pmf(act_sel_full_30(state,:),10);
    end
    
    act_sel_fuller = zeros(10,10);
    for state=1:10
        act_sel_fuller(state,:) = btsp_gen_emp_pmf(act_sel_fuller_30(state,:),10);
    end
    
    act_sel_short = zeros(10,10);
    for state=1:10
        act_sel_short(state,:) = btsp_gen_emp_pmf(act_sel_short_30(state,:),10);
    end
    
    act_sel_shorter = zeros(10,10);
    for state=1:10
        act_sel_shorter(state,:) = btsp_gen_emp_pmf(act_sel_shorter_30(state,:),10);
    end
    
    act_sel_shortest = zeros(10,10);
    for state=1:10
        act_sel_shortest(state,:) = btsp_gen_emp_pmf(act_sel_shortest_30(state,:),10);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    joint_prob_full = act_sel_full*0.1;
    joint_prob_fuller = act_sel_fuller*0.1;
    joint_prob_short = act_sel_short*0.1;
    joint_prob_shorter = act_sel_shorter*0.1;
    joint_prob_shortest = act_sel_shortest*0.1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    prob_act_full = zeros(1,10);
    for act=1:10
        prob_act_full(act) = sum(joint_prob_full(:,act));
    end
    
    prob_act_fuller = zeros(1,10);
    for act=1:10
        prob_act_fuller(act) = sum(joint_prob_fuller(:,act));
    end
    
    prob_act_short = zeros(1,10);
    for act=1:10
        prob_act_short(act) = sum(joint_prob_short(:,act));
    end
    
    prob_act_shorter = zeros(1,10);
    for act=1:10
        prob_act_shorter(act) = sum(joint_prob_shorter(:,act));
    end
    
    prob_act_shortest = zeros(1,10);
    for act=1:10
        prob_act_shortest(act) = sum(joint_prob_shortest(:,act));
    end
    
    prob_act = horzcat(prob_act_full,prob_act_fuller,prob_act_short,prob_act_shorter,prob_act_shortest);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    cond_prob_full=joint_prob_full;
    for act=1:10
        cond_prob_full(:,act) = cond_prob_full(:,act)/prob_act_full(act);
    end
    
    cond_prob_fuller=joint_prob_fuller;
    for act=1:10
        cond_prob_fuller(:,act) = cond_prob_fuller(:,act)/prob_act_fuller(act);
    end
    
    cond_prob_short=joint_prob_short;
    for act=1:10
        cond_prob_short(:,act) = cond_prob_short(:,act)/prob_act_short(act);
    end
    
    cond_prob_shorter=joint_prob_shorter;
    for act=1:10
        cond_prob_shorter(:,act) = cond_prob_shorter(:,act)/prob_act_shorter(act);
    end
    
    cond_prob_shortest=joint_prob_shortest;
    for act=1:10
        cond_prob_shortest(:,act) = cond_prob_shortest(:,act)/prob_act_shortest(act);
    end
    
    % Vec operations
    cond_prob_full = reshape(cond_prob_full,[1,100]);
    cond_prob_fuller = reshape(cond_prob_fuller,[1,100]);
    cond_prob_short = reshape(cond_prob_short,[1,100]);
    cond_prob_shorter = reshape(cond_prob_shorter,[1,100]);
    cond_prob_shortest = reshape(cond_prob_shortest,[1,100]);
    
    joint_prob_full = reshape(joint_prob_full,[1,100]);
    joint_prob_fuller = reshape(joint_prob_fuller,[1,100]);
    joint_prob_short = reshape(joint_prob_short,[1,100]);
    joint_prob_shorter = reshape(joint_prob_shorter,[1,100]);
    joint_prob_shortest = reshape(joint_prob_shortest,[1,100]);
    
    
    % Test
    for denom=1:num_denom
        denom
        [r,x] = feasibility_with_eps(5,eps_test_vec(denom) ,horzcat(cond_prob_full,cond_prob_fuller,cond_prob_short,cond_prob_shorter,cond_prob_shortest),...
        prob_act, horzcat(joint_prob_full, joint_prob_fuller, joint_prob_short, joint_prob_shorter, joint_prob_shortest));
        %[r,x] = feasibility_with_eps(5,eps_test_vec(denom),horzcat(cond_prob_full_30,cond_prob_fuller_30,cond_prob_short_30,cond_prob_shorter_30,cond_prob_shortest_30),...
        %prob_act_30, horzcat(joint_prob_vec_full_30, joint_prob_vec_fuller_30, joint_prob_vec_short_30, joint_prob_vec_shorter_30, joint_prob_vec_shortest_30));

        if x==1
            break
        end
    end
    test_results(dataset,denom:num_denom)=1;     
end

toc