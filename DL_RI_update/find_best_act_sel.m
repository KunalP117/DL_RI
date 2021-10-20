function [opt_act_sel,exitmsg] = find_best_act_sel(K,X,JP,util_costs,my_util,start)
%FIND_BEST_JOINTPROB find that jointprob for K agents that minimizes max
%margin for the dataset
% K: number of agents that form util_costs
% utils_costs: first K*X*X - utility values, next K scalars:
% C(\alpha_k),k=1,2,...,K.

num_vars = X*X;

%linear equality constraint
Aeq = zeros(X,num_vars);
for i_Aeq = 1:X % setting the marginal of state to be equal to prior.
    Aeq(i_Aeq,i_Aeq:X:(X-1)*X+i_Aeq) = 1;
end

%maximize utility - cost : minimize cost - utility
function f = objfun(x)
    f =  - exp_util(x,my_util) + cost(x);
end

%net expected utility
function exp_u = exp_util(x,u)
    exp_u=0;
    for act = 1:X
        ind_act = (act-1)*X+1:act*X; 
        u_vec = zeros(1,X);
        for act_2 = 1:X
            ind_act_2 = (act_2-1)*X+1:act_2*X;
            u_vec(act_2) = sum(x(ind_act).*u(ind_act_2));
        end
        exp_u = exp_u + max(u_vec);
    end    
end
%calculate cost given cost evaluated at K points in the simplex
function c = cost(x)
    cost_vec = zeros(1,K);
    for k_cost=1:K
        index_range = (k_cost-1)*X*X + 1 : k_cost*X*X; 
        cost_vec(k_cost) = util_costs(K*X*X + k_cost) ...
        - sum(util_costs(index_range).*JP(index_range)') ...
        + exp_util(x,util_costs(index_range));
    end
    c = max(cost_vec);
end


lb = zeros(num_vars,1);
ub = ones(num_vars,1);

% function f = objfun(x)
% % X=3;K=3;
% % num_vars = K*X*X;
%     joint_prob = transpose(x);
%     prob_act = zeros(1,K*X);
%     prob_act_long = zeros(1,num_vars);
%     for k=1:K
%         for act = 1:X
%             prob_act((k-1)*X+act) = sum( x( (k-1)*X*X + act: X : (k-1)*X*X + (X-1)*X+ act   ) );
%             prob_act_long( (k-1)*X*X + act: X : (k-1)*X*X + (X-1)*X+ act   ) = prob_act((k-1)*X+act);
%         end
%     end
%     cond_prob = joint_prob./prob_act_long;
%     [~,max_margin] = fmincon_robustness_general(K, X, cond_prob, prob_act,joint_prob);
%     f=max_margin;
% end
% 
% lb = zeros(num_vars,1);
% ub = ones(num_vars,1);

% function stop = outfun(x,optimValues,state)
%     %save('best_LeNet_pred.mat','x'); %%%%%%%%%%%%%%%%%%%%%%%%% CHANGE FOR ARCHITECTURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     stop=0;
% end

%options = optimoptions('fmincon','OutputFcn',@outfun,'PlotFcn','optimplotfval','MaxFunctionEvaluations',10000000000,'MaxIterations',10000,'Display','off');%,'StepTolerance',1e-14,'PlotFcn','optimplotfval');%,'OptimalityTolerance',0.5*1e-3);
options = optimoptions('fmincon','algorithm','active-set','PlotFcn','optimplotfval','MaxFunctionEvaluations',10000000000,'MaxIterations',500);%,'StepTolerance',1e-16,'ConstraintTolerance',1e-16);%,'StepTolerance',1e-14,'PlotFcn','optimplotfval');%,'OptimalityTolerance',0.5*1e-3);
[opt_joint_prob,~,exitmsg,~] = fmincon(@objfun,start,[],[],Aeq,0.1*ones(X,1),lb,ub,[],options);
opt_act_sel = opt_joint_prob*10;
%opt_act_sel = diag(reshape(opt_act_sel,[X,X]))';
end


