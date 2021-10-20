function [opt_joint_prob,min_max_margin] = find_best_jointprob(K,X)
%FIND_BEST_JOINTPROB find that jointprob for K agents that minimizes max
%margin for the dataset
num_vars = K*X*X;

Aeq = zeros(K,K*X);
for i = 1:K
    Aeq(i,(i-1)*X*X+1:i*X*X) = 1;
end
lb = zeros(num_vars,1);
ub = ones(num_vars,1);

function f = objfun(x)
% X=3;K=3;
% num_vars = K*X*X;
    joint_prob = transpose(x);
    prob_act = zeros(1,K*X);
    prob_act_long = zeros(1,num_vars);
    for k=1:K
        for act = 1:X
            prob_act((k-1)*X+act) = sum( x( (k-1)*X*X + act: X : (k-1)*X*X + (X-1)*X+ act   ) );
            prob_act_long( (k-1)*X*X + act: X : (k-1)*X*X + (X-1)*X+ act   ) = prob_act((k-1)*X+act);
        end
    end
    cond_prob = joint_prob./prob_act_long;
    [~,max_margin] = fmincon_robustness_general(K, X, cond_prob, prob_act,joint_prob);
    f=max_margin;
end

function [c,ceq] = nonlin(x)
% X=3;K=3;
    ceq=[];
    %small = 1;
    c=zeros(1,K*(K-1)/2);
    iter=1;
    for k1=1:K
        for k2 = (k1+1):K
            c(iter) = 0.1 - (sum(abs(x((k1-1)*X*X+1:k1*X*X) - x((k2-1)*X*X+1:k2*X*X)))/X*X);
            iter=iter+1;
        end
    end
end

options = optimoptions('fmincon','PlotFcn','optimplotfval','MaxFunctionEvaluations',10000000000,'MaxIterations',1000000000,'Display','off');%,'StepTolerance',1e-14,'PlotFcn','optimplotfval');%,'OptimalityTolerance',0.5*1e-3);
[opt_joint_prob,min_max_margin] = fmincon(@objfun,ones(num_vars,1)/K*X,[],[],Aeq,ones(K,1),lb,ub,@nonlin,options);
end


% function f = objfun(x)
% X=3;K=3;
% num_vars = K*X*X;
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
% function [c,ceq] = nonlin(x)
% X=3;K=3;
%     ceq=[];
%     %small = 1;
%     c=zeros(1,K*(K-1)/2);
%     iter=1;
%     for k1=1:K
%         for k2 = (k1+1):K
%             c(iter) = 0.1 - (sum(abs(x((k1-1)*X*X+1:k1*X*X) - x((k2-1)*X*X+1:k2*X*X)))/X*X);
%             iter=iter+1;
%         end
%     end
% end