function [r1,x] = fmincon_sparse_less_var_specify_util(K, cond_prob, prob_act,joint_prob,start)

%Rational inattention test - general cost, NIAS in A matrix, NIAC in
%nonlcon arguments in c matrix.
% OUTPUT: SPARSEST SOLUTION - Use as start utilities and cost computed for
% a different learning rate schedule. Free variables are costs.
util = start(1:10*K);
start_cost = start(10*K+1:10*K+K);
small = 1e-5;
NIAS = (10*9)*K; %number of constraints

num_var = K; %free variables are costs
NIAC_cond = K*(K-1);
A = zeros(NIAC_cond,num_var); 
b = zeros(NIAC_cond,1);

%NIAC-under non-linear constraints

y = zeros(100*K,1);
for i=1:K
    for j=1:10
        y((i-1)*100+(j-1)*10+j) = util((i-1)*10+j);
    end
end
niac_count = 1;
for main=1:K
    for side=1:K
        if main~=side
            main_u = sum( joint_prob( (main-1)*100 + 1 : main*100 )'.* y((main-1)*100 + 1 : main*100)   );
            side_u = 0;
            for act = 1:10
                vec_act = zeros(1,10);
                for cand=1:10
                    vec_act(cand) = sum( cond_prob( (side-1)*100 + (act-1)*10 + 1 :...
                        (side-1)*100 + act*10  ).* y( (main-1)*100 + (cand-1)*10 + 1 : (main-1)*100 + cand*10 )' );
                end
                side_u = side_u + prob_act( (side-1)*10 + act )*max(vec_act);
            end
            A(niac_count,main) = 1;
            A(niac_count,side) = -1;
            b(niac_count) = main_u - side_u;
            niac_count=niac_count+1;
        end
    end
end
% ensure unit frobenius norm for the utility function of each agent.
% for act = 1:K
%     ceq(act) = sum(y((act-1)*100+1:(act-1)*100+100).^2)-1;
% end

%Can constrain the utility values to be within [0,1] WLOG
lb = 1e-2*ones(K,1); %nominal (ordinal) value TBH
ub = 100*ones(K,1);


function f = objfun(x) 
   f= 0 ; %min L1 norm
end % Compute function value at x
options = optimoptions(@fmincon,'PlotFcn','optimplotconstrviolation','Algorithm','interior-point','MaxFunctionEvaluations',1000000,'MaxIterations',100000,'ConstraintTolerance',3.5e-3,'StepTolerance',1e-20);%,'StepTolerance',1e-14,'OptimalityTolerance',0.5*1e-3);
[r1 r2 x r3] = fmincon(@objfun,start_cost,A,b-small*ones(NIAC_cond,1),[],[],lb,ub,[],options);
end




