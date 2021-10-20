function [r1,x] = fmincon_sparse_less_var(K, cond_prob, prob_act,joint_prob)

%Rational inattention test - general cost, NIAS in A matrix, NIAC in
%nonlcon arguments in c matrix.
% OUTPUT: SPARSEST SOLUTION - I am already assuming off-diagonal elements
% in utility function to  be zero. Hence, number of free variables are 10K
% + K, instead of 100K + K.

small = 1e-3;
NIAS = (10*9)*K; %number of constraints

num_var = 10*K + K;
% A = zeros(NIAS,100*K+K); % 10 states, 10 actions, only NIAS conditions in Ax<=b
A = zeros(NIAS,num_var); % 10 states, 10 actions, only NIAS conditions in Ax<=b

nias_count=1;
for k=1:K % for all decision problems
    for a=1:10 % primary action in NIAS (higher utility)
        for b = 1:10 % secondary action in NIAS (lower utility)
            if a~=b
                % NIAS : \sum_{x} p(x|a) u(x,b) - p(x|a) u(x,a) <= 0.
                A(nias_count,(k-1)*10 +  b) = cond_prob((k-1)*100 + (a-1)*10 + b);
                A(nias_count,(k-1)*10 +  a) = -cond_prob((k-1)*100 + (a-1)*10 + a);
                %A(nias_count,100*K + K + 1) = 1; %only for robustness test
                nias_count=nias_count+1;
            end
        end
    end
end

%NIAC-under non-linear constraints

function [c,ceq] = NIAC(x) % NIAC constraints
c=[];
ceq = [];
y = zeros(100*K+K,1);
for i=1:K
    for j=1:10
        y((i-1)*100+(j-1)*10+j) = x((i-1)*10+j);
    end
    y(100*K+i) = x(10*K+i);
end
niac_count = 1;
for main=1:K
    for side=1:K
        if main~=side
            main_u = sum(  joint_prob( (main-1)*100 + 1 : main*100 )'.* y((main-1)*100 + 1 : main*100)   );
            side_u = 0;
            for act = 1:10
                vec_act = zeros(1,10);
                for cand=1:10
                    vec_act(cand) = sum( cond_prob( (side-1)*100 + (act-1)*10 + 1 :...
                        (side-1)*100 + act*10  ).* y( (main-1)*100 + (cand-1)*10 + 1 : (main-1)*100 + cand*10 )' );
                end
                side_u = side_u + prob_act( (side-1)*10 + act )*max(vec_act);
            end
            c(niac_count) = side_u - y(100*K + side) - (main_u - y(100*K + main));
            niac_count=niac_count+1;
        end
    end
end
% ensure unit frobenius norm for the utility function of each agent.
% for act = 1:K
%     ceq(act) = sum(y((act-1)*100+1:(act-1)*100+100).^2)-1;
% end

end

%Can constrain the utility values to be within [0,1] WLOG
lb = 1e-2*ones(10*K+K,1); %nominal (ordinal) value TBH
ub = ones(10*K+K,1);
ub(10*K+1:10*K+K) = 100;


function f = objfun(x) 
   f= sum(x(1:K*10)) ; %min L1 norm
end % Compute function value at x
options = optimoptions(@fmincon,'PlotFcn','optimplotfval','Algorithm','interior-point','MaxFunctionEvaluations',1000000,'MaxIterations',100000);%,'StepTolerance',1e-14,'OptimalityTolerance',0.5*1e-3);
[r1 r2 x r3] = fmincon(@objfun,(2*lb + ub)/3,A,-small*ones(NIAS,1),[],[],lb,ub,@NIAC,options);
end




