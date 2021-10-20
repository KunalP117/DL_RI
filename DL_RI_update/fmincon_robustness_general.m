function [utils,max_margin] = fmincon_robustness_general(K, X, cond_prob, prob_act,joint_prob)

% Rational inattention test - general cost, NIAS in A matrix, NIAC in
% nonlcon arguments in c matrix.
% OUTPUT: FEASIBLE SOLUTION
small = 0.000001;
NIAS = (X*(X-1))*K; %number of constraints

A = zeros(NIAS,X*X*K+K+1); % 10 states, 10 actions, only NIAS conditions in Ax<=b

%NIAS
nias_count=1;
for k=1:K % for all decision problems
    for a=1:X % primary action in NIAS (higher utility)
        for b = 1:X % secondary action in NIAS (lower utility)
            if a~=b
                % NIAS : \sum_{x} p(x|a) u(x,b) - p(x|a) u(x,a) <= 0.
                A(nias_count,(k-1)*X*X + (b-1)*X + 1: (k-1)*X*X + (b-1)*X + X) = cond_prob((k-1)*X*X + (a-1)*X + 1: (k-1)*X*X + (a-1)*X + X);
                A(nias_count,(k-1)*X*X + (a-1)*X + 1: (k-1)*X*X + (a-1)*X + X) = -cond_prob((k-1)*X*X + (a-1)*X + 1: (k-1)*X*X + (a-1)*X + X);
                A(nias_count,X*X*K + K + 1) = 1; %only for robustness test
                nias_count=nias_count+1;
            end
        end
    end
end

%NIAC-under non-linear constraints

function [c,ceq] = NIAC(x)
c=K*(K-1); %array of NIAC constraints
niac_count = 1;
for main=1:K
    for side=1:K
        if main~=side
            main_u = sum(  joint_prob( (main-1)*X*X + 1 : main*X*X )'.* x((main-1)*X*X + 1 : main*X*X)   );
            side_u = 0;
            for act = 1:X
                vec_act = zeros(1,X);
                for cand=1:X
                    vec_act(cand) = sum( cond_prob( (side-1)*X*X + (act-1)*X + 1 :...
                        (side-1)*X*X + act*X  ).* x( (main-1)*X*X + (cand-1)*X + 1 : (main-1)*X*X + cand*X )' );
                end
                side_u = side_u + prob_act( (side-1)*X + act )*max(vec_act);
            end
            %c(niac_count) = side_u - x(X*X*K + side) - (main_u - x(X*X*K + main)) + small;
            c(niac_count) = side_u - x(X*X*K + side) - (main_u - x(X*X*K + main)) + x(X*X*K + K +1); %add last term for robustness test only
            niac_count=niac_count+1;
        end
    end
end

ceq = sum(x(1:K*X*X).^2) - (K*0.01); % normalizing utility norm
end

%Can constrain the utility values to be within [0,1] WLOG
lb = 1e-4*ones(X*X*K+K+1,1); 
ub = 100*ones(X*X*K+K+1,1);
ub(X*X*K+1:X*X*K+K) = 100;
lb(X*X*K+K+1) = -10;
ub(X*X*K+K+1) = 10;


function f = objfun(x)
   f = -x(X*X*K+K+1);
end % Compute function value at x
options = optimoptions('fmincon','Algorithm','active-set','MaxFunctionEvaluations',10000000000,'MaxIterations',1000000000);%,'StepTolerance',1e-14,'OptimalityTolerance',0.5*1e-3);
[r1,r2,~,~] = fmincon(@objfun,(lb + ub)/2,A,-small*ones(NIAS,1),[],[],lb,ub,@NIAC,options);
utils = r1;
max_margin = -r2;
end




