function [r1,x] = feasibility_with_eps(K, eps, cond_prob, prob_act,joint_prob)

% Rational inattention test - general cost, NIAS in A matrix, NIAC in
% nonlcon arguments in c matrix.
% OUTPUT: FEASIBLE SOLUTION
small = eps;
NIAS = (10*9)*K; %number of constraints

A = zeros(NIAS,100*K+K); % 10 states, 10 actions, only NIAS conditions in Ax<=b

%NIAS
nias_count=1;
for k=1:K % for all decision problems
    for a=1:10 % primary action in NIAS (higher utility)
        for b = 1:10 % secondary action in NIAS (lower utility)
            if a~=b
                % NIAS : \sum_{x} p(x|a) u(x,b) - p(x|a) u(x,a) <= 0.
                A(nias_count,(k-1)*100 + (b-1)*10 + 1: (k-1)*100 + (b-1)*10 + 10) = cond_prob((k-1)*100 + (a-1)*10 + 1: (k-1)*100 + (a-1)*10 + 10);
                A(nias_count,(k-1)*100 + (a-1)*10 + 1: (k-1)*100 + (a-1)*10 + 10) = -cond_prob((k-1)*100 + (a-1)*10 + 1: (k-1)*100 + (a-1)*10 + 10);
                %A(nias_count,100*K + K + 1) = 1; %only for robustness test
                nias_count=nias_count+1;
            end
        end
    end
end

%NIAC-under non-linear constraints

function [c,ceq] = NIAC(x)
c=K*(K-1); %array of NIAC constraints
ceq=[];
niac_count = 1;
for main=1:K
    for side=1:K
        if main~=side
            main_u = sum(  joint_prob( (main-1)*100 + 1 : main*100 )'.* x((main-1)*100 + 1 : main*100)   );
            side_u = 0;
            for act = 1:10
                vec_act = zeros(1,10);
                for cand=1:10
                    vec_act(cand) = sum( cond_prob( (side-1)*100 + (act-1)*10 + 1 :...
                        (side-1)*100 + act*10  ).* x( (main-1)*100 + (cand-1)*10 + 1 : (main-1)*100 + cand*10 )' );
                end
                side_u = side_u + prob_act( (side-1)*10 + act )*max(vec_act);
            end
            c(niac_count) = side_u - x(100*K + side) - (main_u - x(100*K + main))+small;
            %c(niac_count) = side_u - x(100*K + side) - (main_u - x(100*K + main)) + x(100*K + K +1); %add last term for robustness test only
            niac_count=niac_count+1;
        end
    end
end

ceq = [];
end

%Can constrain the utility values to be within [0,1] WLOG
lb = 1e-4*ones(100*K+K,1); 
ub = ones(100*K+K,1);
ub(100*K+1:100*K+K) = 0.1;

function f = objfun(x)
   f = 0; %zero objective function, since we only need feasibility
end % Compute function value at x
options = optimoptions('fmincon','MaxFunctionEvaluations',100000,'MaxIterations',100000);%,'StepTolerance',1e-14,'OptimalityTolerance',0.5*1e-3);
[r1 r2 x r3] = fmincon(@objfun,(2*lb + ub)/3,A,-small*ones(NIAS,1),[],[],lb,ub,@NIAC,options);
end









