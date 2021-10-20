%function [outputArg1,outputArg2] = fmincon_feasibility_singleU_multiple_lambda(inputArg1,inputArg2)
function [r1,r2] = fmincon_feasibility_singleU_multiple_lambda(K, X, cond_prob, prob_act,joint_prob)

% Rational inattention test - general cost, NIAS in A matrix, NIAC in
% nonlcon arguments in c matrix.
% OUTPUT: FEASIBLE SOLUTION
%small = 0.0000005;
NIAS = (X*(X-1))*K; %number of constraints

A = zeros(NIAS,X*X+K+K+1); % 10 states, 10 actions, only NIAS conditions in Ax<=b

%NIAS
nias_count=1;
for k=1:K % for all decision problems
    for a=1:X % primary action in NIAS (higher utility)
        for b = 1:X % secondary action in NIAS (lower utility)
            if a~=b
                % NIAS : \sum_{x} p(x|a) u(x,b) - p(x|a) u(x,a) <= 0.
                A(nias_count, (b-1)*X + 1: (b-1)*X + X) = cond_prob((k-1)*X*X + (a-1)*X + 1 : (k-1)*X*X + a*X);
                A(nias_count, (a-1)*X + 1:  (a-1)*X + X) = -cond_prob((k-1)*X*X + (a-1)*X + 1 : (k-1)*X*X + a*X);
                A(nias_count, X*X + 2*K + 1) = 1; %only for robustness test
                nias_count=nias_count+1;
            end
        end
    end
end

%NIAC-under non-linear constraints

function [c,ceq] = NIAC(x)
ceq = [];
c=zeros(1,K*(K-1)); %array of NIAC constraints
niac_count = 1;
for main=1:K
    for side=1:K
        if main~=side
            c(niac_count) = (joint_prob( (side-1)*X*X + 1 : side*X*X) - joint_prob( (main-1)*X*X + 1 : main*X*X ) )*x(1:X*X) - x(X*X + K + main)*(x(X*X+side) - x(X*X+main)) + x(X*X + 2*K +1);
            %c(niac_count) = side_u - main_u - x(X*X+K+main)*( x(X*X+side) - x(X*X+main)   )  + x(X*X+K+K+1);%x(X*X + side) - (main_u - x(X*X + main)) + small;
            %c(niac_count) = side_u - x(100*K + side) - (main_u - x(100*K + main)) + x(100*K + K +1); %add last term for robustness test only
            niac_count=niac_count+1;
        end
    end
end

ceq = sum(x(1:X*X).^2) - 1;
end

%Can constrain the utility values to be within [0,1] WLOG
lb = 1e-4*ones(X*X+K+K+1,1); 
ub = ones(X*X+K+K+1,1);
%ub(X*X+ 2*K + 1) = 100;
%lb(X*X+ 2*K + 1) = 1e-6;
%ub(X*X+1:X*X+K) = 1;

function f = objfun(x)
   f = -x(X*X+K+K+1); %zero objective function, since we only need feasibility
end % Compute function value at x


start = 0.1*ones(X*X+2*K+1,1); 


start(X*X + 2*K + 1)=0; % for NiN,VGG,LeNet
%start(X*X + 2*K + 1) = 0.001; % for ResNet,AlexNet

%start = (2*lb+ub)/3;
% options =
% optimoptions('fmincon','PlotFcn','optimplotfval','Algorithm','active-set','MaxFunctionEvaluations',1000000,'MaxIterations',100000,'ConstraintTolerance',1e-4,'FunctionTolerance',1e-16);%,'StepTolerance',1e-14,'OptimalityTolerance',0.5*1e-3);
% for ResNet, AlexNet
options =  optimoptions('fmincon','PlotFcn','optimplotfval','Algorithm','active-set','MaxFunctionEvaluations',1000000,'MaxIterations',100000,'ConstraintTolerance',1e-4,'FunctionTolerance',1e-16);%,'StepTolerance',1e-14,'OptimalityTolerance',0.5*1e-3);% for NiN,VGG,LeNet

[r1,r2,~,r3] = fmincon(@objfun,start,A,zeros(NIAS,1),[],[],lb,ub,@NIAC,options);
end

% function u = best_u(X,joint_prob,x)
% u=0;
% mat = transpose(reshape(joint_prob,[X,X]));
% for iter=1:X
%     vec_iter = zeros(1,X);
%     for iter_2 = 1:X
%         vec_iter(iter_2) = sum((mat(iter,:)').*x( (iter_2-1)*X +1: iter_2*X));
%     end
%     u = u+max(vec_iter);
% end
% end