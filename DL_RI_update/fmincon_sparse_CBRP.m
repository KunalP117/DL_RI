function [r1,r2] = fmincon_sparse_CBRP(K, X, cond_prob, prob_act,joint_prob,start)

small = 1e-6;
NIAS = (X*(X-1))*K; %number of constraints

A = zeros(NIAS,X*X + 2*K); % 10 states, 10 actions, only NIAS conditions in Ax<=b

%NIAS
nias_count=1;
for k=1:K % for all decision problems
    for a=1:X % primary action in NIAS (higher utility)
        for b = 1:X % secondary action in NIAS (lower utility)
            if a~=b
                % NIAS : \sum_{x} p(x|a) u(x,b) - p(x|a) u(x,a) <= 0.
                A(nias_count, (b-1)*X + 1: (b-1)*X + X) = cond_prob((k-1)*X*X + (a-1)*X + 1 : (k-1)*X*X + a*X);
                A(nias_count, (a-1)*X + 1:  (a-1)*X + X) = -cond_prob((k-1)*X*X + (a-1)*X + 1 : (k-1)*X*X + a*X);
                nias_count=nias_count+1;
            end
        end
    end
end

%NIAC-under non-linear constraints

function [c,ceq] = NIAC(x)
c=zeros(1,K*(K-1)); %array of NIAC constraints
niac_count = 1;
for main=1:K
    for side=1:K
        if main~=side
            c(niac_count) = (joint_prob( (side-1)*X*X + 1 : side*X*X) - joint_prob( (main-1)*X*X + 1 : main*X*X ) )*x(1:X*X) - x(X*X + K + main)*(x(X*X+side) - x(X*X+main)) +small;
            %c(niac_count) = side_u - main_u - x(X*X+K+main)*( x(X*X+side) - x(X*X+main)   )  + x(X*X+K+K+1);%x(X*X + side) - (main_u - x(X*X + main)) + small;
            %c(niac_count) = side_u - x(100*K + side) - (main_u - x(100*K + main)) + x(100*K + K +1); %add last term for robustness test only
            niac_count=niac_count+1;
        end
    end
end
ceq=0;
%ceq = sum(x(1:X*X).^2) - 1;
end

%Can constrain the utility values to be within [0,1] WLOG
lb = 1e-4*ones(X*X+K+K,1); 
ub = ones(X*X+K+K,1);
%ub(X*X+ 2*K + 1) = 100;
%lb(X*X+ 2*K + 1) = 1e-6;
%ub(X*X+1:X*X+K) = 1;

function f = objfun(x)
   f = sum(x(1:X*X)); %zero objective function, since we only need feasibility
end % Compute function value at x

%start = lb;
%start(X^2 + 2*K) =1;
%start = 0.1*ones(X*X+2*K,1); 
%options = optimoptions('fmincon','PlotFcn','optimplotfval','Algorithm','active-set','MaxFunctionEvaluations',1000000,'MaxIterations',100000,'ConstraintTolerance',1e-4,'FunctionTolerance',1e-16);%,'StepTolerance',1e-14,'OptimalityTolerance',0.5*1e-3);
% for ResNet, AlexNet


function stop = outfun(x,optimValues,state)
    save('sparse_LeNet_CBRP_0.mat','x'); %%%%%%%%%%%%%%%%%%%%%%%%% CHANGE FOR ARCHITECTURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    stop=0;
end
options =  optimoptions('fmincon','OutputFcn',@outfun,'PlotFcn','optimplotfval','Algorithm','active-set','MaxFunctionEvaluations',10000000,'MaxIterations',10000,'ConstraintTolerance',1.1e-4,'StepTolerance',1e-10,'FunctionTolerance',1e-10);%,'StepTolerance',1e-14,'OptimalityTolerance',0.5*1e-3);% for NiN,VGG,LeNet

[r1,r2,~,r3] = fmincon(@objfun,start,A,small*ones(NIAS,1),[],[],lb,ub,@NIAC,options);
end

