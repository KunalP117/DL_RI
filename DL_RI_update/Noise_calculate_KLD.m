clear;
clc;

BIG=1e4;

K = 11; X=10; K_TEST = 10; ERR = [];

JP=[];AS_TEST=[]; AS_INTERPET=[]; AS_PRED = [];
AS_FULL = [];
AS_PRED_BIG = [];
AS_TEST_BIG = [];
% for i=10:20
%     load(sprintf('VGG16_noise_test/DATA_LeNet_noise_%d.mat',i))
%     eval(['JP = horzcat(JP,all_prob_std_',int2str(i),'.JOINT_PROB);']);
%     eval(['CP = horzcat(CP,all_prob_std_',int2str(i),'.COND_PROB);']);
%     eval(['PA = horzcat(PA,all_prob_std_',int2str(i),'.PROB_ACT);']);
%     eval(['AS = horzcat(AS,all_prob_std_',int2str(i),'.ACT_SEL);']);
% end

%FOR ALL BUT LeNet
% for i=100:10:200
%     load(sprintf('AlexNet_noise_test/AlexNet/DATA_AlexNet_noise_%d.mat',i)); % ARCHITECTURE DEPENDENT
%     JP = horzcat(JP,all_prob_std_105.JOINT_PROB);
%     AS_diag = 10*diag(reshape(all_prob_std_105.JOINT_PROB,[10,10]))';
%     AS_INTERPET = [AS_INTERPET;AS_diag]; % each row contains p(x|x) for all x in CIFAR-10, covariance in dataset : i/100
% end

%For LeNet
for i=10:20
    load(sprintf('LeNet_noise_test/LeNet/DATA_LeNet_noise_%d.mat',i)); % ARCHITECTURE DEPENDENT
    eval(['JP = horzcat(JP,all_prob_std_',int2str(i),'.JOINT_PROB);']);
    eval(['AS_diag = 10*diag(reshape(all_prob_std_',int2str(i),'.JOINT_PROB,[10,10]))';]);
    AS_INTERPET = [AS_INTERPET;AS_diag]; % each row contains p(x|x) for all x in CIFAR-10, covariance in dataset : i/100
end

iter=0;val_dp=0;

%FOR LENET
for i=105:10:185
    if i ~= 145
        disp(['noisecov=',num2str(i/100)]);
        iter=iter+1;
        load(sprintf('LeNet_noise_test/LeNet/DATA_LeNet_noise_%d.mat',i)); % ARCHITECTURE DEPENDENT
        eval(['AS_diag = 10*diag(reshape(all_prob_std_',num2str(i),'.JOINT_PROB,[10,10]));']);
        eval(['AS_FULL = reshape(transpose(10*all_prob_std_',num2str(i),'.JOINT_PROB),[10,10]);']); AS_FULL = AS_FULL';
        %%%
        load('robust_LeNet_noise.mat'); % load interpretable model  % ARCHITECTURE DEPENDENT
        %load('robust_VGG16_noise_1.mat'); % load interpretable model  % ARCHITECTURE DEPENDENT, only for VGG16
      
        util_costs=BIG*x(1:K*X*X + K); % interpretable model
        ind_range = (iter-1)*X*X + 1 : iter*X*X; 
        start = (JP(ind_range)' + JP(ind_range+X*X)')/2;
        my_util = (util_costs(ind_range) + util_costs(ind_range+X*X))/2; % interpolated utility for noise covariance 1.05/1.15/1.25
        
        
        %%%   
        [act_sel_pred,exitmsg] = find_best_act_sel(K,X,JP,util_costs,my_util,start); %predicted act_sel given interpolated utility  
        if exitmsg ~= -2 
            val_dp = val_dp+1;
            AS_TEST = [AS_TEST;AS_diag']; % each row contains p(x|x) for all x in CIFAR-10, covariance in dataset : i/100
            AS_TEST_BIG = [AS_TEST_BIG, AS_FULL];
            AS_PRED = [AS_PRED;diag(reshape(act_sel_pred,[10,10]))'];
            AS_PRED_BIG = [AS_PRED_BIG,reshape(act_sel_pred,[10,10])'];
        end
    end
end



%FOR ALL BUT LENET
% for i=105:10:195
%         disp(['noisecov=',num2str(i/100)]);
%         iter=iter+1;
%         load(sprintf('AlexNet_noise_test/AlexNet/DATA_AlexNet_noise_%d.mat',i)); % ARCHITECTURE DEPENDENT
%         AS_diag = 10*diag(reshape(all_prob_std_105.JOINT_PROB,[10,10]));
%         AS_FULL = reshape(10*all_prob_std_105.JOINT_PROB',[X,X]); AS_FULL = AS_FULL';
%         %%%
%         load('robust_AlexNet_noise.mat'); % load interpretable model  % ARCHITECTURE DEPENDENT
%         %load('robust_VGG16_noise_1.mat'); % load interpretable model  % ARCHITECTURE DEPENDENT, only for VGG16
%       
%         util_costs=BIG*x(1:K*X*X + K); % interpretable model
%         ind_range = (iter-1)*X*X + 1 : iter*X*X; 
%         start = (JP(ind_range)' + JP(ind_range+X*X)')/2;
%         my_util = (util_costs(ind_range) + util_costs(ind_range+X*X))/2; % interpolated utility for noise covariance 1.05/1.15/1.25
%         
%         
%         %%%   
%         [act_sel_pred,exitmsg] = find_best_act_sel(K,X,JP,util_costs,my_util,start); %predicted act_sel given interpolated utility  
%         if exitmsg ~= -2 
%             val_dp = val_dp+1;
%             AS_TEST = [AS_TEST;AS_diag']; % each row contains p(x|x) for all x in CIFAR-10, covariance in dataset : i/100
%             AS_TEST_BIG = [AS_TEST_BIG, AS_FULL];
%             AS_PRED = [AS_PRED;diag(reshape(act_sel_pred,[10,10]))'];
%             AS_PRED_BIG = [AS_PRED_BIG,reshape(act_sel_pred,[10,10])'];
%         end
% end

ERR=mean(abs(AS_TEST-AS_PRED),1);
KLD=0;
KLD_mat = AS_TEST_BIG.*log(AS_TEST_BIG./AS_PRED_BIG);
KLD = sum(KLD_mat,'all')/(10*val_dp);

