clear;
clc;

JP=[];CP=[];PA=[];AS=[];
% for i=10:20
%     load(sprintf('LeNet_noise_test/DATA_LeNet_noise_%d.mat',i))
%     eval(['JP = horzcat(JP,all_prob_std_',int2str(i),'.JOINT_PROB);']);
%     eval(['CP = horzcat(CP,all_prob_std_',int2str(i),'.COND_PROB);']);
%     eval(['PA = horzcat(PA,all_prob_std_',int2str(i),'.PROB_ACT);']);
%     eval(['AS = horzcat(AS,all_prob_std_',int2str(i),'.ACT_SEL);']);
% end


% for i=100:10:200
%     load(sprintf('VGG16_noise_test/VGG16/DATA_VGG16_noise_%d.mat',i))
%     JP = horzcat(JP,all_prob_std_105.JOINT_PROB);
%     CP = horzcat(CP,all_prob_std_105.COND_PROB);
%     PA = horzcat(PA,all_prob_std_105.PROB_ACT);
%     AS = horzcat(AS,all_prob_std_105.ACT_SEL);
% end

K = 11; % for testing only one batch of trained CNNs
X=10;
% REPAIR COND_PROB
CP_NEW = [];
for k=1:K
    CB = CP((k-1)*X*X + 1 : k*X*X );
    CB = reshape(CB,[10,10]);  
    CB = CB';
    CB = reshape(CB,[1,100]);
    CP_NEW = [CP_NEW CB];
end
CP = CP_NEW;

load('robust_VGG16_noise.mat');
start = x;
[r,x] = fmincon_feasibility_general(K,X,CP(1:X*X*K),PA(1:X*K),JP(1:X*X*K),start);