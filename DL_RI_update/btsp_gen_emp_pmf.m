function emp_pmf = btsp_gen_emp_pmf(pmf,dim)
%BTSP_GEN_EMP_PMF Summary of this function goes here
%   Detailed explanation goes here
pmf=pmf/sum(pmf);
emp_pmf = zeros(1,dim);
num_iter=10000;
act_iter=zeros(1,num_iter);
for iter = 1: num_iter
    %iter
    val_iter=rand;
    %val_iter
    sum_run=0;
    
    for dim_iter=1:dim
        %dim_iter
        if val_iter>=sum_run && val_iter<=sum_run+pmf(dim_iter)
            act_iter(iter)=dim_iter;
            %sum_run
            break
        else
            sum_run=sum_run+pmf(dim_iter);
        end
    end
end

for dim_iter=1:dim
    emp_pmf(dim_iter) = sum(act_iter==dim_iter)/num_iter;
end
    
end

