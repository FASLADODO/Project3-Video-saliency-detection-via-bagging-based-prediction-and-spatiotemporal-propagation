function result = computeCorreDist(sals)
% ��������ֵΪ����������ÿ��map��Ӧ����ؾ��루С�ã�
% sals : spNum*feaNum
% 2016.12.16
% copyright by xiaofei zhou,shanghai university
% 
[spNum,Nums] = size(sals);
corre_matrixs = zeros(Nums,Nums);
for i=1:Nums
    tmp_i = sals(:,i);
    for j=i:Nums
        tmp_j = sals(:,j);
        diffs = (tmp_i - tmp_j).^2;
        diffs = sqrt(sum(diffs(:)));
        corre_matrixs(i,j) = diffs;
        corre_matrixs(j,i) = diffs;
        clear diffs
    end
end
result = sum(corre_matrixs,2);
clear corre_matrixs sals
end