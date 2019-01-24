% 3 �������ֿ�������룬Ȼ������  *****************************************
% ������Ҫ�� Lab & Flow(man/ori)  2016.11.24
% �����й�һ��������������ÿһά�� max-min��һ���� 2016.11.25
function FeaDist = feaDistIntegrate(regionFea)

for iii=1:size(regionFea,2)
    regionFea(:,iii) = normalizeSal(regionFea(:,iii));
end

FeaDist = zeros(size(regionFea,1),size(regionFea,1));
for i=1:size(regionFea,1)
    tmpI = regionFea(i,:);
    for j=1:size(regionFea,1)
        tmpJ = regionFea(j,:);
        tmpD = sum((tmpI - tmpJ).*(tmpI - tmpJ));
        FeaDist(i,j) = sqrt(tmpD);
        
        clear tmpJ tmpD
    end
    clear tmpI
end

clear regionFea

end