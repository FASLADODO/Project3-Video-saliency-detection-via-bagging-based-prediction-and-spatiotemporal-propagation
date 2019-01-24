% 3 各特征分开计算距离，然后相结合  *****************************************
% 这里主要是 Lab & Flow(man/ori)  2016.11.24
% 采用列归一化，即对特征的每一维做 max-min归一化， 2016.11.25
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