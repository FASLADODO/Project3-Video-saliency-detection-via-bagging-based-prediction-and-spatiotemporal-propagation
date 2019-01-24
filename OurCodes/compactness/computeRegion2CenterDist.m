function dist = computeRegion2CenterDist(regionCenter,objectCenter,imageSize)
% 计算区域中心至物体质心的距离
% 2016.11.14 10:07AM
%
sigmaRatio = 0.25;
rcenter = objectCenter(1);
ccenter = objectCenter(2); 
r = imageSize(1);
c = imageSize(2); 
dist = zeros(size(regionCenter,1),1);
sigma=[r*sigmaRatio c*sigmaRatio];

for i=1:length(regionCenter)
    tmpRegionCenter = regionCenter(i,:);
    yy = tmpRegionCenter(1);% width column
    xx = tmpRegionCenter(2);% height row
    dist(i,1) = (xx-rcenter)^2+(yy-ccenter)^2;
%     dist(i,1) = exp(-(xx-rcenter)^2/(2*sigma(1)^2)-(yy-ccenter)^2/(2*sigma(2)^2));
end

% dist = normalizeSal(dist);
% dist = dist/sum(dist);% 归一化 2016.11.10 14:46PM

clear regionCenter objectCenter imageSize
end