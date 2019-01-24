function [xcenter,ycenter] = computeObjectCenter(refImage)
% 计算物体质心
% 2016.11.14 9:51AM
[r,c] = size(refImage);
row = 1:r;
row = row';
col = 1:c;
XX = repmat(row,1,c).*refImage;
YY = repmat(col,r,1).*refImage;
xcenter = sum(XX(:))/sum(refImage(:));% row
ycenter = sum(YY(:))/sum(refImage(:));% column
clear refImage
end