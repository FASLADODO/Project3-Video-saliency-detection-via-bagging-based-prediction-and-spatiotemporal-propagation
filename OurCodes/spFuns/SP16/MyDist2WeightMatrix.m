function weightMatrix = MyDist2WeightMatrix(distMatrix, distSigma)
% Transform pair-wise distance to pair-wise weight using
% exp(-d^2/(2*sigma^2));
% 新的归一化方式，即方差的选择
%

spNum = size(distMatrix, 1);

% distMatrix(distMatrix > 3 * distSigma) = Inf;   %cut off > 3 * sigma distances
% weightMatrix = exp(-distMatrix.^2 ./ (2 * distSigma * distSigma));



if any(1 ~= weightMatrix(1:spNum+1:end))
    error('Diagonal elements in the weight matrix should be 1');
end