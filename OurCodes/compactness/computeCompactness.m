function result = computeCompactness(regionSal,regionDist)
result = sum(regionDist.*(regionSal/(sum(regionSal(:))+eps)));
if result==0
    result =result + 0;
end
% result = sum(regionDist.*regionSal);
clear regionSal regionDist
end