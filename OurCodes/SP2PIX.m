function [PSALS,imSAL] = SP2PIX(FULLSALS, spinfor)
% 超像素级图像转换为像素级的(optimal:像素级的传播)
% 多尺度结果直接求和取平均
% FULLSALS 全尺寸超像素级显著性图
% spinfor 多尺度分割信息
% 
SPSCALENUM = length(FULLSALS);
PSALS = cell(SPSCALENUM,1);
imSAL = 0;
for ss=1:SPSCALENUM
    tmpFULLSAL = FULLSALS{ss,1};
    tmpSPINFOR = spinfor{ss,1};
%     tmpPSAL = zeros(size(tmpSPINFOR.idxcurrImage));
    [height,width] = size(tmpSPINFOR.idxcurrImage);
    [tmpPSAL, ~] = CreateImageFromSPs(tmpFULLSAL, tmpSPINFOR.pixelList, height, width, true);
    
    % 分配到某一尺度，并累加
    PSALS{ss,1} = tmpPSAL;
    imSAL = imSAL + tmpPSAL;
    
    clear tmpFULLSAL tmpSPINFOR tmpPSAL
    
end
% imSAL = normalizeSal(imSAL);

imSAL = normal_enhanced(imSAL);% 多尺度求平均

clear FULLSALS spinfor

end

