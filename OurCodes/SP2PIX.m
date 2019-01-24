function [PSALS,imSAL] = SP2PIX(FULLSALS, spinfor)
% �����ؼ�ͼ��ת��Ϊ���ؼ���(optimal:���ؼ��Ĵ���)
% ��߶Ƚ��ֱ�����ȡƽ��
% FULLSALS ȫ�ߴ糬���ؼ�������ͼ
% spinfor ��߶ȷָ���Ϣ
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
    
    % ���䵽ĳһ�߶ȣ����ۼ�
    PSALS{ss,1} = tmpPSAL;
    imSAL = imSAL + tmpPSAL;
    
    clear tmpFULLSAL tmpSPINFOR tmpPSAL
    
end
% imSAL = normalizeSal(imSAL);

imSAL = normal_enhanced(imSAL);% ��߶���ƽ��

clear FULLSALS spinfor

end

