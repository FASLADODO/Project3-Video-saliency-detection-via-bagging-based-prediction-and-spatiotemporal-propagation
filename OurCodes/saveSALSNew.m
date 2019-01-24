function saveSALSNew(imwriteInfor, saliencyMapPATH,salNAME)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 保存各阶段的显著性图
% % 1. 直接的分类结果
% imwriteInfor.IMSAL_BOOST_SALS0
% imwriteInfor.IMSAL_BOOST_SALS1
% 
% % 2. 仅时域传播的结果
% imwriteInfor.IMSAL_TPSAL0 
% imwriteInfor.IMSAL_TPSAL1 
% 
% % 3. 时空域传播后的结果
% imwriteInfor.IMSAL_SPSAL0 
% imwriteInfor.IMSAL_SPSAL1 
% 
% V1: 2016.08.27 14:06PM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1 直接的分类结果
% IMSAL1 = normalizeSal(imwriteInfor.IMSAL_BOOST_SALS0);
% IMSAL1 = uint8(255*IMSAL1);
% imwrite(IMSAL1,[saliencyMapPATH,salNAME,'_IMSAL_BOOSTSALS0.png']) 
% clear IMSAL1

IMSAL1 = normalizeSal(imwriteInfor.IMSAL_BOOST_SALS1);
IMSAL1 = uint8(255*IMSAL1);
imwrite(IMSAL1,[saliencyMapPATH,salNAME,'_IMSAL_BOOSTSALS1.png']) 
clear IMSAL1

% 2 仅时域传播的结果
% IMSAL1 = normalizeSal(imwriteInfor.IMSAL_TPSAL0);
% IMSAL1 = uint8(255*IMSAL1);
% imwrite(IMSAL1,[saliencyMapPATH,salNAME,'_IMSAL_TPSAL0.png']) 
% clear IMSAL1

IMSAL1 = normalizeSal(imwriteInfor.IMSAL_TPSAL1);
IMSAL1 = uint8(255*IMSAL1);
imwrite(IMSAL1,[saliencyMapPATH,salNAME,'_IMSAL_TPSAL1.png']) 
clear IMSAL1

% 3 时空域传播后的结果
% IMSAL1 = normalizeSal(imwriteInfor.IMSAL_SPSAL0);
% IMSAL1 = uint8(255*IMSAL1);
% imwrite(IMSAL1,[saliencyMapPATH,salNAME,'_IMSAL_SPSAL0.png']) 
% clear IMSAL1

IMSAL1 = normalizeSal(imwriteInfor.IMSAL_SPSAL1);
IMSAL1 = uint8(255*IMSAL1);
imwrite(IMSAL1,[saliencyMapPATH,salNAME,'_IMSAL_SPSAL1.png']) 
clear IMSAL1

clear imwriteInfor saliencyMapPATH salNAME
end