function saveSALS(imwriteInfor, saliencyMapPATH,salNAME)
% 保存各阶段的显著性图
% 1 由KCR结果直接多尺度平均求和
% imwriteInfor.KCRSAL
% 2 时域传播结果直接多尺度平均求和
% imwriteInfor.TPSAL
% 3 时空域传播后直接多尺度平均求和
% imwriteInfor.ENDSAL
% 2016.08.18 22:19PM
% 
IMSAL1 = normalizeSal(imwriteInfor.KCRSAL);
IMSAL1 = uint8(255*IMSAL1);
imwrite(IMSAL1,[saliencyMapPATH,salNAME,'_kcr_salmap.png']) 
clear IMSAL1

IMSAL2 = normalizeSal(imwriteInfor.TPSAL);
IMSAL2 = uint8(255*IMSAL2);
imwrite(IMSAL2,[saliencyMapPATH,salNAME,'_tp_salmap.png']) 
clear IMSAL2

IMSAL3 = normalizeSal(imwriteInfor.ENDSAL);
IMSAL3 = uint8(255*IMSAL3);
imwrite(IMSAL3,[saliencyMapPATH,salNAME,'_end_salmap.png'])
clear IMSAL3

clear imwriteInfor saliencyMapPATH salNAME
end