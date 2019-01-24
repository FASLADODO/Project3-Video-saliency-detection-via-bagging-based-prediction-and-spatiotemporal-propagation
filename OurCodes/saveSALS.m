function saveSALS(imwriteInfor, saliencyMapPATH,salNAME)
% ������׶ε�������ͼ
% 1 ��KCR���ֱ�Ӷ�߶�ƽ�����
% imwriteInfor.KCRSAL
% 2 ʱ�򴫲����ֱ�Ӷ�߶�ƽ�����
% imwriteInfor.TPSAL
% 3 ʱ���򴫲���ֱ�Ӷ�߶�ƽ�����
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