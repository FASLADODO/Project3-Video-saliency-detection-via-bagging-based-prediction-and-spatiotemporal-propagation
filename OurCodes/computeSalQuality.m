function [salquality] = computeSalQuality(PPSal,tmpSPinfor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 对于显著性图，计算对应的质量评价分数，以此构造 bagging 范式
% copyright by xiaofei zhou, IVPLab, shanghai university,shanghai,china
% zxforchid@163.com; http://www.ivp.shu.edu.cn/
% V1: 2016.12.08 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0. initial & obtain object center
regionCenter = tmpSPinfor.region_center;
[height,width] = size(tmpSPinfor.idxcurrImage);
[PP_Img, ~]  = CreateImageFromSPs(PPSal, tmpSPinfor.pixelList, height, width, true);
[rcenter_PP,ccenter_PP] = computeObjectCenter(PP_Img);
rcenter_PP = round(rcenter_PP);
ccenter_PP = round(ccenter_PP);

% 1. the ratio of center-surround
WIDTH = width/2;
HEIGHT = height/2;
RB = round(rcenter_PP-HEIGHT/2);
RE = round(rcenter_PP+HEIGHT/2);
CB = round(ccenter_PP-WIDTH/2);
CE = round(ccenter_PP+WIDTH/2);
if RB <=0
    RB = 1;
end
if RE >height
    RE = height;
end
if CB <=0
    CB = 1;
end
if CE >width
    CE = width;
end
% salfg = 0;
% for i=RB:RE
%     for j=CB:CE
%         salfg = salfg + PP_Img(i,j);
%     end
% end
salfg = PP_Img(RB:RE,CB:CE);
salfg = sum(salfg(:));
salbg = sum(PP_Img(:)) - salfg;
if salbg==0
    salbg = salbg + eps;
end
fbRatio = salfg/salbg;

% 2. saliency distribution
row = 1:height;
row = row';
col = 1:width;
XX = repmat(row,1,width)  - rcenter_PP;
YY = repmat(col,height,1) - ccenter_PP;
ds = sum(sum(PP_Img.*(XX.^2 +YY.^2)));
if ds==0
    ds = ds + eps;
end
ds = 1/ds;

% saliency variance
vs = var(PP_Img(:));

salquality = [fbRatio,ds,vs];

clear PPSal tmpSPinfor
end