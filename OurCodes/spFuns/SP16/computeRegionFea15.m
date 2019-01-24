function regionFea = computeRegionFea15(image,flow,tmpSPinfor)
% ÌØÕ÷ Lab/Man,Ori/Y,X
% 2016.11.26
% copyright by xiaofei zhou,shanghai university
% 
[height,width,dims] = size(image);
meanRgbCol = GetMeanColor(image, tmpSPinfor.pixelList);
meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255);
clear image

curFlow = double(flow);
Magn    = sqrt(curFlow(:,:,1).^2+curFlow(:,:,2).^2);    
Ori     = atan2(-curFlow(:,:,1),curFlow(:,:,2));
meanMagn = GetMeanColor(Magn, tmpSPinfor.pixelList);
meanOri  = GetMeanColor(Ori, tmpSPinfor.pixelList);
clear Ori Magn flow


normXY = sqrt(height^2+width^2);
im_Y = repmat([1:height]',[1,width])./normXY;
im_X = repmat([1:width], [height,1])./normXY;
meanY = GetMeanColor(im_Y, tmpSPinfor.pixelList);
meanX  = GetMeanColor(im_X, tmpSPinfor.pixelList);
clear im_X im_Y normXY

% [height,width] = size(im_L);
% regionFea = zeros(tmpSPinfor.spNum,5);
regionFea = [meanLabCol,meanMagn,meanOri,meanY,meanX];

clear meanLabCol meanMagn meanOri meanY meanX
clear  tmpSPinfor

end