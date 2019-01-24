function interA = computeOverlap(sppre,spcur,tmpflow)
% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% 计算两帧之间的重叠度
% interA 当前帧超像素数 * 前一帧超像素数（或者下一帧；主要是由cur进行投影）
% 
% 2016.10.09 15:03PM
% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%1 initialize
num_curSP = max(max(spcur));
num_nextSP = max(max(sppre));
Index_Frames{1,1} = uint16(spcur); % current
Index_Frames{2,1} = uint16(sppre); % next
numOfFrames = 1;
flow{1,1} = tmpflow;% cur ---> next/pre

%2 convert the flow to int16
if ~isa(flow{1,1},'int16')
    flow_int16=cell(1,numOfFrames);
    for frame_it=1:numOfFrames
        flow_int16{1,frame_it}=int16(flow{1,frame_it});
    end
    flow=flow_int16;
end

%3 make the superpixels have a unique label 
[superpixels, nodeFrameId, bounds, labels] =makeSuperpixelIndexUnique(Index_Frames); 

%4 connections
[tSource, tDestination, tConnections ]=getTemporalConnections( flow, superpixels, labels );  

%5 construct n_pre*n_cur affinity matrix
interA=zeros(num_curSP,num_nextSP);  
tSource=tSource+1;
tDestination=tDestination+1;
Start=bounds(1);    % first superpixel label for current frame
End=bounds(1+1)-1; 
for sp=Start:End        
    [r,~]=find(tSource==sp);
     neighborhood=tDestination(r);
     %to get the acutal label for neighborhood, -End
     neighborhood=neighborhood-End;
     affinity=tConnections(r);
     interA(sp-Start+1,neighborhood)=affinity;
end

clear sppre spcur tmpflow
end