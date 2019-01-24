function [TPSPSAL_Img,TPSPSAL_RegionSal] = SP17FUN(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param)
switch param.sp17ID
    case '0'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun17_0(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '1'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun17_1(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '2'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun17_2(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '3'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun17_3(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '4'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun17_4(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '5'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun17_5(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '6'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun17_6(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '7'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun17_7(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '9'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun17_9(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
end

clear TPSAL1 CURINFOR cur_image MVF_Foward_f_fp param
end