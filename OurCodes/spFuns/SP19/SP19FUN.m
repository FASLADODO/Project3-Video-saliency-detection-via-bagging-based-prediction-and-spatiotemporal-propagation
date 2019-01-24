function [TPSPSAL_Img,TPSPSAL_RegionSal] = SP19FUN(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param)
switch param.sp19ID
    case '1'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_1(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '2'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_2(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '3'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_3(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '4'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_4(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '5'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_5(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '6'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_6(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '7'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_7(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '8'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_8(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '9'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_9(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '10'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_10(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '11'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_11(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '12'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_12(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '13'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun19_13(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
end

clear TPSAL1 CURINFOR cur_image MVF_Foward_f_fp param
end