library(stringr)
#pdf_txt 파일 라인별로 읽어옴
scr_dir<-c("D:/단국대학교/연구실/빅데이터안과진단기술/안과데이터/Combine/combine_txt")
scr_file <- list.files(scr_dir)


f<-matrix(ncol=24)
colnames(f)<-c("OD/OS","Sid","Sex","MD","PSD","Disc Area","Average GCL+IPL Thickness",
               "Minimum GCL + IPL Thickness","Age",
               "Average C/D Ratio","Rim Area",
               "Vertical C/D Ratio","Cup Volume","Avg_RNFL_Thickness",
               "Zone1","Zone2","Zone3","Zone4","Zone5","Zone6","RNFL_T","RNFL_S","RNFL_N","RNFL_I")


for(x in 1:length(scr_file)){
  data<-str_squish(readLines(scr_file[x])) #여분공백제거
  
  v_od<-c("OD",scr_file[x]) #od정보 저장하는 벡터 
  v_os<-c("OS",scr_file[x]) #os정보 저장하는 벡터
  
  csv_index<-3 #Feature 번호
  md_count<-1  #od/os 구분용
  psd_count<-1 #od/os 구분용
  vfi_count<-1 #od/os 구분용
  ght_count<-1
  zone_count<-0 #1~6 구분용
  age_count<-0 #age 획득 검사용
  avg_rnfl_count<-0
  avg_cd_count<-0
  rim_count<-0
  ver_cd_count<-0
  cup_count<-0
  disc_count<-0
  avg_count<-0
  min_count<-0
  gen<-0
  
  
  GHT_func<-function(x){
    if(x=="Within"){
      return(1)
    }else if(x=="Borderline"){
      return(2)
    }else if(x=="Borderline/general"){
      return(3)
    }else if(x=="Abnormally"){
      return(4)
    }else if(x=="Outside"){
      return(5)
    }
  }
  
  #한 줄에서 feature 추출
  for(i in 1:length(data)){ 
    if(data[i]=="")next #""line 넘김
    
    #Avg_RNFL_Thickness #수정필요
    if(grepl("Average RNFL Thickness",data[i])){ #다음 토큰에 값 없으면 넘김
      if(avg_rnfl_count<1){
        avg_rnfl_count<-avg_rnfl_count+1
        next
      }
      v_od[csv_index]<-unlist(strsplit(data[i]," "))[4]
      v_os[csv_index]<-unlist(strsplit(data[i]," "))[13]
      csv_index<-csv_index+1
      next
    }
    #Average C/D Ratio
    if(avg_cd_count==0&&grepl("Average C/D Ratio",data[i])){
      v_od[csv_index]<- unlist(strsplit(data[i+1]," "))[1]
      v_os[csv_index]<- unlist(strsplit(data[i+1]," "))[7]
      csv_index<-csv_index+1
      avg_cd_count<-avg_cd_count+1
      next
    }
    
    #Rim Area
    if(grepl("Rim Area",data[i])){
      if(rim_count<2){
        rim_count<-rim_count+1
        next
      }
      v_od[csv_index]<- unlist(strsplit(data[i+1]," "))[1]
      v_os[csv_index]<- unlist(strsplit(data[i+1]," "))[5]
      csv_index<-csv_index+1
      rim_count<-rim_count+1
      next
    }
    
    #Vertical C/D Ratio
    if(ver_cd_count==0&&grepl("Vertical C/D Ratio",data[i])){
      v_od[csv_index]<- unlist(strsplit(data[i+1]," "))[1]
      v_os[csv_index]<- unlist(strsplit(data[i+1]," "))[5]
      csv_index<-csv_index+1
      ver_cd_count<-ver_cd_count+1
      next
    }
    
    #Cup Volume
    if(grepl("Cup Volume",data[i])){
      if(cup_count<2){
        cup_count<-cup_count+1
        next
      }
      v_od[csv_index]<- unlist(strsplit(data[i+1]," "))[1]
      v_os[csv_index]<- unlist(strsplit(data[i+1]," "))[5]
      csv_index<-csv_index+1
      cup_count<-cup_count+1
      next
    }
    
    #Disc Area
    if(disc_count==0&&grepl("Disc Area",data[i])){
      v_od[csv_index]<- unlist(strsplit(data[i]," "))[1]
      v_os[csv_index]<- unlist(strsplit(data[i+2]," "))[1]
      csv_index<-csv_index+1
      disc_count<-disc_count+1
      next
    }
    
    #Average GCL+IPL Thickness 
    if(avg_count==0&&grepl("Average GCL",data[i])){
      v_od[csv_index]<- unlist(strsplit(data[i-1]," "))[1]
      v_os[csv_index]<- unlist(strsplit(data[i+4]," "))[1]
      csv_index<-csv_index+1
      avg_count<-avg_count+1
      next
    }
    
    #Minimum GCL + IPL Thickness
    if(min_count==0&&grepl("Minimum GCL",data[i])){
      v_od[csv_index]<- unlist(strsplit(data[i-1]," "))[1]
      v_os[csv_index]<- unlist(strsplit(data[i+4]," "))[1]
      csv_index<-csv_index+1
      min_count<-min_count+1
      next
    }
    
    sp_data <- unlist(strsplit(data[i]," ")) #한 라인을 공백으로 분리해 일시적으로 sp_data에 저장
    
    for(k in 1:length(sp_data)){ #토큰별로 검사
      #MD 추출
      if(grepl("MD",sp_data[k])){ 
        if(md_count==1){ #os 저장
          v_od[csv_index]<-sp_data[k+1] 
          md_count<-md_count+1
          next #다음토큰(os 정보)읽음 #한 line에 od/os 정보 다 들어가 있음
        }else if(md_count==2){
          v_os[csv_index]<-sp_data[k+1] 
          csv_index<-csv_index+1
          md_count<-md_count+1 
          break  #os정보까지 저장되면 더이상 이 line엔 볼일 없음
        }
        
      }
      
      #PSD 추출
      if(grepl("PSD",sp_data[k])){ 
        if(psd_count==1){ #os 저장
          v_od[csv_index]<-sp_data[k+1] 
          psd_count<-psd_count+1
          next
        }else if(psd_count==2){
          v_os[csv_index]<- sp_data[k+1]
          csv_index<-csv_index+1
          psd_count<-psd_count+1 
          break 
        }
      }
      
      #VFI 추출
     # if(sp_data[k]=="VFI:"){ 
      #  if(vfi_count==1){    #od 저장
       #   v_od[csv_index]<-gsub("%","",sp_data[k+1]) 
        #  vfi_count<-vfi_count+1
        #  next
        #}else if(vfi_count==2){
         # v_os[csv_index]<-gsub("%","",sp_data[k+1]) 
          #vfi_count<-vfi_count+1 
          #csv_index<-csv_index+1
          #next 
      #  }
      #}
      

      
      #zone 1~6 추출
      if(zone_count<6&&sp_data[k]=="Zone"){
        v_od[csv_index]<-sp_data[k+2] #od
        v_os[csv_index]<-sp_data[k+10] #os
        csv_index<-csv_index+1
        zone_count<-zone_count+1
        break  #Zone1~6이 line별로 있기 때문에 다음 라인으로 넘어감
      }
      
      #T/S/N/I
      if(sp_data[k]=="Temporal"||sp_data[k]=="Superior"||sp_data[k]=="Nasal"||sp_data[k]=="Inferior"){
        v_od[csv_index]<-gsub("μm","",sp_data[k+2])  #od
        v_os[csv_index]<-gsub("μm","",sp_data[k+10]) #os
        csv_index<-csv_index+1
        break
      }
      
      #Age
      if(age_count==0&&sp_data[k]=="Age:"){
        v_od[csv_index]<-sp_data[k+1]
        v_os[csv_index]<-sp_data[k+1]
        csv_index<-csv_index+1
        age_count<-age_count+1
        break
      }
      
      #gender
      if(gen==0&&sp_data[k]=="Gender:"){
        v_od[csv_index]<-sp_data[k+1]
        v_os[csv_index]<-sp_data[k+1]
        csv_index<-csv_index+1
        gen<-gen+1
        break
      }
    }
  }
  f<-rbind(f,v_od) #matrix에 계속 추가하다가
  f<-rbind(f,v_os)
}
f<-f[-1,] #결측값 제거

#for문 벗어나면 마지막으로 matrix col name 지정해주고 csv에 넣어주기!!

write.csv(f,"Feature.csv")

warnings()

