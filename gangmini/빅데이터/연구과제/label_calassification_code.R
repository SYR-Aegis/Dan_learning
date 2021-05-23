getwd()
setwd("D:\\")

#csv 파일 불러오기
label <- read.csv(file="NEW_LABEL_210330.csv",header=T)
#head(label)  #정상처리 확인
#class(label) #데이터프레임


#녹내장 확정 환자의 정보를 담을 데이터프레임
patient<-data.frame(c(),c(),c()) #빈 데이터프레임

x<-1
count<-0
while(x<nrow(label)){ #seq(1,nrow(label),2)
  if(label[x,"id"]==label[x+1,"id"]){              #같은 환자인지 확인
    if(label[x,"label"]==1|label[x+1,"label"]==1){ #한쪽이라도 녹내장->녹내장
      count<-count+1
      #녹내장 환자 세부사항 저장
      if(label[x,"label"]==1){
        od<-"yes"
      }else{od<-"no"}
      if(label[x+1,"label"]==1){
        os<-"yes"
      }else{os<-"no"}
      patient<-rbind(patient,c(label[x,"id"],od,os))
    }
  }else{ 
    print("환자의 정보가 일부 누락되었습니다.")
    #break
    #한 id의 od,os 정보가 누락되었거나 올바르지 않게 정렬되어 있음을 알려줌
  }
  x<-x+2
}
colnames(patient)<-c("id","od","os")
count
patient
