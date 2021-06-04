install.packages("mlbench") #라이브러리 설치
library(mlbench)
par("mar") #plot시, margin 너무 작아서 다시 설정
par(mar=c(1,1,1,1))


#8
year<-c(2015:2026)
population<-c(51014,51245,51446,51635,51811,51973,52123,52261,52388,52504,52609,52704)
plot(year,population,main="예상 인구수 추계 그래프",type="l",xlab="year",ylab="population")

#9
year<-c(20144,20151,20152,20153,20154,20161,20162,20163,20164,20171,20172,20173)
male<-c(73.9,73.1,74.4,74.2,73.5,73,74.2,74.5,73.8,73.1,74.5,74.2)
female<-c(51.4,50.5,52.4,52.4,51.9,50.9,52.6,52.7,52.2,51.5,53.2,53.1)
plot(year,male,main="경제활동참가율",type="b",lty=1,xlab="year",col="blue",ylim=c(50,75))
lines(year,female,type="b",col="red")
#10
tbl<-data.frame(trees$Girth,trees$Height)#01
tbl
plot(tbl,main="나무의 지름과 높이의 산점도",xlab="지름",ylab="높이",col="red",pch=19)

plot(trees$Girth,trees$Height,main="나무의 지름과 높이의 산점도",xlab="지름",ylab="높이",col="red",pch=19)
cor(trees$Girth,trees$Height)

trees#02
trees
pairs(trees[,])
cor(trees[,])
    