#01
edu<-infert$education#1
edu
unique(edu)#2
ed<-table(edu)#3
ed
barplot(ed,main='education period')#4

#02
gender<-c('F','F','F','M','M','F','F','F','M','M')#1
gen<-table(gender)#2
barplot(gen)#3
pie(gen)#4

#03
season<-c('여름','겨울','봄','가을','여름','가을','겨울','여름','여름','가을')#1
ss<-table(season)#2
barplot(ss)#3
pie(ss)#4

#04
score<-c(90,85,73,80,85,65,78,50,68,96)#1
names(score)<-c('KOR','ENG','ATH','HIST','SOC','MUSIC','BIO','EARTH','PHY','ART')
score#2
mean(score) #3
median(score)
sd(score)#4
index <- which.max(score) #5
names(score[index])
boxplot(score)#6
boxplot.stats(score)
#$out
boxplot.stats(score)$out
hist(score,main='학생 성적',col="purple")#7

#05
mean(mtcars$wt)#1
median(mtcars$wt)
mean(mtcars$wt,trim=0.15)
sd(mtcars$wt)

summary(mtcars$wt)#2
bar<-table(mtcars$cyl)#3
bar
barplot(bar)#4
hist(mtcars$wt)#5
boxplot(mtcars$wt)#6
summary(mtcars$wt)
boxplot(mtcars$disp)#7
summary(mtcars$disp)
boxplot(mtcars$mpg~mtcars$gear)#8 #기어에 따른 연비수

#06
head(trees)#1

mean(trees$Girth)#2
median(trees$Girth)
mean(trees$Girth,trim=0.15) 
#절사평균은 관측값의 크기순으로 정렬한 후, 
#왼쪽 끝과 오른 쪽에 위치하는 이상치들을 같은 개수로 제거 한 후
#남아있는 관측값의 산술평균을 의미
sd(trees$Girth)

hist(trees$Girth)#3

boxplot(trees$Girth)#4

mean(trees$Height)#5
median(trees$Height)
mean(trees$Height,trim=0.15)
sd(trees$Height)

hist(trees$Height)#6

boxplot(trees$Height)#7

#07
head(Orange)#1

mean(Orange$age)#2
median(Orange$age)
mean(Orange$age,trim=0.15)
sd(Orange$age)

hist(Orange$age)#3
boxplot(Orange$age~Orange$Tree)#4

mean(subset(Orange,Tree!=2)[,'circumference']) #5
median(subset(Orange,Tree!=2)[,'circumference'])
mean(subset(Orange,Tree!=2)[,'circumference'],trim=0.15)
sd(subset(Orange,Tree!=2)[,'circumference'])

hist(Orange$circumference)#6
boxplot(Orange$circumference~Orange$Tree)#7
