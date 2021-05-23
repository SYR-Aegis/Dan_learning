#1
score <- matrix(c(10,40,60,20,21,60,70,30),4,2) #1
colnames(score) <- c('m','f')
score

colnames(score) <- c('male','female') #2
score

score[2,] #3
score[,'female'] #4
score[3,2] #5

#2
st <- data.frame(state.x77) #1
st #2
colnames(st) #3
rownames(st) #4
dim(st) #5 #열,행 개수 한 번에
str(st) #6
rowSums(st) #7
rowMeans(st)
colSums(st) #8
colMeans(st) 
st['Florida',] #9
st[1:50,"Income"] #10
st["Texas","Area"] #11
st["Ohio",c("Population","Income")] #12
subset(st,Population>=5000) #13

subset(st,Income>=4500)[,c("Population","Income","Area")]#14
nrow(subset(st,Income>=4500)) #15

length(c(subset(st,Area>=1000000&Frost>=120)))#16
str(subset(st,Population<2000&Murder<12))#17
colMeans(subset(st,Illiteracy>=2.0))#18
#19
#20
#21

#3
class(mtcars) #1
dim(mtcars) #2
class(c(mtcars[1,]))#3#모르겠....
#Help검색창에 mtcars치면 칼럼에 대한 설명 나옴 #4
rownames(subset(mtcars,mpg==max(mtcars[,"mpg"])))#5
rownames(subset(mtcars,mpg==min(subset(mtcars,gear==4)[,"mpg"])))#6
mtcars["Honda Civic",c("mpg","gear")]#7
subset(mtcars,mpg>mtcars["Pontiac Firebird","mpg"])#8
mean(c(mtcars[,"mpg"]))#9
unique(mtcars[,"gear"])#10  

#4
class(airquality) #1
head(airquality) #2
subset(airquality,Temp==max(airquality[,"Temp"]))[,c("Month","Day")] #3
 #4
mean(airquality[-"NA","Ozon"])#6          
       
       
       
       