#01
25+99 #1
456-123 #2
2*(3+4) #3
(3+5*6)/7 #4
(7-4)*3 #5
210+35 #6
1256%%7 #7
184%%5 #8
1976/24 #9
16*25+186*5-67*22 #10

#02
a <- 4
b <- 5
c <- a+b
c

#03
pi <- 3.14
10^2*pi
12^2*pi
15^2*pi

#04
my_f <- function(x){
  return(2*x^2+5*x+10)
  }
my_f(6)
my_f(8)
my_f(10)

#05
#변수 val.b에 대한 값이 없다!

#06
vc.1 <- c(95,86,47,55,68)
vc.1

#07
vc.2 <- c(seq(100,200,2))
vc.2

#08
vc.3 <- rep(c(TRUE),times=20)  #c(rep(TRUE,times=20))하면 오류.??
vc.3

#09
d <- c(seq(100,200))

d #1
d[10] #2
sort(d,decreasing = TRUE)[1:10] #3
d[d%%2==1] #4
d[d%%3==0] #5
d.20 <- d[1:20]  #6
d.20
d.20[-5] #7
d.20[-c(5,7,9)] #8

#10
absent<-c(10,8,14,15,9,10,15,12,9,7,8,7)
names(absent) <- c("JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC")

absent #1
absent["MAY"] #2
absent[c("JUL","SEP")] #3
sum(absent[1:6]) #4
mean(absent[7:12]) #5

#11
d1 <- 1:50
d2 <- 51:100

d1 #1
d2
length(d2) #2
d1+d2 #3 #같은 순서의 원소들끼리 계산
d2-d1
d1*d2
d2/d1
sum(d1) #4
sum(d2)
sum(d1)+sum(d2) #5
max(d2) #6
min(d2)
mean(d1) #7
mean(d2)
abs(mean(d1)-mean(d2)) 
sort(d1,decreasing = TRUE) #8
d3 <- c(sort(d1)[1:10],sort(d2)[1:10]) #9
d3

#12
v1 <- 51:90 

v1[v1<60] #1
length(v1[v1<70]) #2
sum(v1[v1>65]) #3
v1[v1>60&v1<73] #4
v1[v1<65|v1>80] #5
v1[v1%%7==3] #6
v1[v1%%7==0] <- 0 #7
v1

v1 <- 51:90 
sum(v1[v1%%2==0]) #8
v1[v1%%2==1|v1>80] #9
v1[v1%%(3*5)==0] #10
v1[v1%%2==0] <- v1[v1%%2==0]*2 #11
v1

v1 <- 51:90 
v1<-v1[v1%%7!=0] #12
v1
