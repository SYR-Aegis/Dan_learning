#01-1
sum<-0  
count<-0
for(x in 1:100){
  if(x%%3==0){
    sum <- sum+x
    count <- count+1
  }
}
sum
count

#01-2
for(x in 101:200){ 
  if(x%%3==0&&x%%4==0){
    print(x)
  }
}

#01-3
for(x in 1:24){
  if(24%%x==0){
    print(x)
  }
}

#01-4
x <- 1
for(y in 1:9){
  x <- x*y
}
x

#02
for(x in 1:9){
  print(paste("9 * ",x,"=",9*x)) #문자열을 연결해주는 paste
}

#03
x <- 1
while(x<9){
  print(paste("8 * ",x,"=",8*x)) 
  x <- x+1
}

#04
for(x in 1:100){
  if(x%%3==0){
    print('*')
  }else{
    print(x)
  }
}

#05
for(x in 2:1000){
  if(x%%2 !=0){
    print(x)
  }
}

#06
a1<-0
a2<-1
print(a1)
print(a2)
for(x in 1:38){
  print(a1+a2)
  tmp <- a1
  a1<-a2
  a2<-tmp+a2
}
#07-1 
iris
apply(iris[,-5],1,sum)
#07-2
apply(iris[,-5],2,max)

#08-1
apply(mtcars,2,sum)
#08-2
apply(mtcars,2,max)
#08-3
sqrt(apply(mtcars,2,var))#sd

#09
lgm <- function(x,y){
  result_1<-1
  if(x<y){  #이 부분을 벡터로 만들어서 max사용
    small <- x
  }else{
    small <- y
  }
  for(i in 1:small){
    if(x%%i==0 && y%%i==0){
      result_1 = c(result_1,i)
    }
  }
  return(max(result_1))
}
result_2<-lgm(10,8)
result_2
result_2<-lgm(10,20)
result_2

#10
maximin <- function(v1){
  val_max<-max(v1)
  val_min<-min(v1)
  return(list(max=val_max,min=val_min))
}
v1<-c(7,1,2,8,9)
result<-maximin(v1)
result$max;result$min
result<-maximin(iris[,1])
result$max;result$min

#11-1
weight<-c(69,50,55,71,89,64,59,70,71,80)
which.max(weight)
#11-2
which.min(weight)
#11-3
which(61<=weight&weight<=69)#포함 시킴
which(61<weight&&weight<69)#포함 안 시킴
#11-4
weight.2<-which(weight<=60)
weight[weight.2]

#12-1
iris 
which.max(iris[,"Petal.Length"])
iris[which.max(iris[,"Petal.Length"]),"Petal.Length"]
#12-2
iris[which(0.3<=iris[,"Petal.Width"]&iris[,"Petal.Width"]<=0.4),"Petal.Width"]
iris[which(0.3<=iris[,"Petal.Width"]&iris[,"Petal.Width"]<=0.4),"Petal.Width"]








