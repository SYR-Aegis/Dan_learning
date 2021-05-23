install.packages("pdftools")
install.packages("stringr")

library(pdftools)
one_text<-pdf_text("D806_COMB_1.pdf")
write.table(one_text,"D806_COMB_1.txt")

save(one_text,file = "D802_COMB_1.txt") //±ÛÀÚ°¡ ±úÁü
