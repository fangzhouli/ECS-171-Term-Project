# generate fake data as placeholder for feature.csv
library(data.table) # install.packages('data.table')
set.seed(1234)
# 10000 samples x sampeID+(10 features)+result
n_samp <- 10000
dt <- data.table(matrix(nrow=n_samp, ncol = 12))
colnames(dt) <- c('sampleID', paste0('a_f', 1:5), paste0('b_f', 1:5), 'win')

dt[, 'sampleID'] <- 1:10000
dt[, 'a_f1'] <- round(rnorm(n_samp, 10000, 800))
dt[, 'b_f1'] <- round(rnorm(n_samp, 10000, 800))
dt[, 'a_f2'] <- round(rnorm(n_samp, 9000, 200))
dt[, 'b_f2'] <- round(rnorm(n_samp, 9000, 200))
dt[, 'a_f3'] <- round(rnorm(n_samp, 6000, 120))
dt[, 'b_f3'] <- round(rnorm(n_samp, 6000, 120))
dt[, 'a_f4'] <- round(rnorm(n_samp, 4000, 150))
dt[, 'b_f4'] <- round(rnorm(n_samp, 4000, 150))
dt[, 'a_f5'] <- round(rnorm(n_samp, 1000, 210))
dt[, 'b_f5'] <- round(rnorm(n_samp, 1000, 210))

dt[, win := as.numeric((0.5*(a_f1-b_f1) + 0.3*(a_f2-b_f2) - 0.6*(a_f3-b_f3)
                        + 0.4*(a_f4-b_f4) - 0.4*(a_f5-b_f5)
                        + runif(n_samp, -250, 250)) >= 0)]

write.csv(dt, file = 'feature.csv', row.names=F)
