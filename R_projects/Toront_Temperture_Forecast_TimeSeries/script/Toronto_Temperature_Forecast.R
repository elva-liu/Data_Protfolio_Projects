library(TSA)
library(tseries)
library(forecast)

# load the data
df <- read.csv("6A03/Toronto_20years_avgTemperature.csv",sep=',',header = TRUE)
str(df)

# convert Date column data type
df$Date <- as.Date(df$Date, format = "%Y-%m-%d")

df <- df[order(df$Date),]
# check missing value
sum(is.na(df))
# check for duplicate dates
any(duplicated(df$Date))
#check Date range 
range(df$Date)
# check outliers
plot(df$Date,df$Avg.Temperature)

boxplot(df$Avg.Temperature)

## training/test split
# training: from 1999-01-31 to 2017-12-31
train_data = df[df$Date <= as.Date("2017-12-31"), ]
train_ts = ts(train_data$Avg.Temperature, start = c(1999, 1), frequency = 12)

# test: from 2018-01-31 to 2019-12-31
test_data = df[df$Date > as.Date("2017-12-31"),]
test_ts = ts(test_data$Avg.Temperature, start = c(2018, 1), frequency = 12)

summary(train_ts)
str(train_ts)

# plot training data
plot(train_ts,
     main= "Toronto Monthly Average Temperature (1999-2017)",
     xlab = "Year",
     ylab = "Avg Temperature (°C)")

decomp <- decompose(train_ts)
plot(decomp)

stl_decomp <- stl(train_ts, s.window = "periodic")
plot(stl_decomp, 
     main = "Decomposed Training Series")

# Augmented Dickey-Fuller Test for stationarity
adf_result <- adf.test(train_ts)
adf_result

##------------------ Modeling -----------------------
# plot sample ACF and PACF 
acf(train_ts, lag.max = 60, main = "ACF of Toronto Temperature (1999-2017)", 
    xlab = "Lag (Months)", xaxt = "n")
axis(1, at = seq(0, 60, by = 6) / 12, labels = seq(0, 60, by = 6))

pacf(train_ts, lag.max = 60, main = "PACF of Toronto Temperature (1999-2017)", 
     xlab = "Lag (Months)", xaxt = "n")
axis(1, at = seq(0, 60, by = 6) / 12, labels = seq(0, 60, by = 6))

# seasonal differentiation
train_ts_seasdiff <- diff(train_ts, lag = 12)

adf_seasdiff <- adf.test(train_ts_seasdiff)
adf_seasdiff

acf(train_ts_seasdiff, lag.max = 60, main = "ACF of Seasonally Differenced", 
    xlab = "Lag (Months)", xaxt = "n")
axis(1, at = seq(0, 60, by = 6) / 12, labels = seq(0, 60, by = 6))

pacf(train_ts, lag.max = 60, main = "PACF of Seasonally Differenced", 
     xlab = "Lag (Months)", xaxt = "n")
axis(1, at = seq(0, 60, by = 6) / 12, labels = seq(0, 60, by = 6))

# modeling
# MA(q), q=1, Q=1
# AR(p), p=1, P=1
# d=0, D=1, seasonal differencing

# Fit initial model: SARIMA(1,0,1)(1,1,1)[12]
model_1 <- Arima(train_ts, order = c(1, 0, 1), seasonal = list(order = c(1, 1, 1), period = 12))
summary(model_1)

# model diagnostics
checkresiduals(model_1)
qqnorm(model_1$residuals)
qqline(model_1$residuals, col = "red")
acf(model_1$residuals, lag.max = 60, main = "ACF of Residuals", 
    xlab = "Lag (Months)", xaxt = "n")
axis(1, at = seq(0, 60, by = 6) / 12, labels = seq(0, 60, by = 6))
############### residual outliers analysis ########################
# find the outliers in residuals of model_1
residuals_1 <- residuals(model_1)
boxplot(residuals_1)

Q1_res <- quantile(residuals_1, 0.25)
Q3_res <- quantile(residuals_1, 0.75)
IQR_res <- Q3_res - Q1_res
lower_bound_res <- Q1_res - 1.5 * IQR_res
upper_bound_res <- Q3_res + 1.5 * IQR_res

res_outliers <- residuals_1[residuals_1 < lower_bound_res | residuals_1 > upper_bound_res]
res_outlier_dates <- train_data$Date[residuals_1 < lower_bound_res | residuals_1 > upper_bound_res]

res_outlier_info <- data.frame(Date = res_outlier_dates, Residual = res_outliers)
print(res_outlier_info)

# 3 standard deviation outlier
res_mean <- mean(residuals_1)
res_sd <- sd(residuals_1)
# residuals exceeding 3 standard deviations
res_outliers_3sd <- residuals_1[abs(residuals_1 - res_mean) > 3 * res_sd]
res_outlier_dates_3sd <- train_data$Date[abs(residuals_1 - res_mean) > 3 * res_sd]

res_outlier_info_3sd <- data.frame(Date = res_outlier_dates_3sd, Residual = res_outliers_3sd)
print(res_outlier_info_3sd)

# residual oulier computation
# outlier dates
outlier_dates <- as.Date(c("2006-01-31", "2015-02-28"))

original_temps <- train_data$Avg.Temperature[train_data$Date %in% outlier_dates]
outlier_info <- data.frame(Date = outlier_dates, Original_Temperature = original_temps)
print(outlier_info)

# impute the outliers
train_data_copy <- train_data

# calculate the average temperature for a given month acrros all years
calculate_monthly_avg <- function(data, month, exclude_date) {
  exclude_year <- as.numeric(format(exclude_date, "%Y"))
  same_month_data <- data[format(data$Date, "%m") == month & 
                            format(data$Date, "%Y") != exclude_year, ]
  mean(same_month_data$Avg.Temperature, na.rm = TRUE)
}

# outlier 1: January 31, 2006
jan_avg <- calculate_monthly_avg(train_data_copy, "01", as.Date("2006-01-31"))
train_data_copy$Avg.Temperature[train_data_copy$Date == as.Date("2006-01-31")] <- jan_avg

# outlier 2: February 28, 2015
feb_avg <- calculate_monthly_avg(train_data_copy, "02", as.Date("2015-02-28"))
train_data_copy$Avg.Temperature[train_data_copy$Date == as.Date("2015-02-28")] <- feb_avg

cat("Imputed Temperature for January 31, 2006:", jan_avg, "°C\n")
cat("Imputed Temperature for February 28, 2015:", feb_avg, "°C\n")

cat("Original Temperature for January 31, 2006:", 
    train_data$Avg.Temperature[train_data$Date == as.Date("2006-01-31")], "°C\n")
cat("Original Temperature for February 28, 2015:", 
    train_data$Avg.Temperature[train_data$Date == as.Date("2015-02-28")], "°C\n")

# new time series with the imputed data
train_ts_imputed <- ts(train_data_copy$Avg.Temperature, start = c(1999, 1), frequency = 12)

# fit the modele after impution of the residual outliers
model_1_imputed <- Arima(train_ts_imputed, order = c(1, 0, 1), 
                         seasonal = list(order = c(1, 1, 1), period = 12))
summary(model_1_imputed)

# model diagnostics
checkresiduals(model_1_imputed)
qqnorm(model_1_imputed$residuals)
qqline(model_1_imputed$residuals, col = "red")
acf(model_1_imputed$residuals, lag.max = 60, main = "ACF of Residuals", 
    xlab = "Lag (Months)", xaxt = "n")
axis(1, at = seq(0, 60, by = 6) / 12, labels = seq(0, 60, by = 6))

residuals_1 <- as.numeric(residuals(model_1_imputed))  # original model
residuals_2 <- as.numeric(residuals(model_1))  # imputed model

# Shapiro-Wilk test for model_1 (original)
shapiro_test_1 <- shapiro.test(residuals_1)
cat("Shapiro-Wilk Test for Original Model:\n")
cat("W-statistic:", shapiro_test_1$statistic, "\n")
cat("p-value:", shapiro_test_1$p.value, "\n")

# Shapiro-Wilk test for model_2 (imputed)
shapiro_test_2 <- shapiro.test(residuals_2)
cat("\nShapiro-Wilk Test for Imputed Model:\n")
cat("W-statistic:", shapiro_test_2$statistic, "\n")
cat("p-value:", shapiro_test_2$p.value, "\n")



###################################################################
#-----------------------
model_2 <- Arima(train_ts, order = c(1, 0, 1), seasonal = list(order = c(2, 1, 1), period = 12))
summary(model_2)

# model diagnostics
checkresiduals(model_2)
qqnorm(model_2$residuals)
qqline(model_2$residuals, col = "red")
acf(model_2$residuals, lag.max = 60, main = "ACF of Residuals", 
    xlab = "Lag (Months)", xaxt = "n")
axis(1, at = seq(0, 60, by = 6) / 12, labels = seq(0, 60, by = 6))

residuals_model_2 <- as.numeric(residuals(model_2))

# Shapiro-Wilk test
shapiro_test_model_2 <- shapiro.test(residuals_model_2)
cat("W-statistic:", shapiro_test_model_2$statistic, "\n")
cat("p-value:", shapiro_test_model_2$p.value, "\n")

#-----------------------
model_3 <- Arima(train_ts, order = c(2, 0, 1), seasonal = list(order = c(2, 1, 1), period = 12))
summary(model_3)

# model diagnostics
checkresiduals(model_3)
qqnorm(model_3$residuals)
qqline(model_3$residuals, col = "red")
acf(model_3$residuals, lag.max = 60, main = "ACF of Residuals", 
    xlab = "Lag (Months)", xaxt = "n")
axis(1, at = seq(0, 60, by = 6) / 12, labels = seq(0, 60, by = 6))

residuals_model_3 <- as.numeric(residuals(model_3))

# Shapiro-Wilk test
shapiro_test_model_3 <- shapiro.test(residuals_model_3)
cat("W-statistic:", shapiro_test_model_3$statistic, "\n")
cat("p-value:", shapiro_test_model_3$p.value, "\n")

#------------------------------
model_4 <- Arima(train_ts, order = c(2, 0, 1), seasonal = list(order = c(2, 1, 2), period = 12))
summary(model_4)

# model diagnostics
checkresiduals(model_4)
qqnorm(model_4$residuals)
qqline(model_4$residuals, col = "red")
acf(model_4$residuals, lag.max = 60, main = "ACF of Residuals", 
    xlab = "Lag (Months)", xaxt = "n")
axis(1, at = seq(0, 60, by = 6) / 12, labels = seq(0, 60, by = 6))

#------------------------
# forecast 
forecast_model_2 <- forecast(model_2, h = 24)
plot(forecast_model_2, main = "Toronto Temperature Forecast (2018-2019)", 
     ylab = "Avg Temperature (°C)", xlab = "Year")


# performance metrics
mae_model_2 <- mean(abs(forecast_model_2$mean - test_ts))
mse_model_2 <- mean((forecast_model_2$mean - test_ts)^2)
mape_model_2 <- mean(abs((forecast_model_2$mean - test_ts) / test_ts)) * 100

cat("MAE:", mae_model_2, "\n")
cat("MSE:", mse_model_2, "\n")
cat("MAPE:", mape_model_2, "%\n")


#-----------------------------------
forecast_model_4 <- forecast(model_4, h = 24)
plot(forecast_model_4, main = "Toronto Temperature Forecast (2018-2019)", 
     ylab = "Avg Temperature (°C)", xlab = "Year")


# performance metrics
mae_model_4 <- mean(abs(forecast_model_4$mean - test_ts))
mse_model_4 <- mean((forecast_model_4$mean - test_ts)^2)
mape_model_4 <- mean(abs((forecast_model_4$mean - test_ts) / test_ts)) * 100

cat("MAE:", mae_model_4, "\n")
cat("MSE:", mse_model_4, "\n")
cat("MAPE:", mape_model_4, "%\n")

# Plot actual vs forecasted
plot(test_ts, col = "black", main = "Actual vs Forecasted Temperature (2018-2019)", ylab = "Avg Temperature (°C)", xlab = "Year")
lines(forecast_model_2$mean, col = "blue")
lines(forecast_model_4$mean, col = "red")
legend("topleft", legend = c("Actual", "Model_2", "Model_4"), col = c("black", "blue", "red"), lty = 1)

#-----------------------------------
forecast_2 <- forecast(model_5, h = 24)
plot(forecast_2, main = "Toronto Temperature Forecast (2018-2019)", ylab = "Avg Temperature (°C)", xlab = "Year")


# Calculate performance metrics
mae_2 <- mean(abs(forecast_2$mean - test_ts))
rmse_2 <- sqrt(mean((forecast_2$mean - test_ts)^2))
mape_2 <- mean(abs((forecast_2$mean - test_ts) / test_ts)) * 100

cat("MAE:", mae_2, "\n")
cat("RMSE:", rmse_2, "\n")
cat("MAPE:", mape_2, "%\n")

#-------------------------------------
# plot 95% CI for model_2
time_index <- time(test_ts)  

# 95% CI bounds for both models
lower_2 <- forecast_model_2$lower[, 1]  
upper_2 <- forecast_model_2$upper[, 1]  
lower_4 <- forecast_model_4$lower[, 1]  
upper_4 <- forecast_model_4$upper[, 1] 


par(mar = c(5, 4, 4, 2) + 0.1)  
plot(test_ts, 
     col = "black", 
     main = "Actual Value vs 95% CI SARIMA(1,0,1)(2,1,1)[12]", 
     ylab = "Avg Temperature (°C)", 
     bty = "l",  
     xlab = "",
     xaxt = "n",  
     ylim = range(test_ts, lower_2, upper_2, lower_4, upper_4, na.rm = TRUE))  

points(time_index, test_ts, 
       pch = 19,  
       col = "black", 
       cex = 1.2)  

axis(1, at = seq(2018, 2020, by = 0.25), 
     labels = c("Jan 2018", "Apr 2018", "Jul 2018", "Oct 2018", 
                "Jan 2019", "Apr 2019", "Jul 2019", "Oct 2019", "Jan 2020"), 
     las = 2, cex.axis = 0.8)  

grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")

polygon(c(time_index, rev(time_index)), 
        c(lower_2, rev(upper_2)), 
        col = rgb(0, 0, 1, alpha = 0.2),  
        border = NA)

#---------------------------------------------------------------------------------------------
# plot 95% CI for Model_4
par(mar = c(5, 4, 4, 2) + 0.1)  
plot(test_ts, 
     col = "black", 
     main = "Actual Value vs 95% CI SARIMA(2,0,1)(2,1,2)[12]", 
     ylab = "Avg Temperature (°C)", 
     bty = "l",  
     xlab = "",
     xaxt = "n",  
     ylim = range(test_ts, lower_2, upper_2, lower_4, upper_4, na.rm = TRUE))  

points(time_index, test_ts, 
       pch = 19,  
       col = "black", 
       cex = 1.2)  

axis(1, at = seq(2018, 2020, by = 0.25), 
     labels = c("Jan 2018", "Apr 2018", "Jul 2018", "Oct 2018", 
                "Jan 2019", "Apr 2019", "Jul 2019", "Oct 2019", "Jan 2020"), 
     las = 2, cex.axis = 0.8)  

grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")

polygon(c(time_index, rev(time_index)), 
        c(lower_4, rev(upper_4)), 
        col = rgb(1, 0, 0, alpha = 0.2),  
        border = NA)

############################
library(lmtest)

coeftest(model_2)

coeftest(model_4)
