# Homework 2

PSTAT 131/231, Fall 2025

# Due on Thursday Nov 6, 2025 at 23:59 pm

Following packages are needed below:

library(tidyverse) library(ISLR) library(ROCR)

Note: If you are working with a partner, please submit only one homework per group with both names. Submit your Rmarkdown (.Rmd) and the compiled pdf or html file.

• Make sure that both group members are in 131 or both in 232.   
• Please indicate if you (or your partner) are in 131 or 231.

# Linear regression (12 pts)

In this problem, we will make use of the Auto data set, which is part of the ISLR package and can be directly accessed by the name Auto once the ISLR package is loaded. The dataset contains 9 variables of 392 observations of automobiles. The qualitative variable origin takes three values: 1, 2, and 3, where 1 stands for American car, 2 stands for European car, and 3 stands for Japanese car.

1. (2 pts) Fit a linear model to the data, in order to predict mpg using all of the other predictors except for name. Present the estimated coefficients. (2 pts) With a 0.01 threshold, comment on whether you can reject the null hypothesis that there is no linear association between mpg with any of the predictors.   
2. (2 pts) Take the whole dataset as training set. What is the training mean squared error of this model? Can you calculate the test mean squared error?   
3. (2 pts) What gas mileage do you predict for an European car with 4 cylinders, displacement 133, horsepower of 117, weight of 3250, acceleration of 29, built in the year 1997? (Be sure to check how year is coded in the dataset).   
4. (1 pts) On average, holding all other features fixed, what is the difference between the mpg of a Japanese car and the mpg of an American car? (1 pts) What is the difference between the mpg of a European car and the mpg of an American car?   
5. (2 pts) On average, holding all other predictor variables fixed, what is the change in mpg associated with a 30-unit increase in horsepower?

# Algae Classification using Logistic regression (15 pts)

The dataset algaeBloom.txt is available on Canvas. Read it in with the following code:

algae <- read_table2("algaeBloom.txt", col_names= c('season','size','speed','mxPH','mnO2','Cl','NO3','NH4', 'oPO4','PO4','Chla','a1','a2','a3','a4','a5','a6','a7'), na="XXXXXXX")

In homework 1, we investigated basic exploratory data analysis for the algaeBloom dataset. One of the explaining variables is a1, which is a numerical attribute. Here, after standardization, we will transform a1 into a categorical variable with 2 levels: high and low, and conduct its classification using those 11 variables (i.e. everything but a1, a2, a3,. . . , a7).

We first improve the normality of the numerical attributes by taking the log of all chemical variables. After log transformation, we impute missing values using the median method. Finally, we transform the variable a1 into a categorical variable with two levels: high if a1 is greater than 5, and low if a1 is smaller than or equal to 5.

algae.transformed $< -$ algae $\% > \%$ mutate_at(vars(4:11), funs(log(.)))   
algae.transformed $< -$ algae.transformed $\% > \%$ mutate_at(vars(4:11),funs(ifelse(is.na(.),median(.,na.rm $\vDash$ TRUE),.)))   
# $a 1 \ = = \ 0$ means low   
algae.transformed $< -$ algae.transformed $\% > \%$ mutate(a1 $=$ factor(as.integer(a1 > 5), levels $=$ c(0, 1)))

Classification Task: We will build classification models to classify a1 into high vs. low using the dataset algae.transformed as above, and evaluate its training error rates and test error rates. We define a new function, named calc_error_rate(), that will calculate misclassification error rate.

calc_error_rate $< -$ function(predicted.value, true.value){ return(mean(true.value ! $! =$ predicted.value)) }

Training/test sets: Split randomly the data set in a train and a test set:

# set.seed(1)

test.indices $=$ sample(1:nrow(algae.transformed), 50) algae.train $\cdot ^ { = }$ algae.transformed[-test.indices,] algae.test $\ l =$ algae.transformed[test.indices,]

In a binary classification problem, let $p$ represent the probability of class label “1”, which implies that $1 - p$ represents probability of class label “ $0$ ”. The logistic function (also called the “inverse logit”) is the cumulative distribution function of logistic distribution, which maps a real number $z$ to the open interval $( 0 , 1 )$ :

$$
p ( z ) = \frac { e ^ { z } } { 1 + e ^ { z } } .
$$

1. (2 pts) Prove that indeed the inverse of a logistic function is the logit function:

$$
z ( p ) = \ln \left( { \frac { p } { 1 - p } } \right) .
$$

2. Assume that $z = \beta _ { 0 } + \beta _ { 1 } x _ { 1 }$ , and $p = \mathrm { l o g i s t i c } ( z )$ . (2 pts) How does the odds of the outcome change if you increase $x _ { 1 }$ by two? (1 pts) Assume $\beta _ { 1 }$ is negative: what value does $p$ approach as $x _ { 1 } \to \infty$ ? ( $1$ pts) What value does $p$ approach as $x _ { 1 } \to - \infty$ ?

3. Use logistic regression to perform classification in the data application above. Logistic regression specifically estimates the probability that an observation as a particular class label. We can define a probability threshold for assigning class labels based on the probabilities returned by the glm fit.

In this problem, we will simply use the “majority rule”. If the probability is larger than $5 0 \%$ class as label “1”. (2 pts) Fit a logistic regression to predict a1 given all other features (excluding a2 to a7) in the dataset using the glm function. (2 pts) Estimate the class labels using the majority rule and (2 pts) calculate the training and test errors using the calc_error_rate defined earlier.

For logistic regression one needs to predict type response predict(glm.obj, test.data, type="response")

4. We will construct ROC curve based on the predictions of the test data from the model we obtained from the logistic regression above. (3 pts) Plot the ROC for the test data for the logistic regression fit. Compute the area under the curve(AUC).

Hints: In order to construct the ROC curves one needs to use the vector of predicted probabilities for the test data. The usage of the function predict() may be different from model to model. For logistic regression one needs to predict type response, see Lab 4.

# Algae Classification using Discriminant Analysis (12 pts)

1. (4 pts) In LDA we assume that $\Sigma _ { 1 } = \Sigma _ { 2 }$ . Use LDA to predict whether a1 is high or low using the MASS::lda() function. The CV argument in the MASS::lda function uses Leave-one-out cross validation LOOCV) when estimating the fitted values to avoid overfitting. Set the CV argument to true, so the program will automatically do cross-validation. Plot an ROC curve for the fitted values.

2. Quadratic discriminant analysis is strictly more flexible than LDA because it is not required that $\Sigma _ { 1 } = \Sigma _ { 2 }$ . In this sense, LDA can be considered a special case of QDA with the covariances constrained to be the same. (2 pts) Use a quadratic discriminant model to predict the a1 using the function MASS::qda. Again setting CV=TRUE and plot the ROC on the same plot as the LDA ROC. (2 pts) Compute the area under the ROC (AUC) for each model. To get the predicted class probabilities look at the value of posterior in the lda and qda objects. (2 pts) Which model has better performance? (2 pts) Briefly explain, in terms of the bias-variance tradeoff, why you believe the better model outperforms the worse model?

# Fundamentals of the bootstrap (10 pts)

In the first part of this problem we will explore to understand the fact that approximately $1 / 3$ of the observations in a bootstrap sample are out-of-bag.

1. (4 pts) Given a sample of size $n$ , what is the probability that any observation $j$ is not in a bootstrap sample? Express your answer as a function of $n$ .   
2. (2 pts) Compute the above probability for $n = 1 0 0 0$ .   
3. (4 pts) Verify that your calculation is reasonable by resampling the numbers 1 to 1000 with replacement and printing the ratio of missing observations. Hint: use the unique and length functions to identify how many unique observations are in the sample. Note that the answer does not have to be exactly the same as what you get in b) due to randomness in sampling.

# Cross-validation estimate of test error (12 pts)

In this problem, we will apply cross-validation to estimate test error rate of logistic regression on the Smarket dataset available in ISLR package. The dataset contains daily percentage returns for the S&P 500 stock index between 2001 and 2005. In particular, the data contains 1250 observations on the following 9 variables:

• Year: The year that the observation was recorded   
• Lag1: Percentage return for previous day   
• Lag2: Percentage return for 2 days previous   
• Lag3: Percentage return for 3 days previous Lag4: Percentage return for 4 days previous   
• Lag5: Percentage return for 5 days previous   
• Volume: Volume of shares traded (number of daily shares traded in billions)   
• Today: Percentage return for today   
• Direction: A factor with levels Down and Up indicating whether the market ha on a given day

We are interested in building a classifier in order to predict Direction using all variables except for Year and Today as predictors. We do the following transformation to convert the factor response into binary values: 0 for Down and 1 for Up.

dat $=$ subset(Smarket, select = -c(Year,Today)) dat\$Direction $=$ ifelse(dat\$Direction $^ { -- }$ "Up", 1,

In this problem, we will again simply use the “majority rule”. If the predicted probability is larger than $5 0 \%$ , classify the observation as 1.

1. (2 pts) Splite dat into a training set of 700 observations, and a test set of the remaining observations. (2 pts) Fit a logistic regression model, on the training data, to predict the Direction using all other variables except for Year and Today as predictors. (2 pts) Calculate the error rate of this model on the test data. Use set.seed(123) in the begining of your answer. 2. (4 pts) Use a 10-fold cross-validation approach on the whole dat to estimate the test error rate. (2 pts) Report the estimated test error rate you obtain. Use set.seed(123) in the begining of your answer.

Just as what we did in Lab4, you can use the following key function to carry out k-fold cross-validation.

do.chunk $< -$ function(chunkid, folddef, dat, ...){ # Get training index train $=$ (folddef! $=$ chunkid) # Get training set and validation set dat.train $=$ dat[train, ] dat.val $=$ dat[-train, ] # Train logistic regression model on training data fit.train $=$ glm(Direction \~ ., family $=$ binomial, data $=$ dat.train) # get predicted value on the validation set pred.val $=$ predict(fit.train, newdata $=$ dat.val, type $=$ "response") pred.val $=$ ifelse(pred.val $>$ .5, 1,0) data.frame(fold $=$ chunkid, val.error $=$ mean(pred.val ! $! =$ dat.val\$Direction))

# Problems below for 231 students only (12 pts)

# Discrinant functions (12 pts)

A multivariate normal distribution has density

$$
f ( x ) = { \frac { 1 } { ( 2 \pi ) ^ { p / 2 } | { \Sigma } | ^ { 1 / 2 } } } e x p \left( - { \frac { 1 } { 2 } } ( x - \mu ) ^ { T } { \Sigma } ^ { - 1 } ( x - \mu ) \right)
$$

In quadratic discriminant analysis with two groups we use Bayes rule to calculate the probability that $Y$ has class label “1”:

$$
P r ( Y = 1 \mid X = x ) = \frac { f _ { 1 } ( x ) \pi _ { 1 } } { \pi _ { 1 } f _ { 1 } ( x ) + \pi _ { 2 } f _ { 2 } ( x ) }
$$

where $\pi _ { 2 } = 1 - \pi _ { 1 }$ is the prior probability of being in group 2. Suppose we classify ${ \hat { Y } } = k$ whenever $P r ( Y = k \mid X =$ $x ) > \tau$ for some probability threshold $\tau$ and that $f _ { k }$ is a multivariate normal density with covariance $\Sigma _ { k }$ and mean $\mu _ { k }$ . Note that for a vector $x$ of length $p$ and a $p \times p$ symmetric matrix $A$ , $x ^ { T } A x$ is the vector quadratic form (the multivariate analog of $x ^ { 2 }$ ). Show that the decision boundary is indeed quadratic by showing that $\hat { Y } = 1$ if

$$
\delta _ { 1 } ( x ) - \delta _ { 2 } ( x ) > M ( \tau )
$$

where

$$
\delta _ { k } ( x ) = - { \frac { 1 } { 2 } } ( x - \mu _ { k } ) ^ { T } \Sigma _ { k } ^ { - 1 } ( x - \mu _ { k } ) - { \frac { 1 } { 2 } } \log | \Sigma _ { k } | + \log \pi _ { k }
$$

and $M ( \tau )$ is some function of the probability threshold $\tau$ . What is the decision threshold, $\mathrm { M } ( 1 / 2 )$ , corresponding to a probability threshold of $1 / 2$ ?


# 解题方案（不含代码）

## 线性回归（12 分）

1) 拟合与整体/个体显著性（2 分）
- 模型设定：以 `mpg` 为响应变量，自变量包含除 `name` 外的所有变量，并将 `origin` 作为分类因子（基准水平通常为 American=1）。
- 估计系数：给出截距与各自变量（含 `origin` 的虚拟变量）的点估计与标准误。
- 显著性结论（阈值 0.01）：
  - 整体线性关联：使用整体 F 检验（比较含全部自变量的模型与仅含截距模型），若 p 值 < 0.01，拒绝“无任何线性关联”的原假设。
  - 个体显著性：查看各系数的 t 检验 p 值，p < 0.01 判为在 1% 水平显著；注意多重比较语境下可以说明“在传统逐项检验意义下”。

2) 训练 MSE 与测试 MSE（2 分）
- 训练 MSE：基于整数据集的残差平方和除以观测数计算。
- 测试 MSE：在仅使用整数据作为训练的设定下，无法直接得到独立测试误差；可补充说明用留出集或交叉验证估计测试误差（但本问不要求实际计算）。

3) 特定样本的预测（2 分）
- 样本特征：`origin=European`（即 `origin=2`）、`cylinders=4`、`displacement=133`、`horsepower=117`、`weight=3250`、`acceleration=29`、`year=1997`。
- 年份编码核对：`Auto` 数据集的 `year` 通常编码为 70–82（代表 1970–1982）。因此“1997”超出数据编码范围；需按数据实际编码体系设定输入（例如若要代表 1977，输入 77）。
- 预测方法：将上述（经正确编码的）协变量代入线性模型的预测式 \(\hat{y}=\hat{\beta}_0+\sum_j \hat{\beta}_j x_j\)，并包含相应的 `origin` 虚拟变量项，得到预测 `mpg`。

4) 不同产地的平均差异（2 分）
- 约定 American=1 为基准水平，则：
  - Japanese 与 American 的平均差异：等于 `origin=3` 的虚拟变量系数（保持其他变量固定）。
  - European 与 American 的平均差异：等于 `origin=2` 的虚拟变量系数（保持其他变量固定）。

5) `horsepower` 增加 30 的影响（2 分）
- 在线性模型下，其他变量固定，`horsepower` 每增加 1 单位，`mpg` 期望改变量为 \(\hat{\beta}_{horsepower}\)。
- 因此增加 30 单位时，期望改变量为 \(30\,\hat{\beta}_{horsepower}\)（注意若系数为负，则为下降）。


## 使用逻辑回归的藻类分类（15 分）

数据预处理与任务：对化学变量 4–11 取对数、以中位数法填补缺失；将 \(a1>5\) 记为“高”（1），否则“低”（0）；拆分训练/测试集（测试集 50 条）。

1) 逻辑函数的反函数（2 分）
- 已知 \(p(z)=\dfrac{e^{z}}{1+e^{z}}\)。证明 logit 形式：
\[ z(p)=\ln\!\left(\frac{p}{1-p}\right). \]
- 证明思路：由 \(p=\frac{e^{z}}{1+e^{z}}\Rightarrow \frac{p}{1-p}=e^{z}\Rightarrow z=\ln\!\left(\frac{p}{1-p}\right)\)。

2) 几率变换与极限（共 4 分）
- 设 \(z=\beta_0+\beta_1 x_1\), \(p=\text{logistic}(z)\)。当 \(x_1\) 增加 2：
  - 几率变为 \(\dfrac{p'}{1-p'}=\exp(\beta_0+\beta_1(x_1+2))=e^{2\beta_1}\cdot \exp(\beta_0+\beta_1 x_1)=e^{2\beta_1}\cdot \dfrac{p}{1-p}\)。
  - 即几率按因子 \(e^{2\beta_1}\) 成倍变化（2 分）。
- 若 \(\beta_1<0\)：
  - 当 \(x_1\to\infty\), \(z\to -\infty\Rightarrow p\to 0\)（1 分）。
  - 当 \(x_1\to -\infty\), \(z\to +\infty\Rightarrow p\to 1\)（1 分）。

3) 逻辑回归建模与误差（6 分）
- 以处理后的 `algae.transformed` 为数据，使用除 `a2`–`a7` 外的全部特征预测二分类 `a1`。
- 用阈值 0.5 的“多数规则”将概率转为类别标签。
- 计算训练与测试的误分类率：\(\text{err}=\text{mean}(\hat{y}\neq y)\)。
- 报告：给出训练误差与测试误差，并简述可能的过拟合/欠拟合征象（训练误差远低于测试误差提示过拟合）。

4) ROC 与 AUC（3 分）
- 使用测试集的预测概率绘制 ROC 曲线（以 TPR 对 FPR），并计算 AUC；
- 说明：AUC 取值越大（接近 1）表示区分能力越强；若接近 0.5 则接近随机分类。


## 使用判别分析的藻类分类（12 分）

1) LDA + LOOCV 与 ROC（4 分）
- 设定 LDA：`a1` 为响应；特征与逻辑回归相同；`CV=TRUE` 进行留一交叉验证获取拟合值。
- 使用交叉验证预测得到的评分/后验概率绘制 ROC，并报告 AUC。
- 讨论：若类条件协方差近似相等且样本量有限，LDA 往往更稳健。

2) QDA、与 LDA 同图比较及 AUC、模型对比（8 分）
- 设定 QDA：同上，`CV=TRUE` 获取交叉验证预测。
- 将 LDA 与 QDA 的 ROC 画在同一图中，并分别给出 AUC；
- 选择更优模型：以更高的 AUC/更低的错误率为准；
- 偏差-方差权衡解释：
  - QDA 更灵活（较低偏差，较高方差），在样本量有限或真实协方差相近时，可能过拟合，AUC 反而不如 LDA；
  - 若真实协方差差异显著且样本充足，QDA 可优于 LDA。


## 自助法（Bootstrap）基础（10 分）

1) 某观测不在 bootstrap 样本中的概率（4 分）
- 每次有放回抽样从 \(n\) 个观测中抽取 \(n\) 次。任一固定观测在一次抽取中不被选中的概率为 \(1-\tfrac{1}{n}\)。
- 因为独立重复 \(n\) 次，故不被选中的概率为 \((1-\tfrac{1}{n})^{n}\)。

2) 当 \(n=1000\)（2 分）
- 数值近似：\((1-\tfrac{1}{1000})^{1000}\approx e^{-1}\approx 0.3679\)。

3) 经验验证（4 分）
- 从 1–1000 之间有放回抽取 1000 次，统计未出现的不同整数个数/1000 的比例；
- 期望该比例接近 \(\approx 0.368\)，考虑到抽样随机性会有小幅波动。


## 交叉验证估计测试误差（12 分）

数据：`ISLR::Smarket`，去除 `Year` 与 `Today`，将 `Direction` 转换为二元（Down=0, Up=1），以其为响应变量。

1) 留出集评估（6 分）
- 设随机种子（如 123），随机抽取 700 条为训练集，余下为测试集；
- 在训练集上拟合逻辑回归模型（使用其余变量作为自变量），在测试集上输出概率并用 0.5 阈值转为类别；
- 计算并报告测试错误率；可简述：若特征弱、信噪比低，错误率可能接近 0.45–0.5。

2) 10 折交叉验证（6 分）
- 固定随机种子（如 123），将全体样本随机分为 10 折；
- 逐折作为验证集，其余 9 折为训练集，拟合逻辑回归并在该折上预测，记录验证错误率；
- 汇总 10 折验证错误率的平均值作为测试误差的交叉验证估计并报告；
- 对比留出集与 10 折 CV 结果：CV 方差更小、更稳定。


## 仅 231 学生：判别函数（12 分）

目标：证明决策边界为二次型，且存在阈值函数 \(M(\tau)\) 使得 \(\hat{Y}=1\) 当且仅当
\[ \delta_1(x) - \delta_2(x) > M(\tau). \]

推导要点：
- 后验概率：\(\Pr(Y=1\mid X=x)=\dfrac{\pi_1 f_1(x)}{\pi_1 f_1(x)+\pi_2 f_2(x)}\)。阈值判别 \(>\tau\) 等价于
\[
 (1-\tau)\,\pi_1 f_1(x) > \tau\,\pi_2 f_2(x)
 \;\Longleftrightarrow\;
 \log f_1(x)+\log\pi_1 - \log f_2(x)-\log\pi_2 > \log\!\left(\frac{\tau}{1-\tau}\right).
\]
- 对于多元正态密度，\(\log f_k(x)=-\tfrac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k) - \tfrac{1}{2}\log|\Sigma_k| - \tfrac{p}{2}\log(2\pi)\)。常数项相消后，得到
\[ \delta_k(x) = -\tfrac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k) - \tfrac{1}{2}\log|\Sigma_k| + \log\pi_k, \]
从而
\[ \delta_1(x) - \delta_2(x) > M(\tau),\quad M(\tau)=\log\!\left(\frac{\tau}{1-\tau}\right). \]
- 因 \(\delta_k(x)\) 含二次型 \(x^T\Sigma_k^{-1}x\)，决策边界为二次曲面。

当 \(\tau=\tfrac{1}{2}\) 时：
\[ M\!\left(\tfrac{1}{2}\right)=\log 1=0, \]
即以 \(\delta_1(x)-\delta_2(x)>0\) 为判别阈值。
