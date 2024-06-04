import numpy as np
import matplotlib.pyplot as plt

# 高斯函数
def gaussian(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5 * ((x - mu)/sigma)**2)

# 生成合成数据
np.random.seed(0)
N = 1000
mu_M, sigma_M = 176, 8
mu_F, sigma_F = 164, 6
male_data = np.random.normal(mu_M, sigma_M, int(3*N/5))
female_data = np.random.normal(mu_F, sigma_F, int(2*N/5))
data = np.concatenate((male_data, female_data))

# 原始高斯分量
x = np.linspace(130, 200, 1000)
pdf_M_original = gaussian(x, mu_M, sigma_M)
pdf_F_original = gaussian(x, mu_F, sigma_F)

# EM算法
def run_em_algorithm(num_iterations):
    # 初始化参数
    mu_M_hat, mu_F_hat = np.random.uniform(160, 180), np.random.uniform(150, 170)
    sigma_M_hat, sigma_F_hat = np.random.uniform(5, 10), np.random.uniform(4, 8)
    pi_M_hat, pi_F_hat = 0.5, 0.5
    mu_M_values = []  # 存储每次迭代的mu_M值
    
    # EM算法
    for _ in range(num_iterations):
        # E步
        w_M = pi_M_hat * gaussian(data, mu_M_hat, sigma_M_hat)
        w_F = pi_F_hat * gaussian(data, mu_F_hat, sigma_F_hat)
        sum_w = w_M + w_F
        w_M /= sum_w
        w_F /= sum_w
        
        # M步
        mu_M_hat = np.sum(w_M * data) / np.sum(w_M)
        mu_F_hat = np.sum(w_F * data) / np.sum(w_F)
        sigma_M_hat = np.sqrt(np.sum(w_M * (data - mu_M_hat)**2) / np.sum(w_M))
        sigma_F_hat = np.sqrt(np.sum(w_F * (data - mu_F_hat)**2) / np.sum(w_F))
        pi_M_hat = np.mean(w_M)
        pi_F_hat = np.mean(w_F)
        
        mu_M_values.append(mu_M_hat)  # 记录此次迭代的mu_M值
    
    # 输出结果
    print(f"Iterations: {num_iterations}")
    print("Estimated Parameters:")
    print("mu_M:", mu_M_hat)
    print("sigma_M:", sigma_M_hat)
    print("mu_F:", mu_F_hat)
    print("sigma_F:", sigma_F_hat)
    print("pi_M:", pi_M_hat)
    print("pi_F:", pi_F_hat)
    
    # 绘图
    plt.hist(data, bins=30, density=True, alpha=0.5, color='blue', label='Original Data')
    x = np.linspace(130, 200, 1000)
    pdf_M = gaussian(x, mu_M_hat, sigma_M_hat)
    pdf_F = gaussian(x, mu_F_hat, sigma_F_hat)
    pdf = pi_M_hat * pdf_M + pi_F_hat * pdf_F
    plt.plot(x, pdf_M_original, color='blue', linestyle='-', label='Original Male Gaussian')
    plt.plot(x, pdf_F_original, color='blue', linestyle='-', label='Original Female Gaussian')
    plt.plot(x, pdf_M, color='red', linestyle='--', label='Estimated Male Gaussian')
    plt.plot(x, pdf_F, color='green', linestyle='--', label='Estimated Female Gaussian')
    plt.plot(x, pdf, color='purple', linestyle='-.', label='Estimated Mixture Gaussian')
    plt.title('Epoch:{},EM Algorithm for Mixture Gaussian Model'.format(num_iterations))
    plt.xlabel('Height')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

    # Plot mu_M values over iterations
    plt.plot(range(1, num_iterations + 1), mu_M_values, marker='o', linestyle='-')
    plt.title('Variation of mu_M over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('mu_M')
    plt.grid(True)
    plt.show()

# 不同迭代次数下运行EM算法
for num_iterations in range(100,301,50):
    run_em_algorithm(num_iterations)
