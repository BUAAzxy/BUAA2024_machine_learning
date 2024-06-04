import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成虚拟数据
np.random.seed(42)
N = 1000
num_males = int(N * 3 / 5)
num_females = N - num_males
male_heights = np.random.normal(176, 8, num_males)
female_heights = np.random.normal(164, 6, num_females)
heights = np.concatenate((male_heights, female_heights))
np.random.shuffle(heights)

# EM算法实现
def gaussian(x, mu, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) ** 2 / sigma ** 2))

def e_step(data, mu_M, sigma_M, mu_F, sigma_F, pi_M, pi_F):
    r_M = pi_M * gaussian(data, mu_M, sigma_M)
    r_F = pi_F * gaussian(data, mu_F, sigma_F)
    gamma_M = r_M / (r_M + r_F)
    gamma_F = r_F / (r_M + r_F)
    return gamma_M, gamma_F

def m_step(data, gamma_M, gamma_F):
    N_M = np.sum(gamma_M)
    N_F = np.sum(gamma_F)
    
    mu_M = np.sum(gamma_M * data) / N_M
    mu_F = np.sum(gamma_F * data) / N_F
    
    sigma_M = np.sqrt(np.sum(gamma_M * (data - mu_M) ** 2) / N_M)
    sigma_F = np.sqrt(np.sum(gamma_F * (data - mu_F) ** 2) / N_F)
    
    pi_M = N_M / len(data)
    pi_F = N_F / len(data)
    
    return mu_M, sigma_M, mu_F, sigma_F, pi_M, pi_F

# 初始化参数
mu_M, sigma_M = 170, 10
mu_F, sigma_F = 160, 10
pi_M, pi_F = 0.5, 0.5

# 迭代
max_iter = 100
tol = 1e-6

for i in range(max_iter):
    gamma_M, gamma_F = e_step(heights, mu_M, sigma_M, mu_F, sigma_F, pi_M, pi_F)
    mu_M_new, sigma_M_new, mu_F_new, sigma_F_new, pi_M_new, pi_F_new = m_step(heights, gamma_M, gamma_F)
    
    if np.abs(mu_M - mu_M_new) < tol and np.abs(mu_F - mu_F_new) < tol:
        break
    
    mu_M, sigma_M, mu_F, sigma_F, pi_M, pi_F = mu_M_new, sigma_M_new, mu_F_new, sigma_F_new, pi_M_new, pi_F_new

print(f'Estimated mu_M: {mu_M}, sigma_M: {sigma_M}, pi_M: {pi_M}')
print(f'Estimated mu_F: {mu_F}, sigma_F: {sigma_F}, pi_F: {pi_F}')

# 可视化
x = np.linspace(140, 200, 1000)
pdf_males = pi_M * gaussian(x, mu_M, sigma_M)
pdf_females = pi_F * gaussian(x, mu_F, sigma_F)

plt.figure(figsize=(12, 6))
sns.histplot(heights, bins=30, kde=False, color='orange', stat='density', label='Data')
plt.plot(x, pdf_males, 'r', label=f'Male Gaussian ($\mu$={mu_M:.2f}, $\sigma$={sigma_M:.2f})')
plt.plot(x, pdf_females, 'b', label=f'Female Gaussian ($\mu$={mu_F:.2f}, $\sigma$={sigma_F:.2f})')
plt.plot(x, pdf_males + pdf_females, 'g', label='Combined Gaussian')
plt.xlabel('Height (cm)')
plt.ylabel('Density')
plt.title('Height Distribution with Gaussian Mixture Model')
plt.legend()
plt.show()
