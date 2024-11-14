import numpy as np
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt
import pandas as pd
import chardet

class ConvertibleBondPricer:
    """
    可轉換債券定價模型
    包含Black-Scholes模型、二叉樹模型、蒙特卡洛模擬和跳躍擴散模型
    """
    
    def __init__(self, S, K, T, r, sigma, conversion_ratio, credit_spread, trigger_price):
        """
        初始化參數
        S: 標的股票現價
        K: 轉換價格
        T: 剩餘期限(年)
        r: 無風險利率
        sigma: 波動率
        conversion_ratio: 轉換比率
        credit_spread: 信用利差
        trigger_price: 觸發價格
        """
        # 確保所有輸入參數為float類型
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.conversion_ratio = float(conversion_ratio)
        self.credit_spread = float(credit_spread)
        self.trigger_price = float(trigger_price)

    def black_scholes_price(self):
        """
        使用Black-Scholes模型定價
        考慮強制轉換條款
        返回: 可轉債理論價格
        """
        # 檢查參數有效性
        if self.T <= 0 or self.sigma <= 0:
            return self.S * self.conversion_ratio if self.S >= self.trigger_price else self.K

        # 如果股價已超過觸發價格，直接返回轉換價值
        if self.S >= self.trigger_price:
            return self.S * self.conversion_ratio

        # 計算BS參數
        d1 = (np.log(self.S/self.K) + (self.r + self.sigma**2/2)*self.T) / \
             (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        
        # 計算強制轉換機率
        h1 = (np.log(self.trigger_price/self.S) + 
              (self.r + 0.5*self.sigma**2)*self.T)/(self.sigma*np.sqrt(self.T))
        h2 = h1 - self.sigma*np.sqrt(self.T)
        prob_mandatory = norm.cdf(h1) + \
                        np.exp(2*self.r*np.log(self.trigger_price/self.S)/self.sigma**2)*norm.cdf(-h2)
        prob_mandatory = min(prob_mandatory, 1.0)

        # 計算組件價值
        bond_value = self.K * np.exp(-(self.r + self.credit_spread)*self.T) * (1 - prob_mandatory)
        option_value = self.conversion_ratio * (
            self.S * norm.cdf(d1) - 
            self.K * np.exp(-self.r*self.T) * norm.cdf(d2)
        ) * (1 - prob_mandatory)
        mandatory_value = self.S * self.conversion_ratio * prob_mandatory
        
        return bond_value + option_value + mandatory_value

    def binomial_tree_price(self, steps=100):
        """
        使用二叉樹模型定價
        steps: 二叉樹步數
        返回: 可轉債理論價格
        """
        # 檢查參數有效性
        if self.T <= 0 or self.sigma <= 0:
            return self.S * self.conversion_ratio if self.S >= self.trigger_price else self.K

        # 確保dt不為0
        dt = max(self.T/steps, 1e-10)
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.r*dt) - d)/(u - d)
        
        # 限制概率在[0,1]範圍內
        p = max(0, min(1, p))
        
        # 創建股價樹和期權樹
        stock_tree = np.zeros((steps+1, steps+1))
        option_tree = np.zeros((steps+1, steps+1))
        
        # 初始化股價樹
        for j in range(steps+1):
            stock_tree[j,j] = self.S * (u**j) * (d**(steps-j))
            
            # 在末節點計算可轉債價值
            if stock_tree[j,j] >= self.trigger_price:
                option_tree[j,j] = stock_tree[j,j] * self.conversion_ratio
            else:
                option_tree[j,j] = max(
                    self.K,  # 債券價值
                    stock_tree[j,j] * self.conversion_ratio  # 轉換價值
                )
        
        # 向後遞迴
        for i in range(steps-1, -1, -1):
            for j in range(i+1):
                stock_price = self.S * (u**j) * (d**(i-j))
                
                if stock_price >= self.trigger_price:
                    option_tree[j,i] = stock_price * self.conversion_ratio
                else:
                    hold_value = np.exp(-(self.r + self.credit_spread)*dt) * \
                                (p * option_tree[j+1,i+1] + (1-p) * option_tree[j,i+1])
                    convert_value = stock_price * self.conversion_ratio
                    option_tree[j,i] = max(hold_value, convert_value)
        
        return option_tree[0,0]

    def monte_carlo_price(self, num_sims=10000, num_steps=100):
        """
        使用蒙特卡洛模擬定價
        num_sims: 模擬路徑數
        num_steps: 每條路徑的時間步數
        返回: 可轉債理論價格
        """
        # 檢查參數有效性
        if self.T <= 0 or self.sigma <= 0:
            return self.S * self.conversion_ratio if self.S >= self.trigger_price else self.K

        dt = max(self.T/num_steps, 1e-10)  # 確保dt不為0
        prices = np.zeros(num_sims)
        
        for i in range(num_sims):
            price_path = np.zeros(num_steps+1)
            price_path[0] = self.S
            
            # 生成價格路徑
            for t in range(1, num_steps+1):
                z = np.random.standard_normal()
                price_path[t] = price_path[t-1] * np.exp(
                    (self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*z
                )
                
                # 檢查是否觸發強制轉換
                if price_path[t] >= self.trigger_price:
                    prices[i] = price_path[t] * self.conversion_ratio
                    break
                    
                # 如果到達最後時間點
                if t == num_steps:
                    prices[i] = max(
                        self.K * np.exp(-(self.r + self.credit_spread)*self.T),
                        price_path[t] * self.conversion_ratio
                    )
        
        return np.mean(prices)

    def jump_diffusion_price(self, lambda_jump=1.0, mu_jump=-0.1, sigma_jump=0.2, num_sims=10000, num_steps=100):
        """
        使用Merton跳躍擴散模型定價
        
        Parameters:
        lambda_jump: 跳躍強度 (平均每年跳躍次數)
        mu_jump: 平均跳躍大小
        sigma_jump: 跳躍大小波動率
        num_sims: 模擬路徑數
        num_steps: 每條路徑的時間步數
        返回: 可轉債理論價格
        """
        # 檢查參數有效性
        if self.T <= 0 or self.sigma <= 0:
            return self.S * self.conversion_ratio if self.S >= self.trigger_price else self.K

        dt = max(self.T/num_steps, 1e-10)  # 確保dt不為0
        sqrt_dt = np.sqrt(dt)
        
        # 初始化數組
        S = np.zeros((num_sims, num_steps + 1))
        S[:, 0] = self.S
        
        # 計算跳躍補償項
        k = np.exp(mu_jump + 0.5 * sigma_jump**2) - 1
        # 調整漂移項以確保風險中性定價
        adjusted_drift = self.r - lambda_jump * k - 0.5 * self.sigma**2
        
        # 生成擴散項
        dW = np.random.standard_normal((num_sims, num_steps))
        
        # 生成跳躍項
        lambda_dt = max(lambda_jump * dt, 0)  # 確保lambda_dt為非負
        n_jumps = np.random.poisson(lambda_dt, size=(num_sims, num_steps))
        jump_sizes = np.random.lognormal(
            mean=mu_jump - 0.5 * sigma_jump**2,
            sigma=sigma_jump,
            size=(num_sims, num_steps)
        )
        jumps = np.log(jump_sizes) * n_jumps
        
        # 模擬路徑
        for t in range(num_steps):
            # 擴散 + 跳躍項
            S[:, t+1] = S[:, t] * np.exp(
                adjusted_drift * dt +           # 漂移項
                self.sigma * sqrt_dt * dW[:, t] +  # 擴散項
                jumps[:, t]                     # 跳躍項
            )
            
            # 檢查強制轉換
            converted = S[:, t+1] >= self.trigger_price
            if np.any(converted):
                S[converted, t+1:] = S[converted, t+1:t+2]
        
        # 計算收益
        payoff = np.zeros(num_sims)
        
        # 對觸及觸發價格的路徑（強制轉換）
        converted = np.any(S >= self.trigger_price, axis=1)
        payoff[converted] = S[converted, -1] * self.conversion_ratio
        
        # 對未觸及觸發價格的路徑
        not_converted = ~converted
        conversion_value = S[not_converted, -1] * self.conversion_ratio
        bond_value = self.K * np.exp(-(self.r + self.credit_spread) * self.T)
        payoff[not_converted] = np.maximum(conversion_value, bond_value)
        
        # 計算平均折現收益
        price = np.mean(payoff) * np.exp(-self.r * self.T)

        return price

    def compare_methods_with_jumps(self):
        """
        比較不同定價方法的結果
        返回: 各方法計算的價格字典
        """
        methods = {
            'Black-Scholes': self.black_scholes_price(),
            'Binomial Tree': self.binomial_tree_price(),
            'Monte Carlo': self.monte_carlo_price(),
            'Jump Diffusion': self.jump_diffusion_price()
        }
        return methods

    def plot_comparison_with_jumps(self, S_range=None):
        """
        繪製不同定價方法的價格比較圖
        S_range: 股價範圍，如果未提供則使用默認範圍
        """
        if S_range is None:
            S_range = np.linspace(self.S * 0.5, self.S * 1.5, 50)
            
        bs_prices = []
        bin_prices = []
        mc_prices = []
        jd_prices = []
        
        original_S = self.S
        for s in S_range:
            self.S = s
            bs_prices.append(self.black_scholes_price())
            bin_prices.append(self.binomial_tree_price())
            mc_prices.append(self.monte_carlo_price())
            jd_prices.append(self.jump_diffusion_price())
        self.S = original_S
        
        plt.figure(figsize=(12, 6))
        plt.plot(S_range, bs_prices, 'b-', label='Black-Scholes')
        plt.plot(S_range, bin_prices, 'g--', label='Binomial Tree')
        plt.plot(S_range, mc_prices, 'r:', label='Monte Carlo')
        plt.plot(S_range, jd_prices, 'm-.', label='Jump Diffusion')
        plt.axvline(x=self.trigger_price, color='k', linestyle=':', label='Trigger Price')
        plt.xlabel('Stock Price')
        plt.ylabel('CB Price')
        plt.title('Convertible Bond Price Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()