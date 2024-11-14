"""
待處理:
    - 波動率
    - 無風險利率
    - 信用利差
"""




import numpy as np
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt
import pandas as pd
import chardet
import sys
import os
sys.path.extend(['../', '../../'])
from cb_model import ConvertibleBondPricer

#中文字形
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TaiwanCBStrategy:
    """
    台灣可轉債策略回測類
    用於回測不同定價模型在台灣市場的表現
    """
    
    def __init__(self):
        self.pricer = None
        
    def prepare_data(self, df):
        """
        準備分析所需的資料
        
        Parameters:
        df (pandas.DataFrame): 原始資料DataFrame，包含以下欄位：
        年月: 各年月，該資料庫為月資料，所以為發行期間迄今的每個月
        轉換標的-收盤價: 轉換標的每月最後交易日收盤價
        CB-收盤價: 可轉債最新交易價格
        轉換價格: 在該年月適用的轉換價格
        轉換價值 (1000): 每張轉債可轉換股數*市價
        CB-市值溢價%: ((轉換價值-(CB收盤價 * 面額))/(CB收盤價*面額))*100
        到期日: 該債券的實際到期日
        是否擔保(Y/N): 是否有擔保，有擔保為Y，無擔保為N
        
        Returns:
        pandas.DataFrame: 處理後的分析資料
        """
        # 整理必要欄位並計算關鍵指標
        analysis_data = pd.DataFrame({
            'date': pd.to_datetime(df['年月'].astype(str), format='%Y%m'),
            'stock_price': df['轉換標的-收盤價'].astype(float),
            'cb_price': df['CB-收盤價'].astype(float),
            'conversion_price': df['轉換價格'].astype(float),
            'conversion_value': df['轉換價值 (1000)'].astype(float),
            'market_premium': df['CB-市值溢價%'].astype(float),
            'remaining_years': (pd.to_datetime(df['到期日']) - 
                              pd.to_datetime(df['年月'].astype(str), format='%Y%m')).dt.days / 365.0,
            'is_guaranteed': (df['是否擔保(Y/N)'] == 'Y').astype(int)
        })
        
        # 確保沒有無效值
        analysis_data = analysis_data.fillna(0)
        analysis_data['remaining_years'] = analysis_data['remaining_years'].clip(lower=1e-10)
        analysis_data['implied_vol'] = 0.3  # 設定固定波動率為30%
        
        """
        # [註釋掉的隱含波動率計算部分]
        # 計算隱含波動率
        analysis_data['implied_vol'] = analysis_data.apply(
            lambda x: self.calculate_implied_volatility(
                S=x['stock_price'],
                K=x['conversion_price'],
                T=x['remaining_years'],
                r=0.015,  # 假設無風險利率1.5%
                cb_price=x['cb_price']
            ), axis=1
        )
        """
        
        return analysis_data

    def calculate_implied_volatility(self, S, K, T, r, cb_price, tolerance=0.0001, max_iter=100):
        """
        使用牛頓法計算隱含波動率
        
        Parameters:
        - S: 標的股票價格
        - K: 轉換價格
        - T: 剩餘期限
        - r: 無風險利率
        - cb_price: 可轉債市場價格
        - tolerance: 容忍誤差
        - max_iter: 最大迭代次數
        
        Returns:
        - float: 隱含波動率
        """
        sigma = 0.3  # 初始猜測值
        
        for i in range(max_iter):
            self.pricer = ConvertibleBondPricer(
                S=S, 
                K=K, 
                T=T, 
                r=r, 
                sigma=sigma,
                conversion_ratio=1.0,
                credit_spread=0.02,
                trigger_price=K*1.3  # 假設轉換價格130%為強制贖回價
            )
            
            price = self.pricer.black_scholes_price()
            diff = price - cb_price
            
            if abs(diff) < tolerance:
                return sigma
                
            # 計算vega
            sigma_up = sigma * 1.01
            self.pricer.sigma = sigma_up
            price_up = self.pricer.black_scholes_price()
            vega = (price_up - price) / (sigma_up - sigma)
            
            if vega == 0:
                return sigma
                
            sigma = sigma - diff / vega
            
            if sigma <= 0:
                return 0.001
                
        return sigma

    def evaluate_pricing_models(self, analysis_data):
        """
        評估不同定價模型的表現
        
        Parameters:
        - analysis_data: 處理後的分析資料DataFrame
        
        Returns:
        - DataFrame: 各模型的交易機會和績效指標
        """
        models_evaluation = {}
        
        for index, row in analysis_data.iterrows():
            try:
                self.pricer = ConvertibleBondPricer(
                    S=row['stock_price'],
                    K=row['conversion_price'],
                    T=row['remaining_years'],
                    r=0.015,  # 固定無風險利率1.5%
                    sigma=0.3,  # 使用固定波動率30%
                    conversion_ratio=1.0,
                    credit_spread=0.02,  # 固定信用利差2%
                    trigger_price=row['conversion_price']*1.3  # 轉換價格130%為強制贖回價
                )
                
                models_evaluation[index] = {
                    'BS': self.pricer.black_scholes_price(),
                    'Binomial': self.pricer.binomial_tree_price(),
                    'MonteCarlo': self.pricer.monte_carlo_price(),
                    'JumpDiffusion': self.pricer.jump_diffusion_price(
                        lambda_jump=1.0,
                        mu_jump=-0.1,
                        sigma_jump=0.2,
                        num_sims=1000  # 減少模擬次數以提高速度
                    )
                }
            except Exception as e:
                print(f"Error processing index {index}: {str(e)}")
                models_evaluation[index] = {
                    'BS': row['cb_price'],
                    'Binomial': row['cb_price'],
                    'MonteCarlo': row['cb_price'],
                    'JumpDiffusion': row['cb_price']
                }
        
        model_prices = pd.DataFrame(models_evaluation).T
        
        # 計算各模型的套利機會
        trading_opportunities = pd.DataFrame()
        for model in ['BS', 'Binomial', 'MonteCarlo', 'JumpDiffusion']:
            # 計算定價偏差
            mispricing = model_prices[model] - analysis_data['cb_price']
            # 計算Z分數（標準化偏差）
            z_score = (mispricing - mispricing.rolling(20).mean()) / mispricing.rolling(20).std()
            
            # 生成交易信號
            trading_opportunities[f'{model}_signal'] = np.where(
                z_score > 1.5, -1,  # 過高則賣出
                np.where(z_score < -1.5, 1, 0)  # 過低則買入
            )
            
            # 計算報酬率
            position = trading_opportunities[f'{model}_signal'].shift(1)
            returns = (analysis_data['cb_price'].pct_change() * position).fillna(0)
            
            trading_opportunities[f'{model}_returns'] = returns
            trading_opportunities[f'{model}_cumulative_returns'] = (1 + returns).cumprod()
        
        return trading_opportunities
    
    def plot_results(self, analysis_data, trading_opportunities):
        """
        Plot analysis results
        
        Parameters:
        - analysis_data: Processed analysis data
        - trading_opportunities: Trading opportunities and performance metrics
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        
        # 1. Price and Implied Volatility
        ax1 = axes[0]
        ax1.plot(analysis_data.index, analysis_data['cb_price'], 'b-', label='Market Price')
        ax1.set_ylabel('Price')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(analysis_data.index, analysis_data['implied_vol'], 'r--', label='Implied Volatility')
        ax1_twin.set_ylabel('Implied Volatility')
        ax1.set_title('CB Price and Implied Volatility')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
        
        # 2. Trading Signals
        ax2 = axes[1]
        for model in ['BS', 'Binomial', 'MonteCarlo', 'JumpDiffusion']:
            signals = trading_opportunities[f'{model}_signal']
            # Mark buy points
            buy_points = signals[signals == 1].index
            # Mark sell points
            sell_points = signals[signals == -1].index
            
            ax2.scatter(buy_points, analysis_data.loc[buy_points, 'cb_price'], 
                    marker='^', label=f'{model} Buy')
            ax2.scatter(sell_points, analysis_data.loc[sell_points, 'cb_price'], 
                    marker='v', label=f'{model} Sell')
        
        ax2.plot(analysis_data.index, analysis_data['cb_price'], 'k-', alpha=0.5)
        ax2.set_title('Trading Signals')
        ax2.legend()
        
        # 3. Performance Comparison
        ax3 = axes[2]
        for model in ['BS', 'Binomial', 'MonteCarlo', 'JumpDiffusion']:
            ax3.plot(trading_opportunities.index, 
                    trading_opportunities[f'{model}_cumulative_returns'],
                    label=f'{model}')
        ax3.set_title('Cumulative Returns')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()